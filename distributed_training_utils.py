import torch, torchvision
import numpy as np
import itertools as it
import re
from math import sqrt
import random

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time

import compression_utils as comp

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def copy(target, source):
  for name in target:
    target[name].data = source[name].data.clone()

def add(target, source):
  for name in target:
    target[name].data += source[name].data.clone()

def scale(target, scaling):
  for name in target:
    target[name].data = scaling*target[name].data.clone()

def subtract(target, source):
  for name in target:
    target[name].data -= source[name].data.clone()

def subtract_(target, minuend, subtrahend):
  for name in target:
    target[name].data = minuend[name].data.clone()-subtrahend[name].data.clone()

def average(target, sources):
  for name in target:
    target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()

def weighted_average(target, sources, weights):
  for name in target:
    summ = torch.sum(weights)
    n = len(sources)
    modify = [weight/summ*n for weight in weights]
    target[name].data = torch.mean(torch.stack([m*source[name].data for source, m in zip(sources, modify)]), dim=0).clone()

def majority_vote(target, sources, lr):
  for name in target:
    threshs = torch.stack([torch.max(source[name].data) for source in sources])
    mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
    target[name].data = (lr*mask).clone()

def compress(target, source, compress_fun):
  '''
  compress_fun : a function f : tensor (shape) -> tensor (shape)
  '''
  for name in target:
    target[name].data = compress_fun(source[name].data.clone())


      
class DistributedTrainingDevice(object):
  '''
  A distributed training device (Client or Server)
  data : a pytorch dataset consisting datapoints (x,y)
  model : a pytorch neural net f mapping x -> f(x)=y_
  hyperparameters : a python dict containing all hyperparameters
  '''
  def __init__(self, dataloader, model, hyperparameters, experiment):
    self.hp = hyperparameters
    self.xp = experiment
    self.loader = dataloader
    self.model = model
    self.loss_fn = nn.CrossEntropyLoss()


class Client(DistributedTrainingDevice):

  def __init__(self, dataloader, model, hyperparameters, experiment, id_num=0):
    super().__init__(dataloader, model, hyperparameters, experiment)

    self.id = id_num

    # Parameters
    self.W = {name : value for name, value in self.model.named_parameters()}
    self.W_old = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.dW = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.dW_compressed = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.A = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

    self.n_params = sum([T.numel() for T in self.W.values()])
    self.bits_sent = []

    # Optimizer (specified in self.hp, initialized using the suitable parameters from self.hp)
    optimizer_object = getattr(optim, self.hp['optimizer'])
    optimizer_parameters = {k : v for k, v in self.hp.items() if k in optimizer_object.__init__.__code__.co_varnames}

    self.optimizer = optimizer_object(self.model.parameters(), **optimizer_parameters)

    # Learning Rate Schedule
    self.scheduler = getattr(optim.lr_scheduler, self.hp['lr_decay'][0])(self.optimizer, **self.hp['lr_decay'][1])

    # State
    self.epoch = 0
    self.train_loss = 0.0


  def synchronize_with_server(self, server):
    # W_client = W_server
    copy(target=self.W, source=server.W)


  def train_cnn(self, iterations):

    running_loss = 0.0
    for i in range(iterations):
      
      try: # Load new batch of data
        x, y = next(self.epoch_loader)
      except: # Next epoch
        self.epoch_loader = iter(self.loader)
        self.epoch += 1

        # Adapt lr according to schedule
        if isinstance(self.scheduler, optim.lr_scheduler.LambdaLR):
          self.scheduler.step()
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau) and 'loss_test' in self.xp.results:
          self.scheduler.step(self.xp.results['loss_test'][-1])
        
        x, y = next(self.epoch_loader)

      x, y = x.to(device), y.to(device)
        
      # zero the parameter gradients
      self.optimizer.zero_grad()

      # forward + backward + optimize
      y_ = self.model(x)

      loss = self.loss_fn(y_, y)
      loss.backward()
      self.optimizer.step()
      
      running_loss += loss.item()

    return running_loss / iterations


  def compute_weight_update(self, iterations=1):

    # Training mode
    self.model.train()

    # W_old = W
    copy(target=self.W_old, source=self.W)
    
    # W = SGD(W, D)
    self.train_loss = self.train_cnn(iterations)

    # dW = W - W_old
    subtract_(target=self.dW, minuend=self.W, subtrahend=self.W_old)

  
  def compress_weight_update_up(self, compression=None, accumulate=False, count_bits=False):

    if accumulate and compression[0] != "none":
      # compression with error accumulation     
      add(target=self.A, source=self.dW)
      compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
      subtract(target=self.A, source=self.dW_compressed)

    else: 
      # compression without error accumulation
      compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

    if count_bits:
      # Compute the update size
      self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]



class Server(DistributedTrainingDevice):

  def __init__(self, dataloader, model, hyperparameters, experiment, stats):
    super().__init__(dataloader, model, hyperparameters, experiment)

    # Parameters
    self.W = {name : value for name, value in self.model.named_parameters()}
    self.dW_compressed = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.dW = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

    self.A = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}

    self.n_params = sum([T.numel() for T in self.W.values()])
    self.bits_sent = []

    # self.client_sizes = torch.Tensor(stats["split"]).cuda()
    self.client_sizes = torch.tensor(stats["split"], dtype=torch.float32, device=device)

    # Server optimizer state for SR-FedAdam (server-side only)
    self.server_step = 0
    self.m = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    self.v = {name : torch.zeros(value.shape).to(device) for name, value in self.W.items()}
    # EMA fallback for sigma if requested
    self.sigma_ema = {name : torch.zeros(1).to(device) for name, value in self.W.items()}


  def aggregate_weight_updates(self, clients, aggregation="mean"):

    # dW = aggregate(dW_i, i=1,..,n)
    if aggregation == "mean":
      average(target=self.dW, sources=[client.dW_compressed for client in clients])

    elif aggregation == "weighted_mean":
      weighted_average(target=self.dW, sources=[client.dW_compressed for client in clients], 
        weights=torch.stack([self.client_sizes[client.id] for client in clients]))
    
    elif aggregation == "majority":
      majority_vote(target=self.dW, sources=[client.dW_compressed for client in clients], lr=self.hp["lr"])

    # Apply server-side optimizer if configured
    so = self.hp.get("server_optimizer", "none")
    if so == "sr_fedadam":
      self._apply_sr_fedadam(clients)
    elif so == "fedadam":
      self._apply_fedadam(clients)


  def _apply_fedadam(self, clients):
    # Server-side Adam-like aggregator: update moments and scale aggregated update
    beta1 = self.hp.get('server_beta1', 0.9)
    beta2 = self.hp.get('server_beta2', 0.999)
    eps = self.hp.get('server_eps', 1e-8)
    lr = self.hp.get('server_lr', None)
    if lr is None:
      lr = self.hp.get('lr', 0.0)

    # Update biased first and second moments using aggregated delta
    for name in self.dW:
      delta = self.dW[name].to(device)
      self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * delta
      self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (delta.pow(2))

    # Apply adaptive scaling (FedAdam style)
    try:
      for name in self.dW:
        denom = torch.sqrt(self.v[name]) + eps
        self.dW[name] = (lr * self.dW[name]) / denom
    except Exception:
      pass


  def _compute_inter_client_sigma2_block(self, clients, name, delta_block):
    # Compute scalar sigma^2 for a parameter block as mean of squared norms across clients
    if len(clients) == 0:
      return torch.tensor(0.0).to(device)
    per_client_vals = []
    for client in clients:
      diff = client.dW_compressed[name].to(device) - delta_block
      per_client_vals.append(torch.sum(diff.pow(2)))
    return torch.stack(per_client_vals).mean()


  def _apply_sr_fedadam(self, clients):
    # SR-FedAdam: apply Stein shrinkage to aggregated update self.dW
    beta1 = self.hp.get('server_beta1', 0.9)
    beta2 = self.hp.get('server_beta2', 0.999)
    eps = self.hp.get('server_eps', 1e-8)
    lr = self.hp.get('server_lr', None)
    if lr is None:
      lr = self.hp.get('lr', 0.0)

    shrinkage_mode = self.hp.get('shrinkage_mode', 'global')
    shrinkage_scope = self.hp.get('shrinkage_scope', 'all')
    sigma_source = self.hp.get('sigma_source', 'inter_client')

    # Prepare flattened/global quantities if needed
    if shrinkage_mode == 'global':
      # compute delta total and per-client total squared norms
      total_diff_norm = 0.0
      total_d = 0
      for name in self.dW:
        total_diff_norm += torch.sum((self.dW[name] - self.m[name]).pow(2)).to(device)
        total_d += self.dW[name].numel()
      # compute sigma^2 global across clients
      if sigma_source == 'inter_client':
        per_client_totals = []
        for client in clients:
          s = 0.0
          for name in self.dW:
            s += torch.sum((client.dW_compressed[name].to(device) - self.dW[name]).pow(2))
          per_client_totals.append(s)
        sigma2_global = torch.stack(per_client_totals).mean() if len(per_client_totals) > 0 else torch.tensor(0.0).to(device)

    # Update moments (m, v) with aggregated (raw) delta first
    for name in self.dW:
      delta = self.dW[name].to(device)
      # update biased first and second moments
      self.m[name] = beta1 * self.m[name] + (1.0 - beta1) * delta
      self.v[name] = beta2 * self.v[name] + (1.0 - beta2) * (delta.pow(2))

    # Now compute Stein-shrinkage per block or global
    logged_alphas = []
    logged_sigmas = []
    if shrinkage_mode == 'global':
      # global alpha scalar
      denom = total_diff_norm + eps
      d_total = total_d if total_d > 0 else 1
      alpha_raw = 1.0 - ((d_total - 2.0) * sigma2_global) / denom
      alpha = float(alpha_raw.clamp(0.0, 1.0)) if isinstance(alpha_raw, torch.Tensor) else max(0.0, min(1.0, alpha_raw))
      logged_alphas.append(alpha)
      logged_sigmas.append(sigma2_global.item() if isinstance(sigma2_global, torch.Tensor) else float(sigma2_global))
      # apply shrinkage across all blocks
      for name in self.dW:
        if shrinkage_scope == 'conv_only' and len(self.dW[name].shape) != 4:
          # skip shrinkage for non-conv layers
          continue
        delta = self.dW[name].to(device)
        diff = delta - self.m[name]
        delta_sr = self.m[name] + alpha * diff
        # replace aggregated update with shrinkage-corrected update
        self.dW[name] = delta_sr.clone()

    else:
      # per-layer shrinkage
      for name in self.dW:
        if shrinkage_scope == 'conv_only' and len(self.dW[name].shape) != 4:
          # skip shrinkage for non-conv layers
          continue

        delta = self.dW[name].to(device)
        # estimate sigma^2 for this block
        if sigma_source == 'inter_client':
          sigma2 = self._compute_inter_client_sigma2_block(clients, name, delta)
        else:
          # EMA fallback
          sigma2 = self.sigma_ema[name]

        d_b = float(self.dW[name].numel())
        diff = delta - self.m[name]
        denom = torch.sum(diff.pow(2)) + eps
        alpha_raw = 1.0 - ((d_b - 2.0) * sigma2) / denom
        if isinstance(alpha_raw, torch.Tensor):
          alpha = alpha_raw.clamp(0.0, 1.0).item()
        else:
          alpha = max(0.0, min(1.0, alpha_raw))

        logged_alphas.append(alpha)
        logged_sigmas.append(sigma2.item() if isinstance(sigma2, torch.Tensor) else float(sigma2))

        delta_sr = self.m[name] + alpha * diff
        self.dW[name] = delta_sr.clone()
        # update sigma EMA (simple momentum)
        if sigma_source == 'ema':
          self.sigma_ema[name] = 0.99 * self.sigma_ema[name] + 0.01 * sigma2

    # increment server step
    self.server_step += 1
    # Scale by adaptive denominator (FedAdam-like) before compression/downstream add
    final_lr = lr if lr is not None else self.hp.get('lr', 1.0)
    try:
      for name in self.dW:
        denom = torch.sqrt(self.v[name]) + eps
        self.dW[name] = (final_lr * self.dW[name]) / denom
    except Exception:
      pass

    # Logging shrinkage statistics to experiment
    try:
      if len(logged_alphas) > 0 and hasattr(self, 'xp') and self.xp is not None:
        alphas = np.array(logged_alphas)
        sigmas = np.array(logged_sigmas)
        mean_alpha = float(alphas.mean())
        frac_clipped = float(((alphas <= 0.0) | (alphas >= 1.0)).sum() / alphas.size)
        mean_sigma = float(sigmas.mean())
        self.xp.log({'sr_alpha_mean': mean_alpha, 'sr_alpha_frac_clipped': frac_clipped, 'sr_sigma_mean': mean_sigma}, printout=False)
    except Exception:
      pass


  def compress_weight_update_down(self, compression=None, accumulate=False, count_bits=False):
    if accumulate and compression[0] != "none":
      # compression with error accumulation   
      add(target=self.A, source=self.dW)
      compress(target=self.dW_compressed, source=self.A, compress_fun=comp.compression_function(*compression))
      subtract(target=self.A, source=self.dW_compressed)

    else: 
      # compression without error accumulation
      compress(target=self.dW_compressed, source=self.dW, compress_fun=comp.compression_function(*compression))

    add(target=self.W, source=self.dW_compressed)

    if count_bits:
      # Compute the update size
      self.bits_sent += [comp.get_update_size(self.dW_compressed, compression)]

 
  def evaluate(self, loader=None, max_samples=50000, verbose=True):
    """Evaluates local model stored in self.W on local dataset or other 'loader if specified for a maximum of 
    'max_samples and returns a dict containing all evaluation metrics"""
    self.model.eval()

    eval_loss, correct, samples, iters = 0.0, 0, 0, 0
    if not loader:
      loader = self.loader
    with torch.no_grad():
      for i, (x,y) in enumerate(loader):

        x, y = x.to(device), y.to(device)
        y_ = self.model(x)
        _, predicted = torch.max(y_.data, 1)
        eval_loss += self.loss_fn(y_, y).item()
        correct += (predicted == y).sum().item()
        samples += y_.shape[0]
        iters += 1

        if samples >= max_samples:
          break
      if verbose:
        print("Evaluated on {} samples ({} batches)".format(samples, iters))
  
      results_dict = {'loss' : eval_loss/iters, 'accuracy' : correct/samples}

    return results_dict