import glob, os, time

import numpy as np

import itertools as it
import matplotlib.pyplot as plt


def save_results(results_dict, path, name, verbose=True):
    results_numpy = {key : np.array(value)for key, value in results_dict.items()}

    if not os.path.exists(path):
        os.makedirs(path)
    np.savez(path+name, **results_numpy) 
    if verbose:
        print("Saved results to ", path+name+".npz")


def load_results(path, filename, verbose=True):
    results_dict = np.load(path+filename)

    if verbose:
        print("Loaded results from "+path+filename)
    return results_dict


class Experiment():
    '''Class that contains logic to store hyperparameters und results of an experiment'''
    hyperparameters = {}
    results = {}
    parameters = {}

    def __init__(self, hyperparameters=None, hp_dict=None):
        if hp_dict is not None:
            self.from_dict(hp_dict)
        else:
            self.hyperparameters = hyperparameters
            self.hyperparameters_ = {}
            self.results = {}
            self.parameters = {}
            self.hyperparameters['finished'] = False
            self.hyperparameters['log_id'] = np.random.randint(100000)


    def __str__(self):
        selfname = "Hyperparameters: \n"
        for key, value in self.hyperparameters.items():
            selfname += " - "+key+" "*(24-len(key))+str(value)+"\n"
        return selfname

    def __repr__(self):
        return self.__str__()

    def log(self, update_dict, printout=True, override=False):
        # update a result
        for key, value in update_dict.items(): 
            if (not key in self.results) or override:
                self.results[key] = [value]
            else:
                self.results[key] += [value]

        if printout:
            print(update_dict)


    def is_log_round(self, c_round):
        log_freq = self.hyperparameters['log_frequency']
        if c_round == self.hyperparameters['communication_rounds']:
            self.hyperparameters['finished'] = True

        return (c_round == 1) or (c_round % log_freq == 0) or (c_round == self.hyperparameters['communication_rounds'])

    def save_parameters(self, parameters):
        self.parameters = parameters

    def to_dict(self):
        # turns an experiment into a dict that can be saved to disc
        return {'hyperparameters' : self.hyperparameters, 'hyperparameters_' : self.hyperparameters_,
                    'parameters' : self.parameters, **self.results}

    def from_dict(self, hp_dict):
        # takes a dict and turns it into an experiment
        self.results = dict(hp_dict)

        self.hyperparameters = hp_dict['hyperparameters'][np.newaxis][0]

        if 'parameters' in hp_dict:
            self.parameters = hp_dict['parameters'][np.newaxis][0]
            del self.results['parameters']
        else:
            self.parameters = {}
        
        if 'hyperparameters_' in hp_dict:
            self.hyperparameters_ = hp_dict['hyperparameters_'][np.newaxis][0]
            del self.results['hyperparameters_']
        else:
            self.hyperparameters_ = {}

    def prepare(self, hp):
        self.hyperparameters_ = {key : str(value) for key, value in hp.items()}
        for key in ["communication_rounds", "compression_up", "accumulation_up", "compression_down", "accumulation_down",
                "batch_size", "lr", "aggregation", "log_frequency", "local_iterations", "net", "dataset",
                # include optimizer metadata so visualization can detect server/client methods
                "optimizer", "server_optimizer", "server_lr", "server_beta1", "server_beta2", "shrinkage_mode", "sigma_source",
                "client_update_noise_std"]:
            self.hyperparameters[key] = hp[key]

    def save_to_disc(self, path):
        save_results(self.to_dict(), path, 'xp_'+str(self.hyperparameters['log_id']))



def get_all_hp_combinations(hp):
    '''Turns a dict of lists into a list of dicts.
    
    If 'seeds' key is present, expands experiments across all parameter combinations
    and seeds, automatically generating log_path for each run.
    '''
    # Check if this is a compact specification with seeds
    if 'seeds' in hp:
        return _expand_with_seeds(hp)
    
    # Legacy behavior: cartesian product of all parameter lists
    combinations = it.product(*(hp[name] for name in hp))
    hp_dicts = [{key : value[i] for i,key in enumerate(hp)}for value in combinations]
    return hp_dicts


def _expand_with_seeds(hp_spec):
    '''Expands a compact experiment specification with seeds into multiple runs.
    
    Example input:
        {
            "dataset": ["mnist"],
            "server_optimizer": ["fedavg", "fedadam", "sr_fedadam"],
            "client_update_noise_std": [0.0, 0.01, 0.05],
            "seeds": [42, 43, 44],
            "base_log_path": ["results/compare/"],
            ... (other parameters)
        }
    
    Generates log_path as: {base_log_path}/sigma{noise}/{method}/run{seed}/
    '''
    seeds = hp_spec['seeds']
    hp_copy = {k: v for k, v in hp_spec.items() if k != 'seeds'}
    
    # Extract base log path (default if not specified)
    base_log_path = hp_copy.get('base_log_path', ['results/'])[0]
    if 'base_log_path' in hp_copy:
        del hp_copy['base_log_path']
    
    # Get all parameter combinations (excluding seeds and paths)
    param_keys = [k for k in hp_copy.keys() if k != 'log_path']
    param_combinations = it.product(*(hp_copy[name] for name in param_keys))
    
    expanded_experiments = []
    
    for param_values in param_combinations:
        # Create a dict for this parameter combination
        param_dict = {key: param_values[i] for i, key in enumerate(param_keys)}
        
        # Extract key identifiers for path generation
        dataset = param_dict.get('dataset', 'unknown')
        method = param_dict.get('server_optimizer', 'none')
        noise = param_dict.get('client_update_noise_std', 0.0)
        
        # Generate experiments for each seed
        for seed in seeds:
            exp = param_dict.copy()
            exp['seed'] = seed
            
            # Auto-generate log_path if not manually specified
            if 'log_path' not in hp_spec:
                # Format: base/dataset/[sigma{noise}/]method/run{seed}/
                if noise == 0.0:
                    # No noise: base/dataset/method/run{seed}/
                    exp['log_path'] = f"{base_log_path}{dataset}/{method}/run{seed}/"
                else:
                    # With noise: base/dataset/sigma{noise}/method/run{seed}/
                    noise_str = f"sigma{noise}".replace('.', 'p')
                    exp['log_path'] = f"{base_log_path}{dataset}/{noise_str}/{method}/run{seed}/"
            else:
                # Use manually specified log_path (should be a template or list)
                exp['log_path'] = hp_spec['log_path'][0] if isinstance(hp_spec['log_path'], list) else hp_spec['log_path']
            
            expanded_experiments.append(exp)
    
    return expanded_experiments


def list_of_dicts_to_dict(hp_dicts):
    '''Turns a list of dicts into one dict of lists containing all individual values'''
    one_dict = {}
    for hp in hp_dicts:
        for key, value in hp.items():
            if not key in one_dict: 
                one_dict[key] = [value]
            elif value not in one_dict[key]:
                one_dict[key] += [value]
    return one_dict


def get_list_of_experiments(path, only_finished=False, verbose=True):
    '''Returns all the results saved at location path'''
    list_of_experiments = []

    os.chdir(path)
    for file in glob.glob("*.npz"):
        list_of_experiments += [Experiment(hp_dict=load_results(path+"/",file, verbose=False))]

    if only_finished:
        list_of_experiments = [xp for xp in list_of_experiments if 'finished' in xp.hyperparameters and xp.hyperparameters['finished']]

    if list_of_experiments and verbose:
        print("Loaded ",len(list_of_experiments), " Results from ", path)
        print()
        get_experiments_metadata(list_of_experiments)

    if not list_of_experiments:
        print("No finished Experiments. Consider setting only_finished to False")

    return list_of_experiments


def get_experiment(path, name, verbose=False):
    '''Returns one result saved at location path'''
    experiment = Experiment(hp_dict=load_results(path+"/",name+".npz", verbose=False))

    if verbose:
        print("Loaded ",1, " Result from ", path)
        print()
        get_experiments_metadata([experiment])

    return experiment


def get_experiments_metadata(list_of_experiments):
    hp_dicts =  [experiment.hyperparameters for experiment in list_of_experiments]

    print('Hyperparameters: \n' ,list_of_dicts_to_dict(hp_dicts))
    print()
    print('Tracked Variables: \n', list(list_of_experiments[0].results.keys()))



