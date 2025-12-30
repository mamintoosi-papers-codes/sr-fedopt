
 **Context:**
 I am extending a federated learning codebase (`felisat/federated-learning`) to add a new **server-side optimizer** inspired by **Stein-Rule Adam (SR-Adam)**.

 I already have a working PyTorch implementation of SR-Adam in the *centralized* setting, where Stein shrinkage is applied selectively to parameter groups, computed in Adam-whitened space, with warmup, clipping, and stability safeguards.

 In federated learning:

 * **Clients must remain unchanged** (local SGD or Adam as implemented).
 * All changes must be **server-side only**, following the FedOpt paradigm.
 * The new method should be called **SR-FedAdam**.

 ---

 **Design requirements for SR-FedAdam:**

 1. Implement SR-FedAdam as a **drop-in server optimizer**, similar in structure to existing `FedAdam`, `FedYogi`, etc.
 2. Stein shrinkage should be applied to the **aggregated client update** (global delta), not to client gradients.
 3. Use the **server momentum (FedAdam first moment)** as the restricted estimator in the Stein rule.
 4. Estimate noise variance preferably from **inter-client variance of updates**:
    [
    \sigma^2 = \frac{1}{K} \sum_k |\Delta^{(k)} - \Delta|^2
    ]
    with a safe fallback to an EMA-based estimate if needed.
 5. Support **selective shrinkage**:

    * Global (all parameters)
    * Per-layer
    * Conv-only layers (parameters with ndim == 4)
 6. Use **positive-part Stein shrinkage** with clipping for stability.
 7. Keep all existing training scripts, client logic, and protocols unchanged.

 ---

 **Implementation guidance:**

 * Modify or extend `server_optimizer.py` to add a new optimizer class `SRFedAdam`.
 * Follow the structure and coding style of existing FedAdam implementation.
 * Maintain server-side state: first moment, second moment, and optional variance EMA.
 * Add configuration flags for:

   * `shrinkage_mode` (global / per-layer)
   * `shrinkage_scope` (all / conv_only)
   * `sigma_source` (inter_client / ema)
 * Do not refactor unrelated code.

 ---

 **Goal:**
 Produce clean, research-quality code suitable for experimental evaluation and inclusion in a conference or journal paper on federated optimization.

 Please generate the necessary code modifications with minimal intrusion, clear structure, and well-documented logic.


---

# Additional info:


## 1. Conceptual Mapping: SR-Adam → Federated Optimization

The key conceptual shift is to reinterpret **SR-Adam not as a client optimizer**, but as a **server-side adaptive aggregation rule** operating on *client updates* rather than stochastic minibatch gradients.

### Centralized SR-Adam (recap, abstracted)

* Observation: raw gradient ( g_t ) is noisy.
* Restricted estimator: momentum-based estimate ( m_t ).
* Shrinkage:
  [
  \tilde g_t = m_t + \alpha_t (g_t - m_t), \quad
  \alpha_t = \left[1 - \frac{(d-2)\sigma^2}{|g_t - m_t|^2}\right]_+
  ]
* ( \sigma^2 ) estimated from Adam second moments.

### Federated reinterpretation

| SR-Adam concept              | FL analogue                                                       |
| ---------------------------- | ----------------------------------------------------------------- |
| Noisy gradient ( g_t )       | Aggregated client update ( \Delta_t = \sum_k w_k \Delta_t^{(k)} ) |
| Noise source                 | Inter-client heterogeneity + local SGD noise                      |
| Restricted estimator ( m_t ) | Server-side momentum (FedAdam first moment)                       |
| Dimension ( d )              | Parameter block size (global or per-layer)                        |
| ( \sigma^2 )                 | Inter-client variance of updates (preferred) or EMA fallback      |
| Selective shrinkage          | Layer-wise or param-group-wise shrinkage at server                |

**Key insight:**
SR-FedAdam is best viewed as *variance-aware correction of the global update*, not as modification of the client objective or protocol.

---

## 2. Where SR-FedAdam Fits in the FL Stack

In felistat/federated-learning, the canonical server update pipeline is:

1. **Clients** compute local updates ( \Delta^{(k)} )
2. **Server** aggregates updates
3. **Server optimizer** updates global model

Your method belongs entirely in step (3), with minimal intrusion into (1) and (2).

---

## 3. Concrete Repository-Level Modification Plan

### Relevant files (current structure)

From the repository organization:

```
federated-learning/
├── federated_learning/
│   ├── server/
│   │   ├── server.py
│   │   ├── server_optimizer.py
│   ├── clients/
│   ├── models/
│   ├── experiment_manager.py
│   ├── sweep_runner.py
```

### Files to modify / extend

#### (A) `server_optimizer.py` — **primary location**

Add a new optimizer class:

* `SRFedAdam(ServerOptimizer)`

This mirrors existing:

* `FedAdam`
* `FedYogi`
* `FedAdagrad`

You should **not** touch client code.

---

#### (B) `server.py`

Minimal change:

* Allow selection of `server_optimizer="sr_fedadam"`
* Pass optional flags:

  * `shrinkage_mode`: `{"global", "per_layer"}`
  * `shrinkage_scope`: `{"all", "conv_only"}`
  * `sigma_source`: `{"inter_client", "ema"}`

---

#### (C) `experiment_manager.py`

* Register SR-FedAdam as a valid optimizer
* Ensure logging of:

  * Shrinkage factor statistics (mean, fraction clipped)
  * Inter-client variance norms

---

#### (D) (Optional) `utils/metrics.py`

Add diagnostics:

* Update disagreement:
  [
  \mathbb{E}_k |\Delta^{(k)} - \Delta|^2
  ]
* Effective shrinkage ratio per round

---

## 4. SR-FedAdam: Algorithmic Design

### State maintained at server

For each parameter block ( \theta ):

* First moment ( m_t )
* Second moment ( v_t ) (FedAdam-style)
* Optional EMA of inter-client variance

---

### Step-by-step server update (per round)

#### 1. Aggregate client updates

[
\Delta_t = \sum_k w_k \Delta_t^{(k)}
]

#### 2. Compute inter-client variance (preferred)

[
\sigma^2_t = \frac{1}{K} \sum_k |\Delta_t^{(k)} - \Delta_t|^2
]

* Compute **per-layer** if shrinkage is layerwise
* Fall back to EMA if ( K ) small or variance unstable

---

#### 3. FedAdam moments

[
m_t = \beta_1 m_{t-1} + (1-\beta_1)\Delta_t
]
[
v_t = \beta_2 v_{t-1} + (1-\beta_2)\Delta_t^2
]

---

#### 4. Stein shrinkage on updates

For each parameter block ( B ):

[
\alpha_B = \left[1 - \frac{(d_B - 2)\sigma_B^2}{|\Delta_B - m_B|^2 + \varepsilon}\right]_+
]

[
\tilde{\Delta}_B = m_B + \alpha_B(\Delta_B - m_B)
]

Notes:

* Positive-part Stein is **essential** for stability.
* You should cap ( \alpha_B \in [0,1] ).
* Conv-only mode: apply only if `param.ndim == 4`.

---

#### 5. Parameter update

Standard FedAdam rule:
[
\theta_{t+1} = \theta_t - \eta \frac{\tilde{\Delta}_t}{\sqrt{v_t} + \epsilon}
]

---

## 5. Pseudocode Skeleton (Server Optimizer)

```python
class SRFedAdam(ServerOptimizer):
    def step(self, client_updates):
        delta = weighted_average(client_updates)
        var = compute_inter_client_variance(client_updates, delta)

        self.m = beta1 * self.m + (1 - beta1) * delta
        self.v = beta2 * self.v + (1 - beta2) * delta.pow(2)

        delta_sr = {}
        for block in param_blocks(delta):
            if not shrinkage_enabled(block):
                delta_sr[block] = delta[block]
                continue

            diff = delta[block] - self.m[block]
            denom = diff.norm()**2 + eps
            alpha = max(0.0, 1.0 - (block.dim - 2) * var[block] / denom)

            delta_sr[block] = self.m[block] + alpha * diff

        apply_update(self.model, delta_sr, self.v)
```

---

## 6. Design Choices Worth Justifying in a Paper

### (A) Why server-side only?

* Preserves client simplicity and privacy assumptions
* Avoids protocol drift
* Aligns with FedOpt philosophy (Reddi et al.)

---

### (B) Why inter-client variance as σ²?

* Statistically aligned with Stein’s original noise model
* Directly measures heterogeneity-induced noise
* Superior to Adam-based proxy under non-IID data

---

### (C) Why selective (conv-only) shrinkage?

* High-dimensional convolutional filters benefit from shrinkage
* Dense layers exhibit structured bias under heterogeneity
* Matches your centralized empirical findings

This is a strong *cross-setting consistency argument*.

---

## 7. Experimental Protocols (Paper-Ready)

### Baselines

* FedAvg
* FedAdam
* FedYogi
* FedAdagrad

### Ablations

1. No shrinkage vs global vs per-layer
2. Conv-only vs all-params
3. σ² source: inter-client vs EMA
4. Positive-part vs raw Stein

---

### Stress Tests

* Dirichlet α ∈ {0.1, 0.3, 0.5}
* Partial participation (10–20%)
* Label noise at client level (heterogeneous corruption)
* Vary local steps ( E )

---

### Reporting

* Mean ± std over ≥5 seeds
* Communication rounds to target accuracy
* Shrinkage factor dynamics over rounds
* Inter-client variance vs round index

---

## 8. Positioning the Contribution

**SR-FedAdam** can be framed as:

> *A variance-aware, shrinkage-based server optimizer that explicitly regularizes client drift without modifying client behavior.*

This differentiates it from:

* Robust aggregation (median, trimmed mean)
* Client-side regularization (FedProx)
* Adaptive optimizers without statistical grounding
