# Mathematical Specification

This document provides the detailed mathematical foundations for the Universal Tiled Sinkhorn Engine.

## 1. Optimal Transport Problem

### 1.1 Kantorovich Formulation

Given source distribution $\mathbf{a} \in \Sigma_N$ and target distribution $\mathbf{b} \in \Sigma_M$ on probability simplices, and cost matrix $\mathbf{C} \in \mathbb{R}^{N \times M}$, the optimal transport problem seeks:

$$
\text{OT}(\mathbf{a}, \mathbf{b}) = \min_{\mathbf{P} \in \Pi(\mathbf{a}, \mathbf{b})} \langle \mathbf{P}, \mathbf{C} \rangle
$$

where $\Pi(\mathbf{a}, \mathbf{b}) = \{\mathbf{P} \in \mathbb{R}_+^{N \times M} : \mathbf{P}\mathbf{1}_M = \mathbf{a}, \mathbf{P}^\top\mathbf{1}_N = \mathbf{b}\}$.

### 1.2 Entropic Regularization

Adding entropic regularization with parameter $\varepsilon > 0$:

$$
\text{OT}_\varepsilon(\mathbf{a}, \mathbf{b}) = \min_{\mathbf{P} \in \Pi(\mathbf{a}, \mathbf{b})} \langle \mathbf{P}, \mathbf{C} \rangle - \varepsilon H(\mathbf{P})
$$

where $H(\mathbf{P}) = -\sum_{ij} P_{ij}(\log P_{ij} - 1)$ is the entropy.

---

## 2. Balanced Sinkhorn Algorithm

### 2.1 Dual Formulation

The dual problem introduces potentials $\mathbf{f} \in \mathbb{R}^N$ and $\mathbf{g} \in \mathbb{R}^M$:

$$
\max_{\mathbf{f}, \mathbf{g}} \langle \mathbf{f}, \mathbf{a} \rangle + \langle \mathbf{g}, \mathbf{b} \rangle - \varepsilon \sum_{ij} \exp\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right)
$$

### 2.2 Optimality Conditions

At optimality, the transport plan is:

$$
P_{ij}^* = \exp\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right)
$$

With marginal constraints $\mathbf{P}^*\mathbf{1} = \mathbf{a}$ and $(\mathbf{P}^*)^\top\mathbf{1} = \mathbf{b}$.

### 2.3 Sinkhorn Iterations (Log-Domain)

Define the **softmin** operator:

$$
\text{softmin}_\varepsilon(\mathbf{x}) = -\varepsilon \log \sum_i \exp(-x_i/\varepsilon)
$$

**Update Rules:**

$$
\begin{aligned}
f_i^{(k+1)} &= \text{softmin}_\varepsilon^j \left(C_{ij} - g_j^{(k)}\right) + \varepsilon \log a_i \\
g_j^{(k+1)} &= \text{softmin}_\varepsilon^i \left(C_{ij} - f_i^{(k+1)}\right) + \varepsilon \log b_j
\end{aligned}
$$

---

## 3. Unbalanced Optimal Transport

### 3.1 KL-Relaxed Formulation

Relax marginal constraints using KL divergence with parameters $\tau_a, \tau_b > 0$:

$$
\text{UOT}(\mathbf{a}, \mathbf{b}) = \min_{\mathbf{P} \geq 0} \langle \mathbf{P}, \mathbf{C} \rangle - \varepsilon H(\mathbf{P}) + \tau_a \text{KL}(\mathbf{P1} \| \mathbf{a}) + \tau_b \text{KL}(\mathbf{P}^\top\mathbf{1} \| \mathbf{b})
$$

### 3.2 Unified Scaling Factor

Define the scaling coefficients:

$$
\rho_a = \frac{\varepsilon}{\varepsilon + \tau_a}, \quad \rho_b = \frac{\varepsilon}{\varepsilon + \tau_b}
$$

**Properties:**

- As $\tau \to \infty$: $\rho \to 0$ (balanced OT)
- As $\tau \to 0$: $\rho \to 1$ (no constraint)

### 3.3 Unified Update Equations

$$
\begin{aligned}
f_i^{(k+1)} &= \rho_a \cdot f_i^{(k)} + (1 - \rho_a) \left[\text{softmin}_\varepsilon^j(C_{ij} - g_j^{(k)}) + \varepsilon \log a_i\right] \\
g_j^{(k+1)} &= \rho_b \cdot g_j^{(k)} + (1 - \rho_b) \left[\text{softmin}_\varepsilon^i(C_{ij} - f_i^{(k+1)}) + \varepsilon \log b_j\right]
\end{aligned}
$$

---

## 4. Masking for Variable-Length Data

### 4.1 Problem Setup

For batched data with varying lengths, we use masks $\mathbf{m}_a \in \{0,1\}^N$ and $\mathbf{m}_b \in \{0,1\}^M$.

### 4.2 Implementation

**Cost Matrix Masking:**
$$
\tilde{C}_{ij} = \begin{cases}
C_{ij} & \text{if } m_a^i = 1 \text{ and } m_b^j = 1 \\
+\infty & \text{otherwise}
\end{cases}
$$

**Marginal Masking:**
$$
\tilde{a}_i = \begin{cases}
a_i / \sum_{k: m_a^k=1} a_k & \text{if } m_a^i = 1 \\
0 & \text{otherwise}
\end{cases}
$$

This ensures $\exp(-\infty/\varepsilon) = 0$ in the transport plan.

---

## 5. Backward Pass: Gradient Checkpointing

### 5.1 Motivation

Standard autodiff through Sinkhorn iterations has:

- Memory: $O(K \cdot NM)$ for $K$ iterations (stores all intermediate states)
- Computation: Forward + backward through all iterations

We use **gradient checkpointing** to achieve:

- Memory: $O(NM)$ (only stores inputs)
- Computation: Forward 2× + backward 1×

### 5.2 Checkpointed Recomputation

Instead of storing intermediate states during forward pass, we:

1. **Forward pass**: Run Sinkhorn without gradient tracking (`torch.no_grad()`)
2. **Backward pass**: Re-run forward with gradients enabled, then backprop

```python
# Forward: memory-efficient, no grad tracking
with torch.no_grad():
    f, g = sinkhorn_iterations(C, a, b, ...)

# Backward: recompute with grad tracking
with torch.enable_grad():
    C_grad = C.detach().requires_grad_(True)
    f, g = sinkhorn_iterations(C_grad, a, b, ...)
    loss = (f * grad_f).sum() + (g * grad_g).sum()
    loss.backward()  # → C_grad.grad
```

### 5.3 Trade-offs

| Approach        | Memory                | Compute                    | Gradient Accuracy |
| --------------- | --------------------- | -------------------------- | ----------------- |
| Full Unrolled   | $O(K \cdot NM)$       | 1× forward + 1× backward   | Exact             |
| Checkpointing   | $O(NM)$               | 2× forward + 1× backward   | Exact             |
| Implicit Diff   | $O(NM)$               | 1× forward + linear solve  | Approximate*      |

*\*Implicit differentiation has convergence issues for balanced Sinkhorn due to gauge freedom (see Appendix A).*

### 5.4 Implementation

Both balanced and unbalanced Sinkhorn use the same checkpointing approach, implemented in `autograd.py`:

```python
class SinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, a, b, ...):
        with torch.no_grad():
            f, g = sinkhorn_forward(C, a, b, ...)
        ctx.save_for_backward(C, a, b, ...)
        return f, g

    @staticmethod
    def backward(ctx, grad_f, grad_g):
        C, a, b, ... = ctx.saved_tensors
        with torch.enable_grad():
            C_grad = C.detach().requires_grad_(True)
            f, g = sinkhorn_forward(C_grad, a, b, ...)
            loss = (f * grad_f).sum() + (g * grad_g).sum()
            loss.backward()
        return C_grad.grad, ...
```

### 5.5 Truncated Gradients

For faster training, you can use **truncated backpropagation** via the `grad_iters` parameter. Instead of recomputing all `max_iters` iterations, only the last `grad_iters` iterations are computed with gradient tracking.

```python
# Full recomputation (exact gradients, slower)
f, g = sinkhorn_differentiable(C, a, b, epsilon, grad_iters=0)

# Truncated (approximate gradients, faster)
f, g = sinkhorn_differentiable(C, a, b, epsilon, grad_iters=20)
```

#### Accuracy vs Epsilon

The required `grad_iters` depends on **epsilon** (entropic regularization):

| epsilon | Convergence | grad_iters=5     | grad_iters=10    | grad_iters=50    |
| ------- | ----------- | ---------------- | ---------------- | ---------------- |
| 0.1     | Slow        | ~34% error       | ~19% error       | ~1% error        |
| 1.0     | Fast        | **~0.01% error** | **~0% error**    | **~0% error**    |

> [!TIP]
> With larger epsilon (faster convergence), fewer gradient iterations are needed.
> A good rule of thumb: `grad_iters = max_iters // 4` for balanced trade-off.

#### When to Use Truncated Gradients

- **Use `grad_iters=0`** (full recompute): When gradient accuracy is critical
- **Use `grad_iters>0`** (truncated): For faster training with approximate gradients
- **Large epsilon** (≥0.5): `grad_iters=10` is often sufficient
- **Small epsilon** (<0.1): Use larger `grad_iters` or full recompute

---

## Appendix A: Implicit Differentiation (Reference)

> **Note**: This section describes implicit differentiation for reference.
> The current implementation uses gradient checkpointing instead due to
> convergence issues with the adjoint system for balanced Sinkhorn.

### A.1 Fixed-Point Formulation

At convergence, the potentials satisfy:

$$
\begin{pmatrix} \mathbf{f}^* \\ \mathbf{g}^* \end{pmatrix} = T\begin{pmatrix} \mathbf{f}^* \\ \mathbf{g}^* \end{pmatrix}
$$

where $T$ is the Sinkhorn operator.

### A.2 Adjoint System

Given upstream gradients $\bar{\mathbf{f}}, \bar{\mathbf{g}}$, solve for adjoint variables:

$$
(\mathbf{I} - \mathbf{J}_T^\top) \begin{pmatrix} \boldsymbol{\lambda}_f \\ \boldsymbol{\lambda}_g \end{pmatrix} = \begin{pmatrix} \bar{\mathbf{f}} \\ \bar{\mathbf{g}} \end{pmatrix}
$$

### A.3 Why It Fails for Balanced Sinkhorn

For balanced OT, the potentials have **gauge freedom**: adding constant $c$ to all $f_i$ and subtracting $c$ from all $g_j$ leaves the transport plan unchanged:

$$
P_{ij} = \exp\left(\frac{(f_i + c) + (g_j - c) - C_{ij}}{\varepsilon}\right) = \exp\left(\frac{f_i + g_j - C_{ij}}{\varepsilon}\right)
$$

This means $(\mathbf{I} - \mathbf{J}_T)$ has eigenvalue 1 for the constant eigenvector, making the system nearly singular and the Neumann series slow to converge or divergent

---

## 6. Numerical Stability Algorithms

### 6.1 Online Log-Sum-Exp (Streaming LSE)

Computing $\text{LSE}(\mathbf{x}) = \log \sum_i \exp(x_i)$ naively causes overflow.

**Two-Pass Algorithm:**

```text
m = max(x)
lse = m + log(sum(exp(x - m)))
```

**Online Algorithm (Single Pass):**

For streaming or tiled computation, we maintain state $(m, s)$ where $m$ is the running max and $s$ is the scaled sum:

$$
s = \sum_{i \text{ seen}} \exp(x_i - m)
$$

**Block Update Rule:**
Given new block with local $(\hat{m}, \hat{s})$ and global state $(m, s)$:

$$
\begin{aligned}
m' &= \max(m, \hat{m}) \\
s' &= s \cdot \exp(m - m') + \hat{s} \cdot \exp(\hat{m} - m')
\end{aligned}
$$

**Final Result:**
$$
\text{LSE} = m' + \log s'
$$

**Edge Cases:**

- If $m = -\infty$ (empty block): $\exp(m - m') = 0$ (no contribution)
- If all $x_i = -\infty$: return $-\infty$

### 6.2 Triton Kernel Implementation

The Triton LSE kernel processes in BLOCK_SIZE chunks:

```python
# Pseudocode
global_max = -inf
global_sum_exp = 0

for block in tiles(x):
    block_max = max(block)
    if block_max > -inf:
        exp_vals = exp(block - block_max)
        block_sum = sum(exp_vals)
        
        new_max = max(global_max, block_max)
        global_scale = exp(global_max - new_max)
        block_scale = exp(block_max - new_max)
        
        # Handle first block case
        if global_max == -inf:
            global_scale = 0
            
        global_sum_exp = global_sum_exp * global_scale + block_sum * block_scale
        global_max = new_max

lse = global_max + log(max(global_sum_exp, 1e-38))
```

### 6.3 Underflow in Marginals

When $a_i$ or $b_j$ are very small:

- Use $\log a_i$ directly instead of computing $a_i$ then taking log
- Clamp to minimum value: $\log(\max(a_i, \delta))$ where $\delta \approx 10^{-38}$

### 6.4 Convergence Criterion

Monitor potential change instead of marginal error (more stable):

$$
\text{converged} = \max\left(\|\mathbf{f}^{(k+1)} - \mathbf{f}^{(k)}\|_\infty, \|\mathbf{g}^{(k+1)} - \mathbf{g}^{(k)}\|_\infty\right) < \theta
$$

### 6.5 Epsilon Sensitivity

| $\varepsilon$      | Behavior                                                  |
| ------------------ | --------------------------------------------------------- |
| Large (>1)         | Fast convergence, smooth plan, but poor OT approximation |
| Medium (0.01-0.1)  | Good balance of speed and accuracy                       |
| Small (<0.01)      | Accurate OT, but slow convergence and numerical issues   |

---

## 7. Jacobian-Vector Product

### 7.1 JVP for Sinkhorn Operator

The Jacobian of the Sinkhorn operator $T$ at fixed point $(f^*, g^*)$ acts on vectors $(v_f, v_g)$ as:

$$
\mathbf{J}_T \begin{pmatrix} v_f \\ v_g \end{pmatrix} = \begin{pmatrix}
\rho_a v_f + (1-\rho_a) \frac{\mathbf{P} v_g}{\mathbf{P}\mathbf{1}} \\
\rho_b v_g + (1-\rho_b) \frac{\mathbf{P}^\top v_f}{\mathbf{P}^\top\mathbf{1}}
\end{pmatrix}
$$

where division is element-wise.

### 7.2 Efficient Implementation

No explicit Jacobian materialization:

- $\mathbf{P} v_g$: batched matrix-vector product
- Division by row/column sums: element-wise

---

## References

1. Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
2. Peyré, G. & Cuturi, M. (2019). "Computational Optimal Transport"
3. Chizat, L. et al. (2018). "Scaling Algorithms for Unbalanced Optimal Transport Problems"
4. Eisenberger, M. et al. (2022). "Unified Differentiable Optimal Transport"
5. Milakov, M. & Gimelshein, N. (2018). "Online normalizer calculation for softmax"
