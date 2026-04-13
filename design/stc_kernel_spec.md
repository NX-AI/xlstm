# Sparse Ternary Covariance (STC) Kernel Specification

## Objective
Implement an optional backend for the mLSTM matrix memory update that uses sparse ternary quantization to achieve write-skip acceleration.

## Ternary Quantization Rule
The activations $v$ and $k$ are quantized into the ternary set $\{-1, 0, 1\}$ using a symmetric thresholding rule:

$$Q(x) = 
\begin{cases} 
1 & x > \tau \\
0 & |x| \le \tau \\
-1 & x < -\tau 
\end{cases}$$

## Adaptive Thresholding
To maintain numerical stability across layers and training steps, the threshold $\tau$ must be adaptive based on the magnitude of the activations.

$$ \tau_t = 0.1 \cdot \text{EMA}(|x|) $$

where $\text{EMA}$ is an exponential moving average over recent activation magnitudes.

## Sparse Update Logic (STC Backend)
The dense update:
$$ C_{t} = \lambda C_{t-1} + \text{outer}(k, v) $$

is transformed into:
$$ v_q = Q(v) $$
$$ k_q = Q(k) $$
$$ C_{t} = \lambda C_{t-1} + v_q k_q^T $$

## Write-Skip Principle
The core performance optimization is to skip updates to $C_{ij}$ if either $v_i = 0$ or $k_j = 0$.

```python
for i in nonzero(v_q):
    for j in nonzero(k_q):
        C[i, j] += v_q[i] * k_q[j]
```

At 90% sparsity, this should result in a 10x reduction in write operations to the covariance matrix.

## Straight-Through Estimator (STE)
During training, gradients must be passed through the quantizer to ensure stability.

```python
class TernaryQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau):
        return Q(x, tau)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None
```
