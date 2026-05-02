## 2026-05-02 - [Vectorized Metric Calculations]
**Learning:** Python loops over batches for metric calculations (RMSD, TM-score) create significant overhead, especially when combined with repeated .cpu().numpy() calls. Vectorizing these functions to handle batches natively in NumPy improves performance and reduces CPU-GPU synchronization bottlenecks.
**Action:** Always prefer vectorized NumPy operations for batch-wide metrics. Ensure metric functions can handle both (N, D) and (B, N, D) shapes.
