# Bolt's Performance Journal

## 2025-05-15 - [Evoformer PairUpdate Optimization]
**Learning:** The unoptimized `PairUpdate` block used a memory-intensive $O(L^2 D^2)$ outer product followed by a linear projection. By reordering the 3-tensor contraction into two sequential `torch.einsum` operations ($O(L^2 D)$), we achieved a ~7.6x speedup on CPU and significantly reduced peak memory consumption.
**Action:** Always check for high-rank intermediate tensors in attention-like blocks; sequential contractions are almost always better for memory and speed.

## 2025-05-15 - [Repository Structural Fragility]
**Learning:** Several core files (`rna_metrics.py`, `backend_manager.py`) were found with duplicated definitions and syntax errors, likely due to previous tool-assisted edits. These errors block CI and linting even if the local change is correct.
**Action:** Run `python3 -m compileall src/` after any structural change or when inheriting a "dirty" environment to catch syntax issues early.

## 2025-05-15 - [Notebook Syntax Validation in CI]
**Learning:** Standard `ast.parse` fails on IPython magics (`%`) and shell commands (`!`) in notebooks.
**Action:** Use a regex-based filter to replace these lines with `pass` while preserving indentation before running syntax checks in GitHub Actions.
