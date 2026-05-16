"""Protein structure evaluation metrics.

Implements standard metrics used in CASP to compare predicted and
experimental protein structures:

  - RMSD (with Kabsch superposition)
  - TM-score
  - GDT-TS / GDT-HA
  - lDDT (local distance difference test)

References:
    - Kabsch, Acta Crystallogr. A32, 922–923 (1976).
    - Zhang & Skolnick, Proteins 57, 702–710 (2004) — TM-score.
    - Zemla, Nucleic Acids Res. 31, 3370–3374 (2003) — GDT-TS.
    - Mariani et al., Bioinformatics 29, 2722–2728 (2013) — lDDT.
"""

from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Kabsch alignment
# ---------------------------------------------------------------------------


def kabsch_align(
    mobile: np.ndarray, target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch algorithm: find optimal rotation and translation.

    Args:
        mobile: (N, 3) coordinates to align.
        target: (N, 3) reference coordinates.

    Returns:
        aligned: (N, 3) aligned mobile coordinates.
        R: (3, 3) rotation matrix.
        t: (3,) translation vector.
    """
    assert mobile.shape == target.shape
    n = mobile.shape[0]

    # Centre
    cm = mobile.mean(axis=0)
    ct = target.mean(axis=0)
    p = mobile - cm
    q = target - ct

    # Cross-covariance
    H = p.T @ q  # (3, 3)
    U, S, Vt = np.linalg.svd(H)

    # Correct reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T
    t = ct - R @ cm
    aligned = (R @ mobile.T).T + t
    return aligned, R, t


# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------


def compute_rmsd(pred: np.ndarray, true: np.ndarray, align: bool = True) -> float:
    """Root-mean-square deviation between two coordinate sets.

    Args:
        pred: (N, 3) predicted coordinates.
        true: (N, 3) experimental coordinates.
        align: If True, apply Kabsch superposition first.

    Returns:
        RMSD in Angstroms.
    """
    assert pred.shape == true.shape and pred.ndim == 2
    if align:
        pred, _, _ = kabsch_align(pred, true)
    diff = pred - true
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


# ---------------------------------------------------------------------------
# TM-score
# ---------------------------------------------------------------------------


def compute_tm_score(
    pred: np.ndarray,
    true: np.ndarray,
    target_length: int | None = None,
    align: bool = True,
) -> float:
    """Template-Modelling score.

    TM-score = (1 / L_target) * max Σ_i 1 / (1 + (d_i / d_0)²)

    where d_0 = 1.24 * ∛(L_target - 15) − 1.8  (for L > 21).

    A score > 0.5 generally indicates the same fold.

    Args:
        pred: (N, 3)
        true: (N, 3)
        target_length: Normalisation length (defaults to N).
        align: Kabsch align first.

    Returns:
        TM-score in (0, 1].
    """
    assert pred.shape == true.shape
    N = pred.shape[0]
    L = target_length if target_length is not None else N

    if align:
        pred, _, _ = kabsch_align(pred, true)

    d0 = 1.24 * np.cbrt(max(L - 15, 1)) - 1.8
    d0 = max(d0, 0.5)  # floor

    distances = np.sqrt(np.sum((pred - true) ** 2, axis=1))
    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    return float(np.sum(scores) / L)


# ---------------------------------------------------------------------------
# GDT-TS / GDT-HA
# ---------------------------------------------------------------------------


def _gdt_at_cutoff(pred: np.ndarray, true: np.ndarray, cutoff: float) -> float:
    """Fraction of Cα atoms within *cutoff* Å after superposition."""
    distances = np.sqrt(np.sum((pred - true) ** 2, axis=1))
    return float(np.mean(distances <= cutoff))


def compute_gdt_ts(pred: np.ndarray, true: np.ndarray, align: bool = True) -> float:
    """Global Distance Test — Total Score.

    GDT-TS = mean(P_1, P_2, P_4, P_8)  where P_c = fraction within c Å.

    Args:
        pred, true: (N, 3)
        align: Kabsch align first.
    Returns:
        GDT-TS in [0, 1].
    """
    if align:
        pred, _, _ = kabsch_align(pred, true)
    return float(np.mean([_gdt_at_cutoff(pred, true, c) for c in (1, 2, 4, 8)]))


def compute_gdt_ha(pred: np.ndarray, true: np.ndarray, align: bool = True) -> float:
    """GDT — High Accuracy (stricter cutoffs).

    GDT-HA = mean(P_0.5, P_1, P_2, P_4).
    """
    if align:
        pred, _, _ = kabsch_align(pred, true)
    return float(np.mean([_gdt_at_cutoff(pred, true, c) for c in (0.5, 1, 2, 4)]))


# ---------------------------------------------------------------------------
# lDDT
# ---------------------------------------------------------------------------


def compute_lddt(
    pred: np.ndarray,
    true: np.ndarray,
    cutoff: float = 15.0,
    thresholds: tuple = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """Local Distance Difference Test.

    For every pair of atoms within *cutoff* Å in the true structure,
    compute the fraction whose predicted distance deviates by less than
    each threshold.

    Args:
        pred, true: (N, 3)
        cutoff: Inclusion radius in the reference (Å).
        thresholds: Distance tolerance thresholds (Å).

    Returns:
        lDDT score in [0, 1].
    """
    N = pred.shape[0]
    # Pairwise distances
    true_dm = np.sqrt(((true[:, None] - true[None, :]) ** 2).sum(-1))
    pred_dm = np.sqrt(((pred[:, None] - pred[None, :]) ** 2).sum(-1))

    # Pairs within cutoff in reference (exclude self)
    mask = (true_dm < cutoff) & (np.eye(N) == 0)

    if mask.sum() == 0:
        return 0.0

    diff = np.abs(pred_dm - true_dm)
    scores = []
    for thr in thresholds:
        scores.append((diff[mask] < thr).mean())
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Convenience: evaluate all metrics at once
# ---------------------------------------------------------------------------


def evaluate_structure(
    pred: np.ndarray,
    true: np.ndarray,
    target_length: int | None = None,
) -> Dict[str, float]:
    """Compute all standard structure-quality metrics.

    Returns a dict with keys: rmsd, tm_score, gdt_ts, gdt_ha, lddt.
    """
    aligned, _, _ = kabsch_align(pred, true)
    return {
        "rmsd": compute_rmsd(aligned, true, align=False),
        "tm_score": compute_tm_score(aligned, true, target_length, align=False),
        "gdt_ts": compute_gdt_ts(aligned, true, align=False),
        "gdt_ha": compute_gdt_ha(aligned, true, align=False),
        "lddt": compute_lddt(pred, true),
    }
