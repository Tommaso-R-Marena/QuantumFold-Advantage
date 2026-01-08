#!/usr/bin/env python3
"""Statistical evaluation of quantum advantage claims."""
import sys
import json
import logging
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging

logger = logging.getLogger(__name__)

def load_experiment_results(results_file='outputs/quantum_experiment/experiment_results.json'):
    """Load experimental results."""
    with open(results_file) as f:
        return json.load(f)

def compute_rmsd(coords1, coords2):
    """
    Compute RMSD between two coordinate sets.
    
    Args:
        coords1, coords2: (N, 3) arrays
    
    Returns:
        float: RMSD value
    """
    return np.sqrt(np.mean(np.sum((coords1 - coords2)**2, axis=1)))

def compute_tm_score(coords1, coords2, seq_len):
    """
    Simplified TM-score calculation.
    
    Real TM-score requires optimal superposition. This is a placeholder.
    
    Args:
        coords1, coords2: (N, 3) arrays
        seq_len: Sequence length
    
    Returns:
        float: Approximate TM-score
    """
    # Simplified: normalize RMSD by sequence length
    # Real TM-score: doi.org/10.1093/nar/gki524
    rmsd = compute_rmsd(coords1, coords2)
    d0 = 1.24 * (seq_len - 15)**(1/3) - 1.8
    tm_score = 1.0 / (1 + (rmsd / d0)**2)
    return tm_score

def permutation_test(classical_losses, hybrid_losses, n_permutations=10000):
    """
    Two-sample permutation test.
    
    Null hypothesis: No difference between classical and hybrid.
    
    Returns:
        dict: Test results
    """
    observed_diff = np.mean(classical_losses) - np.mean(hybrid_losses)
    
    # Pool samples
    pooled = np.concatenate([classical_losses, hybrid_losses])
    n1 = len(classical_losses)
    
    # Generate permutations
    null_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(pooled)
        perm_classical = pooled[:n1]
        perm_hybrid = pooled[n1:]
        null_diff = np.mean(perm_classical) - np.mean(perm_hybrid)
        null_diffs.append(null_diff)
    
    # P-value: proportion of null diffs >= observed
    p_value = np.mean(np.abs(null_diffs) >= np.abs(observed_diff))
    
    return {
        'observed_difference': float(observed_diff),
        'p_value': float(p_value),
        'significant_at_0.05': p_value < 0.05,
        'n_permutations': n_permutations
    }

def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    """
    Bootstrap confidence interval.
    
    Returns:
        tuple: (lower, upper) bounds
    """
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return lower, upper

def evaluate_results(results):
    """
    Perform statistical evaluation of experiment results.
    
    Args:
        results: Dict from quantum_experiment.py
    
    Returns:
        dict: Statistical evaluation
    """
    classical_losses = np.array([r['final_val_loss'] for r in results['classical']])
    hybrid_losses = np.array([r['final_val_loss'] for r in results['hybrid']])
    
    # Descriptive statistics
    evaluation = {
        'classical': {
            'mean': float(np.mean(classical_losses)),
            'std': float(np.std(classical_losses)),
            'ci_95': list(bootstrap_ci(classical_losses))
        },
        'hybrid': {
            'mean': float(np.mean(hybrid_losses)),
            'std': float(np.std(hybrid_losses)),
            'ci_95': list(bootstrap_ci(hybrid_losses))
        }
    }
    
    # Permutation test
    perm_results = permutation_test(classical_losses, hybrid_losses)
    evaluation['permutation_test'] = perm_results
    
    # Paired t-test (assumes paired samples with same seed)
    if len(classical_losses) == len(hybrid_losses):
        t_stat, t_pval = stats.ttest_rel(classical_losses, hybrid_losses)
        evaluation['paired_ttest'] = {
            't_statistic': float(t_stat),
            'p_value': float(t_pval),
            'significant_at_0.05': t_pval < 0.05
        }
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(classical_losses)**2 + np.std(hybrid_losses)**2) / 2)
    cohens_d = (np.mean(classical_losses) - np.mean(hybrid_losses)) / pooled_std
    evaluation['effect_size'] = {
        'cohens_d': float(cohens_d),
        'interpretation': (
            'large' if abs(cohens_d) >= 0.8 else
            'medium' if abs(cohens_d) >= 0.5 else
            'small'
        )
    }
    
    return evaluation

if __name__ == '__main__':
    setup_logging()
    
    # Load results
    results_file = 'outputs/quantum_experiment/experiment_results.json'
    if not Path(results_file).exists():
        logger.error(f"Results file not found: {results_file}")
        logger.error("Run quantum_experiment.py first.")
        sys.exit(1)
    
    logger.info(f"Loading results from {results_file}")
    results = load_experiment_results(results_file)
    
    # Evaluate
    logger.info("Performing statistical evaluation...")
    evaluation = evaluate_results(results)
    
    # Save
    output_file = 'outputs/quantum_experiment/statistical_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    logger.info(f"Evaluation saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("STATISTICAL EVALUATION")
    print("="*60)
    print(f"\nClassical: {evaluation['classical']['mean']:.4f} ± {evaluation['classical']['std']:.4f}")
    print(f"Hybrid: {evaluation['hybrid']['mean']:.4f} ± {evaluation['hybrid']['std']:.4f}")
    print(f"\nPermutation test p-value: {evaluation['permutation_test']['p_value']:.4f}")
    print(f"Significant at α=0.05: {evaluation['permutation_test']['significant_at_0.05']}")
    print(f"\nEffect size (Cohen's d): {evaluation['effect_size']['cohens_d']:.3f} ({evaluation['effect_size']['interpretation']})")
    print("\n" + "="*60)
    print("\nSee docs/ADVANTAGE_CLAIM_PROTOCOL.md for interpretation guidelines.")
    print("="*60)