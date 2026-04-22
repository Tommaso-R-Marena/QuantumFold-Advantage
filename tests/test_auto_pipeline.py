"""Tests for auto-improvement and auto-benchmark pipeline."""

from src.auto_pipeline import AutoBenchmarkRunner, AutoImprovementEngine, BenchmarkCandidate


def test_auto_improvement_reduces_lr_on_plateau():
    engine = AutoImprovementEngine()
    history = {
        "train_loss": [0.61, 0.6098, 0.6097],
        "val_loss": [0.71, 0.7097, 0.7098],
    }
    current = {
        "learning_rate": 1e-3,
        "batch_size": 32.0,
        "weight_decay": 1e-4,
        "model_width_multiplier": 1.0,
    }

    improved = engine.suggest(history=history, current=current)
    assert improved["learning_rate"] < current["learning_rate"]


def test_auto_benchmark_returns_ranked_candidates():
    runner = AutoBenchmarkRunner(seed=123)
    candidates = [
        BenchmarkCandidate("a", 1e-3, 32, 3, False, False),
        BenchmarkCandidate("b", 5e-4, 64, 3, True, True),
    ]

    report = runner.benchmark(candidates, n_samples=3)

    assert report["n_candidates"] == 2
    assert len(report["ranked"]) == 2
    assert report["best"]["name"] in {"a", "b"}
    assert report["ranked"][0]["composite_score"] >= report["ranked"][1]["composite_score"]


def test_auto_benchmark_rejects_non_positive_sample_count():
    runner = AutoBenchmarkRunner(seed=123)
    candidates = [BenchmarkCandidate("a", 1e-3, 32, 3, False, False)]

    try:
        runner.benchmark(candidates, n_samples=0)
        assert False, "Expected ValueError for n_samples < 1"
    except ValueError as exc:
        assert "n_samples must be >= 1" in str(exc)


def test_auto_benchmark_rejects_empty_candidates():
    runner = AutoBenchmarkRunner(seed=123)

    try:
        runner.benchmark([], n_samples=3)
        assert False, "Expected ValueError for empty candidates"
    except ValueError as exc:
        assert "at least one benchmark candidate" in str(exc)
