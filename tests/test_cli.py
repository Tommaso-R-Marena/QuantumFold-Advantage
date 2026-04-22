"""CLI validation tests."""

import pytest

from src.cli import train_cli


def test_train_cli_rejects_non_positive_benchmark_samples(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "quantumfold-train",
            "--auto-benchmark",
            "--benchmark-samples",
            "0",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        train_cli()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--benchmark-samples must be >= 1" in captured.err
