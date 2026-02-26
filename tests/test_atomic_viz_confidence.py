"""Tests for confidence-aware atomic visualization helpers."""

import numpy as np

from src.visualization.atomic_viz import ProteinVisualizer


def test_confidence_bucket_ranges_groups_contiguous_residues():
    visualizer = ProteinVisualizer()
    confidence = np.array([95, 93, 72, 68, 49, 48, 88])

    grouped = visualizer._confidence_bucket_ranges(confidence)

    assert grouped["#0053D6"] == [(1, 2)]
    assert grouped["#65CBF3"] == [(3, 3), (7, 7)]
    assert grouped["#FFDB13"] == [(4, 4)]
    assert grouped["#FF7D45"] == [(5, 6)]


def test_confidence_legend_html_includes_summary_stats():
    visualizer = ProteinVisualizer()
    confidence = np.array([50.0, 75.0, 95.0])

    legend = visualizer._confidence_legend_html(confidence)

    assert "Prediction confidence (pLDDT)" in legend
    assert "Mean: 73.3" in legend
    assert "Min: 50.0" in legend
    assert "Max: 95.0" in legend
