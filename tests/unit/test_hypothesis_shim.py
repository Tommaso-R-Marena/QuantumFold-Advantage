"""Tests for the local Hypothesis compatibility shim."""

from __future__ import annotations

import hypothesis
from hypothesis import given
from hypothesis import strategies as st


def test_given_injects_deterministic_generated_arguments():
    # Arrange
    captured = {}

    @given(value=st.integers(min_value=7, max_value=11))
    def sample_test(value):
        captured["value"] = value

    # Act
    sample_test()

    # Assert
    assert captured["value"] == 7


def test_given_marks_wrapper_as_hypothesis_test():
    # Arrange
    @given(item=st.integers(min_value=1, max_value=3))
    def sample_test(item):
        return item

    # Act
    result = hypothesis.is_hypothesis_test(sample_test)

    # Assert
    assert result is True


def test_settings_profile_helpers_are_noops():
    # Arrange / Act
    hypothesis.settings.register_profile("ci", max_examples=10)
    hypothesis.settings.load_profile("ci")

    @hypothesis.settings(deadline=100)
    def sample_test():
        return "ok"

    # Assert
    assert sample_test() == "ok"


def test_list_strategy_respects_min_size_boundary():
    # Arrange
    integer_strategy = st.integers(min_value=3, max_value=8)
    list_strategy = st.lists(integer_strategy, min_size=4, max_size=9)

    # Act
    value = list_strategy.example()

    # Assert
    assert len(value) == 4
    assert value == [3, 3, 3, 3]
