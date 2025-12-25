"""Tests for scoring_v2 module."""

import pytest
from pickleball_pose_analyzer.comparison.scoring_v2 import (
    score_metric,
    score_metric_distance,
    score_metric_percentile,
    score_metric_tolerance,
)


def test_score_metric_tolerance_in_range():
    """Test tolerance scoring when value is in ideal range."""
    score = score_metric_tolerance(95.0, 85.0, 105.0, tolerance=0.2)

    assert score == 100.0


def test_score_metric_tolerance_outside_range():
    """Test tolerance scoring when value is outside ideal range."""
    score = score_metric_tolerance(120.0, 85.0, 105.0, tolerance=0.2)

    assert 0 <= score < 100.0
    assert score < 50.0  # Should be penalized


def test_score_metric_tolerance_within_tolerance():
    """Test tolerance scoring when value is within tolerance zone."""
    # Range: 85-105, center: 95, tolerance: 0.2 * 20 = 4
    # Value 110 is just outside range but within tolerance
    score = score_metric_tolerance(110.0, 85.0, 105.0, tolerance=0.2)

    # Score should be between 0 and 100, likely in the 40-60 range for this case
    assert 0.0 <= score < 100.0
    assert score < 60.0  # Should be penalized but not too harshly


def test_score_metric_distance_in_range():
    """Test distance scoring when value is in range."""
    score = score_metric_distance(0.5, 0.4, 0.6, max_distance=0.1)

    assert score == 100.0


def test_score_metric_distance_outside_range():
    """Test distance scoring when value is outside range."""
    score = score_metric_distance(0.8, 0.4, 0.6, max_distance=0.1)

    assert 0 <= score < 100.0


def test_score_metric_distance_at_max_distance():
    """Test distance scoring at maximum distance."""
    score = score_metric_distance(0.7, 0.4, 0.6, max_distance=0.1)

    # Allow small floating point error
    assert abs(score - 0.0) < 0.01  # Approximately 0


def test_score_metric_percentile_in_range():
    """Test percentile scoring when value is in percentile range."""
    teacher_values = [85.0, 90.0, 95.0, 100.0, 105.0]
    score = score_metric_percentile(95.0, teacher_values, percentile_low=25.0, percentile_high=75.0)

    assert score == 100.0


def test_score_metric_percentile_outside_range():
    """Test percentile scoring when value is outside percentile range."""
    teacher_values = [85.0, 90.0, 95.0, 100.0, 105.0]
    score = score_metric_percentile(120.0, teacher_values, percentile_low=25.0, percentile_high=75.0)

    assert 0 <= score < 100.0


def test_score_metric_percentile_empty_values():
    """Test percentile scoring with empty teacher values."""
    score = score_metric_percentile(95.0, [], percentile_low=25.0, percentile_high=75.0)

    assert score == 0.0


def test_score_metric_tolerance_method():
    """Test score_metric with tolerance method."""
    teacher_range = {
        'min': 85.0,
        'max': 105.0,
        'mean': 95.0
    }
    scoring_config = {
        'type': 'tolerance',
        'tolerance': 0.2
    }

    score = score_metric(95.0, teacher_range, scoring_config)

    assert score == 100.0


def test_score_metric_distance_method():
    """Test score_metric with distance method."""
    teacher_range = {
        'min': 0.4,
        'max': 0.6,
        'mean': 0.5
    }
    scoring_config = {
        'type': 'distance',
        'max_distance': 0.1
    }

    score = score_metric(0.5, teacher_range, scoring_config)

    assert score == 100.0


def test_score_metric_percentile_method():
    """Test score_metric with percentile method."""
    teacher_range = {
        'min': 90.0,
        'max': 100.0,
        'mean': 95.0,
        'teacher_values': [85.0, 90.0, 95.0, 100.0, 105.0]
    }
    scoring_config = {
        'type': 'percentile',
        'percentile_low': 25.0,
        'percentile_high': 75.0
    }

    score = score_metric(95.0, teacher_range, scoring_config)

    assert score == 100.0


def test_score_metric_percentile_fallback():
    """Test score_metric with percentile method falls back to tolerance if no teacher_values."""
    teacher_range = {
        'min': 90.0,
        'max': 100.0,
        'mean': 95.0,
        # No teacher_values
    }
    scoring_config = {
        'type': 'percentile',
        'percentile_low': 25.0,
        'percentile_high': 75.0
    }

    score = score_metric(95.0, teacher_range, scoring_config)

    # Should fall back to tolerance method
    assert 0 <= score <= 100.0


def test_score_metric_unknown_method():
    """Test score_metric with unknown method raises ValueError."""
    teacher_range = {'min': 85.0, 'max': 105.0, 'mean': 95.0}
    scoring_config = {'type': 'unknown_method'}

    with pytest.raises(ValueError, match="Unknown scoring method type"):
        score_metric(95.0, teacher_range, scoring_config)
