"""Tests for teacher_learner module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pickleball_pose_analyzer.comparison.teacher_learner import (
    calculate_metric_ranges,
    load_teacher_ranges,
)


@pytest.fixture
def sample_pose_data():
    """Sample pose data for testing."""
    return {
        'pred_keypoints_3d': np.random.rand(70, 3).astype(np.float32),
        'pred_keypoints_2d': np.random.rand(70, 2).astype(np.float32),
    }


def test_calculate_metric_ranges_percentile():
    """Test calculating ranges using percentile method."""
    metric_values = [85.0, 90.0, 95.0, 100.0, 105.0]
    ranges = calculate_metric_ranges(
        metric_values,
        method='percentile',
        percentile_low=25.0,
        percentile_high=75.0
    )

    assert ranges['method'] == 'percentile'
    assert 'min' in ranges
    assert 'max' in ranges
    assert 'mean' in ranges
    assert 'std_dev' in ranges
    assert ranges['min'] <= ranges['mean'] <= ranges['max']
    assert ranges['sample_count'] == 5


def test_calculate_metric_ranges_normal():
    """Test calculating ranges using normal distribution method."""
    metric_values = [95.0]  # Single value
    ranges = calculate_metric_ranges(
        metric_values,
        method='normal',
        tolerance_percentage=0.10
    )

    assert ranges['method'] == 'normal'
    assert 'min' in ranges
    assert 'max' in ranges
    assert 'mean' in ranges
    assert ranges['mean'] == 95.0
    # Range should be mean ± 10%
    assert abs(ranges['min'] - (95.0 - 9.5)) < 0.01
    assert abs(ranges['max'] - (95.0 + 9.5)) < 0.01
    assert ranges['sample_count'] == 1


def test_calculate_metric_ranges_single_value_uses_normal():
    """Test that single value automatically uses normal method."""
    metric_values = [95.0]
    ranges = calculate_metric_ranges(
        metric_values,
        method='percentile',  # Request percentile but only 1 value
        percentile_low=25.0,
        percentile_high=75.0
    )

    # Should fall back to normal method
    assert ranges['method'] == 'normal'


def test_calculate_metric_ranges_empty_values():
    """Test calculating ranges with empty values raises ValueError."""
    with pytest.raises(ValueError, match="metric_values cannot be empty"):
        calculate_metric_ranges([], method='percentile')


def test_load_teacher_ranges(sample_pose_data):
    """Test loading teacher ranges from JSON file."""
    ranges_data = {
        'stroke_type': 'serve',
        'preparation': {
            'shoulder_angle': {
                'min': 85.0,
                'max': 105.0,
                'mean': 95.0,
                'method': 'percentile',
                'sample_count': 5
            }
        },
        'contact': {},
        'finish': {}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ranges_data, f)
        ranges_path = f.name

    try:
        loaded_ranges = load_teacher_ranges(ranges_path)

        assert loaded_ranges['stroke_type'] == 'serve'
        assert 'preparation' in loaded_ranges
        assert 'shoulder_angle' in loaded_ranges['preparation']
        assert loaded_ranges['preparation']['shoulder_angle']['mean'] == 95.0
    finally:
        Path(ranges_path).unlink(missing_ok=True)


def test_load_teacher_ranges_file_not_found():
    """Test loading non-existent ranges file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_teacher_ranges('/nonexistent/path/ranges.json')


def test_calculate_metric_ranges_statistics():
    """Test that calculated ranges include correct statistics."""
    metric_values = [80.0, 85.0, 90.0, 95.0, 100.0]
    ranges = calculate_metric_ranges(
        metric_values,
        method='percentile',
        percentile_low=25.0,
        percentile_high=75.0
    )

    # Check mean
    expected_mean = sum(metric_values) / len(metric_values)
    assert abs(ranges['mean'] - expected_mean) < 0.01

    # Check percentiles
    assert ranges['percentile_25'] == 85.0
    assert ranges['percentile_75'] == 95.0

    # Check min/max are within data range
    assert ranges['min'] >= min(metric_values)
    assert ranges['max'] <= max(metric_values)
