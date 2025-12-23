"""Tests for scoring module."""

import numpy as np
import pytest
from pickleball_pose_analyzer.scoring import (
    calculate_3d_angle,
    score_contact_position,
    score_finish_position,
    score_preparation_position,
)


def test_calculate_3d_angle_right_angle():
    """Test 90-degree angle calculation."""
    point_a = np.array([1.0, 0.0, 0.0])
    point_b = np.array([0.0, 0.0, 0.0])
    point_c = np.array([0.0, 1.0, 0.0])

    angle = calculate_3d_angle(point_a, point_b, point_c)

    assert abs(angle - 90.0) < 1.0  # Allow small floating point error


def test_calculate_3d_angle_straight_line():
    """Test 180-degree angle (straight line)."""
    point_a = np.array([-1.0, 0.0, 0.0])
    point_b = np.array([0.0, 0.0, 0.0])
    point_c = np.array([1.0, 0.0, 0.0])

    angle = calculate_3d_angle(point_a, point_b, point_c)

    assert abs(angle - 180.0) < 1.0


def test_score_preparation_position_basic(sample_pose_data):
    """Test basic preparation position scoring."""
    result = score_preparation_position(sample_pose_data)

    assert 'shoulder_angle_right' in result
    assert 'shoulder_angle_left' in result
    assert 'weight_distribution' in result
    assert 'scores' in result
    assert 'preparation_score' in result

    # Check score is in valid range
    assert 0 <= result['preparation_score'] <= 100
    assert 0 <= result['scores']['shoulder_angle'] <= 100
    assert 0 <= result['scores']['weight_distribution'] <= 100


def test_score_preparation_position_missing_keypoints():
    """Test handling of missing keypoints."""
    incomplete_data = {
        'pred_keypoints_3d': np.random.rand(70, 3),
    }

    # Should handle gracefully
    result = score_preparation_position(incomplete_data)
    assert 'preparation_score' in result
    # Score should be lower when keypoints are missing
    assert result['preparation_score'] >= 0


def test_score_contact_position_basic(sample_pose_data):
    """Test basic contact position scoring."""
    result = score_contact_position(sample_pose_data)

    assert 'paddle_position' in result
    assert 'paddle_height' in result
    assert 'scores' in result
    assert 'contact_score' in result

    # Check score is in valid range
    assert 0 <= result['contact_score'] <= 100


def test_score_finish_position_basic(sample_pose_data):
    """Test basic finish position scoring."""
    result = score_finish_position(sample_pose_data)

    assert 'finish_position' in result
    assert 'follow_through_angle' in result
    assert 'scores' in result
    assert 'finish_score' in result

    # Check score is in valid range
    assert 0 <= result['finish_score'] <= 100
