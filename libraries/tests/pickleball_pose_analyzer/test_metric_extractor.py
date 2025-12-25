"""Tests for metric_extractor module."""

import numpy as np
import pytest
from pickleball_pose_analyzer.comparison.metric_extractor import (
    extract_body_alignment,
    extract_body_rotation,
    extract_contact_height,
    extract_finish_position,
    extract_follow_through_angle,
    extract_knee_bend,
    extract_metrics,
    extract_paddle_position,
    extract_shoulder_angle,
    extract_torso_angle,
    extract_weight_distribution,
)


@pytest.fixture
def sample_pose_data_with_keypoints():
    """Sample pose data with realistic keypoint positions."""
    # Create keypoints array (70 keypoints)
    keypoints_3d = np.zeros((70, 3), dtype=np.float32)

    # Set some key keypoints for testing
    # Right shoulder (index 16), right elbow (index 18), right wrist (index 20)
    # Left shoulder (index 15), left elbow (index 17), left wrist (index 19)
    # Right hip (index 2), left hip (index 1)
    # Right knee (index 4), left knee (index 3)
    # Right ankle (index 6), left ankle (index 5)

    # Right arm: shoulder -> elbow -> wrist
    keypoints_3d[16] = [0.15, -0.4, 1.2]  # right_shoulder
    keypoints_3d[18] = [0.25, -0.3, 1.15]  # right_elbow
    keypoints_3d[20] = [0.35, -0.2, 1.1]  # right_wrist

    # Left arm
    keypoints_3d[15] = [0.05, -0.4, 1.2]  # left_shoulder
    keypoints_3d[17] = [-0.05, -0.3, 1.15]  # left_elbow
    keypoints_3d[19] = [-0.15, -0.2, 1.1]  # left_wrist

    # Hips
    keypoints_3d[2] = [0.12, -0.6, 1.0]  # right_hip
    keypoints_3d[1] = [0.08, -0.6, 1.0]  # left_hip

    # Knees
    keypoints_3d[4] = [0.13, -0.8, 0.9]  # right_knee
    keypoints_3d[3] = [0.07, -0.8, 0.9]  # left_knee

    # Ankles
    keypoints_3d[6] = [0.14, -1.0, 0.8]  # right_ankle
    keypoints_3d[5] = [0.06, -1.0, 0.8]  # left_ankle

    return {
        'pred_keypoints_3d': keypoints_3d,
        'pred_keypoints_2d': keypoints_3d[:, :2],
    }


def test_extract_metrics_basic(sample_pose_data_with_keypoints):
    """Test basic metric extraction."""
    metrics = extract_metrics(
        sample_pose_data_with_keypoints,
        ['shoulder_angle', 'weight_distribution'],
        'preparation'
    )

    assert isinstance(metrics, dict)
    assert 'shoulder_angle' in metrics
    assert 'weight_distribution' in metrics


def test_extract_metrics_unknown_metric(sample_pose_data_with_keypoints):
    """Test extraction with unknown metric returns None."""
    metrics = extract_metrics(
        sample_pose_data_with_keypoints,
        ['unknown_metric'],
        'preparation'
    )

    assert metrics['unknown_metric'] is None


def test_extract_shoulder_angle(sample_pose_data_with_keypoints):
    """Test shoulder angle extraction."""
    angle = extract_shoulder_angle(sample_pose_data_with_keypoints)

    assert angle is not None
    assert isinstance(angle, float)
    assert 0 <= angle <= 180


def test_extract_shoulder_angle_missing_keypoints():
    """Test shoulder angle extraction with missing keypoints."""
    # Create pose data with NaN keypoints (more realistic missing case)
    keypoints_3d = np.full((70, 3), np.nan, dtype=np.float32)
    pose_data = {
        'pred_keypoints_3d': keypoints_3d,
    }
    angle = extract_shoulder_angle(pose_data)

    # Should return None when keypoints are missing/invalid
    # Note: NaN values might still calculate, so check for None or NaN
    assert angle is None or (isinstance(angle, float) and np.isnan(angle))


def test_extract_weight_distribution(sample_pose_data_with_keypoints):
    """Test weight distribution extraction."""
    distribution = extract_weight_distribution(sample_pose_data_with_keypoints)

    assert distribution is not None
    assert isinstance(distribution, float)
    assert 0.0 <= distribution <= 1.0


def test_extract_knee_bend(sample_pose_data_with_keypoints):
    """Test knee bend extraction."""
    knee_bend = extract_knee_bend(sample_pose_data_with_keypoints)

    assert knee_bend is not None
    assert isinstance(knee_bend, float)
    assert knee_bend >= 0  # Knee bend should be positive


def test_extract_paddle_position(sample_pose_data_with_keypoints):
    """Test paddle position extraction."""
    position = extract_paddle_position(sample_pose_data_with_keypoints)

    assert position is not None
    assert isinstance(position, float)
    assert position >= 0  # Distance should be positive


def test_extract_body_alignment(sample_pose_data_with_keypoints):
    """Test body alignment extraction."""
    alignment = extract_body_alignment(sample_pose_data_with_keypoints)

    assert alignment is not None
    assert isinstance(alignment, float)
    assert 0.0 <= alignment <= 1.0


def test_extract_contact_height(sample_pose_data_with_keypoints):
    """Test contact height extraction."""
    height = extract_contact_height(sample_pose_data_with_keypoints)

    assert height is not None
    assert isinstance(height, float)


def test_extract_torso_angle(sample_pose_data_with_keypoints):
    """Test torso angle extraction."""
    angle = extract_torso_angle(sample_pose_data_with_keypoints)

    assert angle is not None
    assert isinstance(angle, float)


def test_extract_finish_position(sample_pose_data_with_keypoints):
    """Test finish position extraction."""
    position = extract_finish_position(sample_pose_data_with_keypoints)

    assert position is not None
    assert isinstance(position, float)


def test_extract_follow_through_angle(sample_pose_data_with_keypoints):
    """Test follow-through angle extraction."""
    angle = extract_follow_through_angle(sample_pose_data_with_keypoints)

    assert angle is not None
    assert isinstance(angle, float)
    assert 0 <= angle <= 180


def test_extract_body_rotation(sample_pose_data_with_keypoints):
    """Test body rotation extraction."""
    rotation = extract_body_rotation(sample_pose_data_with_keypoints)

    assert rotation is not None
    assert isinstance(rotation, float)
    assert 0.0 <= rotation <= 1.0
