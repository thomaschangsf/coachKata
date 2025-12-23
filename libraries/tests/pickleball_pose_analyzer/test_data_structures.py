"""Tests for data_structures module."""

import numpy as np
import pytest
from pickleball_pose_analyzer.data_structures import (
    KEYPOINT_NAMES_TO_INDICES,
    PICKLEBALL_KEYPOINTS,
    PoseData,
    PositionScore,
)


def test_pickleball_keypoints_defined():
    """Test that keypoint mappings are defined."""
    assert len(PICKLEBALL_KEYPOINTS) > 0
    assert 'left_shoulder' in PICKLEBALL_KEYPOINTS
    assert 'right_shoulder' in PICKLEBALL_KEYPOINTS
    assert 'left_hip' in PICKLEBALL_KEYPOINTS
    assert 'right_hip' in PICKLEBALL_KEYPOINTS


def test_keypoint_indices_valid():
    """Test that keypoint indices are within valid range (0-69 for mhr70)."""
    for name, idx in PICKLEBALL_KEYPOINTS.items():
        assert 0 <= idx < 70, f"Keypoint {name} has invalid index {idx}"


def test_pose_data_dataclass(sample_pose_data):
    """Test PoseData dataclass."""
    pose_data = PoseData(
        position_name='test',
        keypoints_3d=sample_pose_data['pred_keypoints_3d'],
        keypoints_2d=sample_pose_data['pred_keypoints_2d'],
        vertices=sample_pose_data['pred_vertices'],
        body_pose_params=sample_pose_data['body_pose_params'],
        hand_pose_params=sample_pose_data['hand_pose_params'],
        shape_params=sample_pose_data['shape_params'],
        scale_params=sample_pose_data['scale_params'],
        global_rots=sample_pose_data['pred_global_rots'],
        bbox=sample_pose_data['bbox'],
        focal_length=sample_pose_data['focal_length'],
        metadata={'test': 'data'},
    )

    assert pose_data.position_name == 'test'
    assert pose_data.keypoints_3d.shape == (70, 3)
    assert pose_data.get_keypoint('left_shoulder') is not None


def test_position_score_dataclass():
    """Test PositionScore dataclass."""
    score = PositionScore(
        position_name='preparation',
        metrics={'shoulder_angle': 90.0, 'weight_dist': 0.5},
        component_scores={'shoulder_angle': 85.0, 'weight_dist': 80.0},
        overall_score=82.5,
    )

    assert score.position_name == 'preparation'
    assert score.overall_score == 82.5
    assert 0 <= score.overall_score <= 100

    # Test to_dict
    score_dict = score.to_dict()
    assert 'preparation_score' in score_dict
    assert score_dict['preparation_score'] == 82.5
