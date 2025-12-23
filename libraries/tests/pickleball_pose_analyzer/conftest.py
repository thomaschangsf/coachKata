"""Pytest fixtures and shared test utilities for pickleball_pose_analyzer tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Setup paths to import pickleball_pose_analyzer
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "libraries" / "src"))

# Setup paths for sam_3d_body (if needed for some tests)
sam3d_path = project_root / "models" / "sam-3d-body"
if sam3d_path.exists() and str(sam3d_path) not in sys.path:
    sys.path.insert(0, str(sam3d_path))


@pytest.fixture
def sample_pose_data():
    """Sample pose data dictionary for testing."""
    return {
        'pred_keypoints_3d': np.random.rand(70, 3).astype(np.float32),
        'pred_keypoints_2d': np.random.rand(70, 2).astype(np.float32),
        'pred_vertices': np.random.rand(18439, 3).astype(np.float32),
        'body_pose_params': np.random.rand(260).astype(np.float32),
        'hand_pose_params': np.random.rand(108).astype(np.float32),
        'shape_params': np.random.rand(45).astype(np.float32),
        'scale_params': np.random.rand(28).astype(np.float32),
        'pred_global_rots': np.random.rand(127, 3, 3).astype(np.float32),
        'bbox': np.array([100.0, 50.0, 300.0, 500.0], dtype=np.float32),
        'focal_length': 800.5,
    }


@pytest.fixture
def sample_keypoints_dict():
    """Sample keypoints dictionary for testing."""
    return {
        'left_shoulder': np.array([0.1, -0.4, 1.2], dtype=np.float32),
        'right_shoulder': np.array([0.15, -0.4, 1.2], dtype=np.float32),
        'left_elbow': np.array([0.05, -0.3, 1.15], dtype=np.float32),
        'right_elbow': np.array([0.2, -0.3, 1.15], dtype=np.float32),
        'left_wrist': np.array([0.0, -0.2, 1.1], dtype=np.float32),
        'right_wrist': np.array([0.25, -0.2, 1.1], dtype=np.float32),
        'left_hip': np.array([0.08, -0.6, 1.0], dtype=np.float32),
        'right_hip': np.array([0.12, -0.6, 1.0], dtype=np.float32),
        'left_knee': np.array([0.07, -0.8, 0.9], dtype=np.float32),
        'right_knee': np.array([0.13, -0.8, 0.9], dtype=np.float32),
        'left_ankle': np.array([0.06, -1.0, 0.8], dtype=np.float32),
        'right_ankle': np.array([0.14, -1.0, 0.8], dtype=np.float32),
    }
