"""
Data structures and constants for pickleball pose analysis.

This module defines:
- Keypoint mappings for pickleball-specific analysis
- Data classes for pose data and scores
- Constants for scoring weights and thresholds
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

# Keypoint indices for pickleball analysis (based on mhr70 format)
# These map to the 70 keypoints defined in sam_3d_body/metadata/mhr70.py
PICKLEBALL_KEYPOINTS = {
    # Upper body
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 62,
    'right_wrist': 41,
    'neck': 69,

    # Lower body
    'left_hip': 9,
    'right_hip': 10,
    'left_knee': 11,
    'right_knee': 12,
    'left_ankle': 13,
    'right_ankle': 14,

    # Additional useful keypoints
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
}

# Reverse mapping: name -> index for easy lookup
KEYPOINT_NAMES_TO_INDICES = PICKLEBALL_KEYPOINTS.copy()

# Reverse mapping: index -> name
KEYPOINT_INDICES_TO_NAMES = {v: k for k, v in PICKLEBALL_KEYPOINTS.items()}


@dataclass
class PoseData:
    """Structured pose data container for a single image."""

    position_name: str
    keypoints_3d: np.ndarray  # (70, 3) - 3D keypoint coordinates
    keypoints_2d: np.ndarray  # (70, 2) - 2D projections
    vertices: np.ndarray  # (18439, 3) - Full 3D mesh vertices
    body_pose_params: np.ndarray  # Body pose parameters
    hand_pose_params: np.ndarray  # Hand pose parameters (108-dim: 54 per hand)
    shape_params: np.ndarray  # Body shape parameters (45-dim)
    scale_params: np.ndarray  # Scale parameters (28-dim)
    global_rots: np.ndarray  # (127, 3, 3) - Joint rotation matrices
    bbox: np.ndarray  # Bounding box [x1, y1, x2, y2]
    focal_length: float  # Camera focal length
    metadata: dict[str, Any]  # Additional metadata (image_path, timestamp, etc.)

    def get_keypoint(self, name: str) -> np.ndarray | None:
        """Get 3D keypoint by name."""
        idx = PICKLEBALL_KEYPOINTS.get(name)
        if idx is None:
            return None
        if idx >= len(self.keypoints_3d):
            return None
        return self.keypoints_3d[idx]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'position_name': self.position_name,
            'pred_keypoints_3d': self.keypoints_3d,
            'pred_keypoints_2d': self.keypoints_2d,
            'pred_vertices': self.vertices,
            'body_pose_params': self.body_pose_params,
            'hand_pose_params': self.hand_pose_params,
            'shape_params': self.shape_params,
            'scale_params': self.scale_params,
            'pred_global_rots': self.global_rots,
            'bbox': self.bbox,
            'focal_length': self.focal_length,
            'metadata': self.metadata,
        }


@dataclass
class PositionScore:
    """Score data for a single position."""

    position_name: str
    metrics: dict[str, float]  # Raw metrics (angles, distances, etc.)
    component_scores: dict[str, float]  # Individual component scores (0-100)
    overall_score: float  # Weighted overall score (0-100)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'position_name': self.position_name,
            'metrics': self.metrics,
            'scores': self.component_scores,
            f'{self.position_name}_score': self.overall_score,
        }


# Scoring weights for each position (sum to 1.0)
SCORING_WEIGHTS = {
    'preparation': {
        'shoulder_angle': 0.4,
        'weight_distribution': 0.3,
        'knee_bend': 0.2,
        'body_alignment': 0.1,
    },
    'contact': {
        'paddle_position': 0.35,
        'body_alignment': 0.3,
        'contact_height': 0.25,
        'torso_angle': 0.1,
    },
    'finish': {
        'finish_position': 0.4,
        'follow_through': 0.35,
        'body_rotation': 0.25,
    },
}

# Ideal ranges for metrics (used in scoring)
IDEAL_RANGES = {
    'preparation': {
        'shoulder_angle': (80, 120),  # degrees - good shoulder turn
        'weight_distribution': (0.4, 0.6),  # ratio on back leg
        'knee_bend': (20, 40),  # degrees - slight bend
    },
    'contact': {
        'paddle_height': (0.8, 1.2),  # meters - contact height
        'torso_angle': (5, 15),  # degrees - slight forward lean
    },
    'finish': {
        'follow_through_angle': (30, 60),  # degrees
        'wrist_height': (1.0, 1.5),  # meters
    },
}
