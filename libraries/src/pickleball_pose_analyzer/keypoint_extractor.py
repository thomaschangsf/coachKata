"""
Keypoint extraction module for pickleball pose analysis.

This module provides functions to extract and work with specific keypoints
needed for pickleball analysis.
"""

from typing import Any

import numpy as np

from .data_structures import PICKLEBALL_KEYPOINTS


def extract_keypoints(pose_data: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Extract keypoints by name from pose data.

    Args:
        pose_data: Dictionary containing 'pred_keypoints_3d' (70, 3) array

    Returns:
        Dictionary mapping keypoint names to 3D coordinates
        Example: {'left_shoulder': array([x, y, z]), ...}
    """
    if 'pred_keypoints_3d' not in pose_data:
        raise ValueError("pose_data must contain 'pred_keypoints_3d'")

    keypoints_3d = pose_data['pred_keypoints_3d']
    if keypoints_3d.shape != (70, 3):
        raise ValueError(f"Expected keypoints_3d shape (70, 3), got {keypoints_3d.shape}")

    extracted = {}
    for name, idx in PICKLEBALL_KEYPOINTS.items():
        if idx < len(keypoints_3d):
            extracted[name] = keypoints_3d[idx].copy()
        else:
            extracted[name] = None

    return extracted


def calculate_body_center(pose_data: dict[str, Any]) -> np.ndarray:
    """
    Calculate body center as midpoint between left and right hips.

    Args:
        pose_data: Dictionary containing 'pred_keypoints_3d' (70, 3) array

    Returns:
        3D coordinates of body center: array([x, y, z])
    """
    keypoints = extract_keypoints(pose_data)

    left_hip = keypoints.get('left_hip')
    right_hip = keypoints.get('right_hip')

    if left_hip is None or right_hip is None:
        raise ValueError("Could not extract hip keypoints")

    # Calculate midpoint
    body_center = (left_hip + right_hip) / 2.0
    return body_center


def get_keypoint_by_name(pose_data: dict[str, Any], name: str) -> np.ndarray | None:
    """
    Get a specific keypoint by name.

    Args:
        pose_data: Dictionary containing 'pred_keypoints_3d'
        name: Keypoint name (e.g., 'left_shoulder')

    Returns:
        3D coordinates as numpy array, or None if not found
    """
    idx = PICKLEBALL_KEYPOINTS.get(name)
    if idx is None:
        return None

    keypoints_3d = pose_data.get('pred_keypoints_3d')
    if keypoints_3d is None:
        return None

    if idx >= len(keypoints_3d):
        return None

    return keypoints_3d[idx].copy()


def calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    Calculate 3D Euclidean distance between two points.

    Args:
        point_a: First point [x, y, z]
        point_b: Second point [x, y, z]

    Returns:
        Distance in meters
    """
    return float(np.linalg.norm(point_a - point_b))


def calculate_angle_between_vectors(
    vec_a: np.ndarray,
    vec_b: np.ndarray
) -> float:
    """
    Calculate angle between two 3D vectors in degrees.

    Args:
        vec_a: First vector
        vec_b: Second vector

    Returns:
        Angle in degrees (0-180)
    """
    # Normalize vectors
    vec_a_norm = vec_a / (np.linalg.norm(vec_a) + 1e-8)
    vec_b_norm = vec_b / (np.linalg.norm(vec_b) + 1e-8)

    # Calculate dot product and clip to [-1, 1] for numerical stability
    dot_product = np.clip(np.dot(vec_a_norm, vec_b_norm), -1.0, 1.0)

    # Calculate angle in radians, then convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)
