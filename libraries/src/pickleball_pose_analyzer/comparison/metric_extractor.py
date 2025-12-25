"""
Metric extraction module for Phase 2 comparison system.

This module extracts raw metrics from pose data without scoring them.
Metrics can be extracted based on configurable metric keys per stroke type.
"""

from typing import Any

import numpy as np

from ..keypoint_extractor import (
    calculate_angle_between_vectors,
    calculate_body_center,
    extract_keypoints,
)
from ..scoring import calculate_3d_angle


def extract_metrics(
    pose_data: dict[str, Any],
    metric_keys: list[str],
    position_name: str
) -> dict[str, float | None]:
    """
    Extract raw metrics from pose data.

    Args:
        pose_data: Pose data dictionary from Phase 1
        metric_keys: List of metric keys to extract (e.g., ["shoulder_angle", "weight_distribution"])
        position_name: Position name ("preparation", "contact", "finish")

    Returns:
        Dictionary mapping metric keys to values (or None if unavailable)
    """
    results = {}

    for metric_key in metric_keys:
        try:
            value = _extract_single_metric(pose_data, metric_key, position_name)
            results[metric_key] = value
        except (ValueError, KeyError, IndexError):
            # Metric extraction failed - set to None
            results[metric_key] = None

    return results


def _extract_single_metric(
    pose_data: dict[str, Any],
    metric_key: str,
    position_name: str
) -> float | None:
    """
    Extract a single metric by key.

    Args:
        pose_data: Pose data dictionary
        metric_key: Name of metric to extract
        position_name: Position name for context

    Returns:
        Metric value or None if unavailable
    """
    # Map metric keys to extraction functions
    metric_extractors = {
        # Preparation metrics
        'shoulder_angle': extract_shoulder_angle,
        'weight_distribution': extract_weight_distribution,
        'knee_bend': extract_knee_bend,
        # Contact metrics
        'paddle_position': extract_paddle_position,
        'body_alignment': extract_body_alignment,
        'contact_height': extract_contact_height,
        'torso_angle': extract_torso_angle,
        # Finish metrics
        'finish_position': extract_finish_position,
        'follow_through_angle': extract_follow_through_angle,
        'body_rotation': extract_body_rotation,
    }

    extractor = metric_extractors.get(metric_key)
    if extractor is None:
        raise ValueError(f"Unknown metric key: {metric_key}")

    return extractor(pose_data)


# Preparation metrics

def extract_shoulder_angle(pose_data: dict[str, Any]) -> float | None:
    """
    Extract average shoulder angle.

    Returns:
        Average shoulder angle in degrees, or None if unavailable
    """
    keypoints = extract_keypoints(pose_data)

    # Right shoulder angle: right_hip -> right_shoulder -> right_elbow
    right_shoulder_angle = None
    if all(k in keypoints and keypoints[k] is not None
           for k in ['right_hip', 'right_shoulder', 'right_elbow']):
        right_shoulder_angle = calculate_3d_angle(
            keypoints['right_hip'],
            keypoints['right_shoulder'],
            keypoints['right_elbow']
        )

    # Left shoulder angle: left_hip -> left_shoulder -> left_elbow
    left_shoulder_angle = None
    if all(k in keypoints and keypoints[k] is not None
           for k in ['left_hip', 'left_shoulder', 'left_elbow']):
        left_shoulder_angle = calculate_3d_angle(
            keypoints['left_hip'],
            keypoints['left_shoulder'],
            keypoints['left_elbow']
        )

    # Average shoulder angle
    if right_shoulder_angle is not None and left_shoulder_angle is not None:
        return (right_shoulder_angle + left_shoulder_angle) / 2.0
    elif right_shoulder_angle is not None:
        return right_shoulder_angle
    elif left_shoulder_angle is not None:
        return left_shoulder_angle
    else:
        return None


def extract_weight_distribution(pose_data: dict[str, Any]) -> float | None:
    """
    Extract weight distribution ratio.

    Returns:
        Weight distribution ratio (0-1), where 0.5 is balanced.
        >0.5 means more weight on right leg, <0.5 means more on left leg.
        Returns None if unavailable.
    """
    keypoints = extract_keypoints(pose_data)

    if not all(k in keypoints and keypoints[k] is not None
               for k in ['left_hip', 'right_hip']):
        return None

    left_hip_z = keypoints['left_hip'][2]  # Z coordinate (height)
    right_hip_z = keypoints['right_hip'][2]
    hip_height_diff = abs(left_hip_z - right_hip_z)

    # Weight distribution: ratio on back leg (assuming right is back for right-handed)
    # If right hip is lower, more weight on right leg
    if right_hip_z < left_hip_z:
        weight_distribution = 0.5 + (hip_height_diff / 0.1) * 0.3  # Cap at 0.8
        weight_distribution = min(0.8, weight_distribution)
    else:
        weight_distribution = 0.5 - (hip_height_diff / 0.1) * 0.3  # Cap at 0.2
        weight_distribution = max(0.2, weight_distribution)

    return weight_distribution


def extract_knee_bend(pose_data: dict[str, Any]) -> float | None:
    """
    Extract average knee bend angle.

    Returns:
        Knee bend in degrees (180° = straight, lower = more bent).
        Returns None if unavailable.
    """
    keypoints = extract_keypoints(pose_data)

    # Left knee angle
    left_knee_angle = None
    if all(k in keypoints and keypoints[k] is not None
           for k in ['left_hip', 'left_knee', 'left_ankle']):
        left_knee_angle = calculate_3d_angle(
            keypoints['left_hip'],
            keypoints['left_knee'],
            keypoints['left_ankle']
        )

    # Right knee angle
    right_knee_angle = None
    if all(k in keypoints and keypoints[k] is not None
           for k in ['right_hip', 'right_knee', 'right_ankle']):
        right_knee_angle = calculate_3d_angle(
            keypoints['right_hip'],
            keypoints['right_knee'],
            keypoints['right_ankle']
        )

    # Average knee angle
    avg_knee_angle = None
    if left_knee_angle is not None and right_knee_angle is not None:
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0
    elif left_knee_angle is not None:
        avg_knee_angle = left_knee_angle
    elif right_knee_angle is not None:
        avg_knee_angle = right_knee_angle
    else:
        return None

    # Convert to knee bend (180° is straight, so bend = 180 - angle)
    knee_bend = 180.0 - avg_knee_angle
    return knee_bend


# Contact metrics

def extract_paddle_position(pose_data: dict[str, Any]) -> float | None:
    """
    Extract paddle position metric (distance from wrist to body center).

    Returns:
        Distance in meters from right wrist to body center, or None if unavailable.
    """
    keypoints = extract_keypoints(pose_data)

    if 'right_wrist' not in keypoints or keypoints['right_wrist'] is None:
        return None

    try:
        body_center = calculate_body_center(pose_data)
        wrist_pos = keypoints['right_wrist']
        distance = float(np.linalg.norm(wrist_pos - body_center))
        return distance
    except (ValueError, KeyError):
        return None


def extract_body_alignment(pose_data: dict[str, Any]) -> float | None:
    """
    Extract body alignment metric (shoulder-hip alignment in frontal plane).

    Returns:
        Alignment value (0-1), where 1.0 is perfect alignment, or None if unavailable.
    """
    keypoints = extract_keypoints(pose_data)

    if not all(k in keypoints and keypoints[k] is not None
               for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
        return None

    # Calculate shoulder and hip vectors (in XY plane - frontal view)
    shoulder_vec = keypoints['right_shoulder'][:2] - keypoints['left_shoulder'][:2]
    hip_vec = keypoints['right_hip'][:2] - keypoints['left_hip'][:2]

    # Calculate angle between shoulder and hip vectors
    angle = calculate_angle_between_vectors(
        np.append(shoulder_vec, 0),  # Add z=0 for 3D vector
        np.append(hip_vec, 0)
    )

    # Good alignment: angle close to 0 (shoulders and hips aligned)
    # Convert angle to alignment score (0-1 scale)
    # 0° = perfect alignment (1.0), 90° = perpendicular (0.0)
    alignment = 1.0 - (angle / 90.0)
    alignment = max(0.0, min(1.0, alignment))  # Clip to [0, 1]

    return alignment


def extract_contact_height(pose_data: dict[str, Any]) -> float | None:
    """
    Extract contact height (paddle height at contact point).

    Returns:
        Height in meters (Z coordinate of right wrist), or None if unavailable.
    """
    keypoints = extract_keypoints(pose_data)

    if 'right_wrist' not in keypoints or keypoints['right_wrist'] is None:
        return None

    # Z coordinate represents height (assuming camera coordinate system)
    return float(keypoints['right_wrist'][2])


def extract_torso_angle(pose_data: dict[str, Any]) -> float | None:
    """
    Extract torso angle (forward lean angle).

    Returns:
        Torso angle in degrees from vertical (positive = leaning forward), or None if unavailable.
    """
    keypoints = extract_keypoints(pose_data)

    if not all(k in keypoints and keypoints[k] is not None
               for k in ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']):
        return None

    # Calculate shoulder center and hip center
    shoulder_center = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2.0
    hip_center = (keypoints['left_hip'] + keypoints['right_hip']) / 2.0

    # Vector from hip to shoulder
    torso_vec = shoulder_center - hip_center

    # Vertical vector (pointing up)
    vertical = np.array([0, 0, 1])

    # Calculate angle from vertical
    torso_angle = calculate_angle_between_vectors(torso_vec, vertical)
    # Adjust: 90 degrees is horizontal, so we want angle from vertical
    torso_angle = 90.0 - torso_angle  # Positive = leaning forward

    return torso_angle


# Finish metrics

def extract_finish_position(pose_data: dict[str, Any]) -> float | None:
    """
    Extract finish position metric (wrist height at finish).

    Returns:
        Height in meters (Z coordinate of right wrist), or None if unavailable.
    """
    keypoints = extract_keypoints(pose_data)

    if 'right_wrist' not in keypoints or keypoints['right_wrist'] is None:
        return None

    # Z coordinate represents height
    return float(keypoints['right_wrist'][2])


def extract_follow_through_angle(pose_data: dict[str, Any]) -> float | None:
    """
    Extract follow-through angle (shoulder-elbow-wrist angle at finish).

    Returns:
        Follow-through angle in degrees, or None if unavailable.
    """
    keypoints = extract_keypoints(pose_data)

    # Right arm follow-through: right_shoulder -> right_elbow -> right_wrist
    if all(k in keypoints and keypoints[k] is not None
           for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        return calculate_3d_angle(
            keypoints['right_shoulder'],
            keypoints['right_elbow'],
            keypoints['right_wrist']
        )

    return None


def extract_body_rotation(pose_data: dict[str, Any]) -> float | None:
    """
    Extract body rotation metric (shoulder-hip rotation in XY plane).

    Returns:
        Rotation value (0-1), where 1.0 is ideal rotation, or None if unavailable.
    """
    keypoints = extract_keypoints(pose_data)

    if not all(k in keypoints and keypoints[k] is not None
               for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
        return None

    # Calculate shoulder and hip vectors (in XY plane)
    shoulder_vec = keypoints['right_shoulder'][:2] - keypoints['left_shoulder'][:2]
    hip_vec = keypoints['right_hip'][:2] - keypoints['left_hip'][:2]

    # Calculate angle between shoulder and hip vectors
    # Good rotation: shoulders rotated relative to hips
    rotation_angle = calculate_angle_between_vectors(
        np.append(shoulder_vec, 0),  # Add z=0 for 3D vector
        np.append(hip_vec, 0)
    )

    # Ideal rotation: around 30-60 degrees
    # Score based on how close to ideal
    if 30 <= rotation_angle <= 60:
        return 1.0
    elif 20 <= rotation_angle < 30 or 60 < rotation_angle <= 70:
        return 0.75
    elif 10 <= rotation_angle < 20 or 70 < rotation_angle <= 80:
        return 0.5
    else:
        return 0.25
