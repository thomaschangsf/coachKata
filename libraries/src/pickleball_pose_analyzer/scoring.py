"""
Scoring functions for pickleball pose analysis.

This module implements scoring functions for each position:
- Preparation position
- Contact position
- Finish position
"""

from typing import Any

import numpy as np

from .data_structures import IDEAL_RANGES, SCORING_WEIGHTS
from .keypoint_extractor import (
    calculate_angle_between_vectors,
    calculate_body_center,
    extract_keypoints,
)


def calculate_3d_angle(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray
) -> float:
    """
    Calculate 3D angle at point_b between vectors ba and bc.

    Args:
        point_a: First point [x, y, z]
        point_b: Vertex point [x, y, z]
        point_c: Third point [x, y, z]

    Returns:
        Angle in degrees (0-180)
    """
    # Calculate vectors
    vec_ba = point_a - point_b
    vec_bc = point_c - point_b

    # Calculate angle between vectors
    return calculate_angle_between_vectors(vec_ba, vec_bc)


def _score_metric(value: float, ideal_min: float, ideal_max: float, tolerance: float = 0.2) -> float:
    """
    Score a metric value based on ideal range.

    Args:
        value: Actual value
        ideal_min: Minimum ideal value
        ideal_max: Maximum ideal value
        tolerance: Fraction of range to allow outside ideal (0-1)

    Returns:
        Score from 0-100
    """
    ideal_center = (ideal_min + ideal_max) / 2.0
    ideal_range = ideal_max - ideal_min
    tolerance_range = ideal_range * tolerance

    # Calculate distance from ideal center
    distance = abs(value - ideal_center)

    # Score based on distance
    if distance <= ideal_range / 2:
        # Within ideal range: 100 points
        return 100.0
    elif distance <= ideal_range / 2 + tolerance_range:
        # Within tolerance: linear decrease from 100 to 50
        excess = distance - ideal_range / 2
        score = 100.0 - (excess / tolerance_range) * 50.0
        return max(50.0, score)
    else:
        # Outside tolerance: linear decrease from 50 to 0
        excess = distance - ideal_range / 2 - tolerance_range
        max_excess = ideal_range  # Maximum expected excess
        score = 50.0 - (excess / max_excess) * 50.0
        return max(0.0, score)


def score_preparation_position(pose_data: dict[str, Any]) -> dict[str, Any]:
    """
    Score preparation position.

    Metrics:
    - Shoulder angle (both left and right)
    - Weight distribution (hip height difference)
    - Knee bend angles

    Returns:
        {
            'shoulder_angle_right': float,  # degrees
            'shoulder_angle_left': float,
            'weight_distribution': float,  # 0-1 ratio
            'hip_height_diff': float,  # meters
            'knee_angles': dict,  # left and right knee angles
            'scores': {
                'shoulder_angle': float,  # 0-100
                'weight_distribution': float,  # 0-100
                'knee_bend': float,  # 0-100
            },
            'preparation_score': float  # 0-100 (weighted average)
        }
    """
    keypoints = extract_keypoints(pose_data)

    # Calculate shoulder angles
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
    avg_shoulder_angle = None
    if right_shoulder_angle is not None and left_shoulder_angle is not None:
        avg_shoulder_angle = (right_shoulder_angle + left_shoulder_angle) / 2.0
    elif right_shoulder_angle is not None:
        avg_shoulder_angle = right_shoulder_angle
    elif left_shoulder_angle is not None:
        avg_shoulder_angle = left_shoulder_angle

    # Calculate weight distribution (hip height difference)
    # Lower hip indicates more weight on that leg
    hip_height_diff = 0.0
    weight_distribution = 0.5  # Default: balanced
    if all(k in keypoints and keypoints[k] is not None
           for k in ['left_hip', 'right_hip']):
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

    # Calculate knee angles
    left_knee_angle = None
    if all(k in keypoints and keypoints[k] is not None
           for k in ['left_hip', 'left_knee', 'left_ankle']):
        left_knee_angle = calculate_3d_angle(
            keypoints['left_hip'],
            keypoints['left_knee'],
            keypoints['left_ankle']
        )

    right_knee_angle = None
    if all(k in keypoints and keypoints[k] is not None
           for k in ['right_hip', 'right_knee', 'right_ankle']):
        right_knee_angle = calculate_3d_angle(
            keypoints['right_hip'],
            keypoints['right_knee'],
            keypoints['right_ankle']
        )

    avg_knee_angle = None
    if left_knee_angle is not None and right_knee_angle is not None:
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2.0
    elif left_knee_angle is not None:
        avg_knee_angle = left_knee_angle
    elif right_knee_angle is not None:
        avg_knee_angle = right_knee_angle

    # Score each metric
    scores = {}

    # Shoulder angle score
    if avg_shoulder_angle is not None:
        ideal_range = IDEAL_RANGES['preparation']['shoulder_angle']
        scores['shoulder_angle'] = _score_metric(
            avg_shoulder_angle,
            ideal_range[0],
            ideal_range[1]
        )
    else:
        scores['shoulder_angle'] = 0.0

    # Weight distribution score
    ideal_range = IDEAL_RANGES['preparation']['weight_distribution']
    scores['weight_distribution'] = _score_metric(
        weight_distribution,
        ideal_range[0],
        ideal_range[1]
    )

    # Knee bend score
    if avg_knee_angle is not None:
        # Knee angle of 180 is straight, so we want it to be less (more bent)
        # Ideal is around 140-160 degrees (20-40 degrees of bend)
        knee_bend = 180.0 - avg_knee_angle
        ideal_range = IDEAL_RANGES['preparation']['knee_bend']
        scores['knee_bend'] = _score_metric(
            knee_bend,
            ideal_range[0],
            ideal_range[1]
        )
    else:
        scores['knee_bend'] = 0.0

    # Calculate weighted overall score
    weights = SCORING_WEIGHTS['preparation']
    overall_score = (
        scores['shoulder_angle'] * weights['shoulder_angle'] +
        scores['weight_distribution'] * weights['weight_distribution'] +
        scores['knee_bend'] * weights['knee_bend']
    )

    return {
        'shoulder_angle_right': right_shoulder_angle,
        'shoulder_angle_left': left_shoulder_angle,
        'weight_distribution': weight_distribution,
        'hip_height_diff': hip_height_diff,
        'knee_angles': {
            'left': left_knee_angle,
            'right': right_knee_angle,
        },
        'scores': scores,
        'preparation_score': overall_score,
    }


def score_contact_position(pose_data: dict[str, Any]) -> dict[str, Any]:
    """
    Score contact position.

    Metrics:
    - Paddle position (wrist position relative to body)
    - Paddle height (Z coordinate)
    - Body alignment (shoulder-hip alignment)
    - Torso angle (forward lean)

    Returns:
        {
            'paddle_position': np.ndarray,  # 3D position relative to body
            'paddle_height': float,  # Z-coordinate
            'torso_angle': float,  # degrees
            'body_alignment': float,  # alignment metric
            'scores': {
                'paddle_position': float,  # 0-100
                'body_alignment': float,  # 0-100
                'contact_height': float,  # 0-100
            },
            'contact_score': float  # 0-100 (weighted average)
        }
    """
    keypoints = extract_keypoints(pose_data)
    body_center = calculate_body_center(pose_data)

    # Get paddle position (assuming right hand holds paddle)
    paddle_position = None
    paddle_height = 0.0
    if 'right_wrist' in keypoints and keypoints['right_wrist'] is not None:
        paddle_position = keypoints['right_wrist']
        paddle_height = float(paddle_position[2])  # Z coordinate

    # Calculate paddle position relative to body center
    paddle_relative = None
    if paddle_position is not None:
        paddle_relative = paddle_position - body_center

    # Calculate torso angle (forward lean)
    # Angle between vertical and line from hip to shoulder
    torso_angle = None
    if all(k in keypoints and keypoints[k] is not None
           for k in ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']):
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

    # Calculate body alignment (shoulder-hip alignment in frontal plane)
    # Good alignment: shoulders and hips should be relatively aligned
    body_alignment = 0.0
    if all(k in keypoints and keypoints[k] is not None
           for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
        # Calculate shoulder and hip widths
        np.linalg.norm(
            keypoints['left_shoulder'] - keypoints['right_shoulder']
        )
        hip_width = np.linalg.norm(
            keypoints['left_hip'] - keypoints['right_hip']
        )

        # Calculate alignment: how well shoulders align with hips
        # Good alignment: shoulder center should be above hip center
        shoulder_center = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2.0
        hip_center = (keypoints['left_hip'] + keypoints['right_hip']) / 2.0

        # Project to XY plane (frontal view)
        shoulder_center_xy = shoulder_center[:2]
        hip_center_xy = hip_center[:2]

        # Alignment score: distance between shoulder and hip centers in XY
        alignment_distance = np.linalg.norm(shoulder_center_xy - hip_center_xy)
        # Normalize by body size (hip width)
        body_alignment = 1.0 - min(alignment_distance / (hip_width + 0.1), 1.0)

    # Score each metric
    scores = {}

    # Paddle position score (simplified: check if it's in front of body)
    if paddle_relative is not None:
        # Good position: paddle should be in front (positive X) and at reasonable height
        front_distance = float(paddle_relative[0])  # X coordinate
        if front_distance > 0.2:  # At least 20cm in front
            scores['paddle_position'] = 100.0
        elif front_distance > 0.1:
            scores['paddle_position'] = 75.0
        elif front_distance > 0.0:
            scores['paddle_position'] = 50.0
        else:
            scores['paddle_position'] = 25.0
    else:
        scores['paddle_position'] = 0.0

    # Contact height score
    if paddle_height > 0:
        ideal_range = IDEAL_RANGES['contact']['paddle_height']
        scores['contact_height'] = _score_metric(
            paddle_height,
            ideal_range[0],
            ideal_range[1]
        )
    else:
        scores['contact_height'] = 0.0

    # Body alignment score
    scores['body_alignment'] = body_alignment * 100.0

    # Torso angle score (optional, not in weights but can be included)
    if torso_angle is not None:
        ideal_range = IDEAL_RANGES['contact']['torso_angle']
        scores['torso_angle'] = _score_metric(
            torso_angle,
            ideal_range[0],
            ideal_range[1]
        )
    else:
        scores['torso_angle'] = 0.0

    # Calculate weighted overall score
    weights = SCORING_WEIGHTS['contact']
    overall_score = (
        scores['paddle_position'] * weights['paddle_position'] +
        scores['body_alignment'] * weights['body_alignment'] +
        scores['contact_height'] * weights['contact_height'] +
        scores.get('torso_angle', 0.0) * weights.get('torso_angle', 0.0)
    )

    return {
        'paddle_position': paddle_relative,
        'paddle_height': paddle_height,
        'torso_angle': torso_angle,
        'body_alignment': body_alignment,
        'scores': scores,
        'contact_score': overall_score,
    }


def score_finish_position(pose_data: dict[str, Any]) -> dict[str, Any]:
    """
    Score finish position.

    Metrics:
    - Finish position (wrist position)
    - Follow-through angle (shoulder-elbow-wrist angle)
    - Body rotation (shoulder alignment)

    Returns:
        {
            'finish_position': np.ndarray,  # 3D wrist position
            'follow_through_angle': float,  # degrees
            'body_rotation': float,  # rotation metric
            'scores': {
                'finish_position': float,  # 0-100
                'follow_through': float,  # 0-100
                'body_rotation': float,  # 0-100
            },
            'finish_score': float  # 0-100 (weighted average)
        }
    """
    keypoints = extract_keypoints(pose_data)

    # Get finish position (right wrist)
    finish_position = None
    wrist_height = 0.0
    if 'right_wrist' in keypoints and keypoints['right_wrist'] is not None:
        finish_position = keypoints['right_wrist']
        wrist_height = float(finish_position[2])  # Z coordinate

    # Calculate follow-through angle
    # Right shoulder -> right elbow -> right wrist
    follow_through_angle = None
    if all(k in keypoints and keypoints[k] is not None
           for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
        follow_through_angle = calculate_3d_angle(
            keypoints['right_shoulder'],
            keypoints['right_elbow'],
            keypoints['right_wrist']
        )

    # Calculate body rotation
    # Good rotation: shoulders should be rotated relative to hips
    body_rotation = 0.0
    if all(k in keypoints and keypoints[k] is not None
           for k in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']):
        # Calculate shoulder and hip vectors (in XY plane)
        shoulder_vec = keypoints['right_shoulder'][:2] - keypoints['left_shoulder'][:2]
        hip_vec = keypoints['right_hip'][:2] - keypoints['left_hip'][:2]

        # Calculate angle between shoulder and hip vectors
        # Good rotation: shoulders rotated relative to hips
        rotation_angle = calculate_angle_between_vectors(
            shoulder_vec,
            hip_vec
        )

        # Ideal rotation: around 30-60 degrees
        # Score based on how close to ideal
        if 30 <= rotation_angle <= 60:
            body_rotation = 1.0
        elif 20 <= rotation_angle < 30 or 60 < rotation_angle <= 70:
            body_rotation = 0.75
        elif 10 <= rotation_angle < 20 or 70 < rotation_angle <= 80:
            body_rotation = 0.5
        else:
            body_rotation = 0.25

    # Score each metric
    scores = {}

    # Finish position score (wrist height and position)
    if wrist_height > 0:
        ideal_range = IDEAL_RANGES['finish']['wrist_height']
        scores['finish_position'] = _score_metric(
            wrist_height,
            ideal_range[0],
            ideal_range[1]
        )
    else:
        scores['finish_position'] = 0.0

    # Follow-through angle score
    if follow_through_angle is not None:
        ideal_range = IDEAL_RANGES['finish']['follow_through_angle']
        scores['follow_through'] = _score_metric(
            follow_through_angle,
            ideal_range[0],
            ideal_range[1]
        )
    else:
        scores['follow_through'] = 0.0

    # Body rotation score
    scores['body_rotation'] = body_rotation * 100.0

    # Calculate weighted overall score
    weights = SCORING_WEIGHTS['finish']
    overall_score = (
        scores['finish_position'] * weights['finish_position'] +
        scores['follow_through'] * weights['follow_through'] +
        scores['body_rotation'] * weights['body_rotation']
    )

    return {
        'finish_position': finish_position,
        'follow_through_angle': follow_through_angle,
        'body_rotation': body_rotation,
        'scores': scores,
        'finish_score': overall_score,
    }
