"""
Configurable scoring module for Phase 2 comparison system.

This module provides multiple scoring methods that can be configured per metric
via the configuration file. Supports tolerance-based, distance-based, and
percentile-based scoring.
"""

from typing import Any

import numpy as np


def score_metric(
    value: float,
    teacher_range: dict[str, float],
    scoring_config: dict[str, Any]
) -> float:
    """
    Score a metric value against teacher range using configurable method.

    Args:
        value: Student metric value
        teacher_range: Dictionary with 'min', 'max', 'mean', etc. from learned ranges
        scoring_config: Scoring method configuration from config file

    Returns:
        Score from 0-100
    """
    method_type = scoring_config.get('type', 'tolerance')

    if method_type == 'tolerance':
        return score_metric_tolerance(
            value,
            teacher_range['min'],
            teacher_range['max'],
            tolerance=scoring_config.get('tolerance', 0.2)
        )
    elif method_type == 'distance':
        return score_metric_distance(
            value,
            teacher_range['min'],
            teacher_range['max'],
            max_distance=scoring_config.get('max_distance', 0.1)
        )
    elif method_type == 'percentile':
        # Percentile-based scoring requires original teacher values
        if 'teacher_values' not in teacher_range:
            # Fall back to tolerance if teacher values not available
            return score_metric_tolerance(
                value,
                teacher_range['min'],
                teacher_range['max'],
                tolerance=scoring_config.get('tolerance', 0.2)
            )
        teacher_values = teacher_range.get('teacher_values')
        if not isinstance(teacher_values, list):
            # Fall back to tolerance if teacher_values is not a list
            return score_metric_tolerance(
                value,
                teacher_range['min'],
                teacher_range['max'],
                tolerance=scoring_config.get('tolerance', 0.2)
            )
        return score_metric_percentile(
            value,
            teacher_values,
            percentile_low=scoring_config.get('percentile_low', 25.0),
            percentile_high=scoring_config.get('percentile_high', 75.0)
        )
    else:
        raise ValueError(f"Unknown scoring method type: {method_type}")


def score_metric_tolerance(
    value: float,
    range_min: float,
    range_max: float,
    tolerance: float = 0.2
) -> float:
    """
    Score using tolerance-based method (similar to Phase 1 _score_metric).

    Scoring logic:
    - Within ideal range [min, max]: 100 points
    - Within tolerance (20% beyond range): linear decrease from 100 to 50
    - Outside tolerance: linear decrease from 50 to 0

    Args:
        value: Student metric value
        range_min: Minimum of teacher range
        range_max: Maximum of teacher range
        tolerance: Fraction of range to allow outside ideal (0-1)

    Returns:
        Score from 0-100
    """
    ideal_center = (range_min + range_max) / 2.0
    ideal_range = range_max - range_min
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


def score_metric_distance(
    value: float,
    range_min: float,
    range_max: float,
    max_distance: float
) -> float:
    """
    Score using distance-based method.

    Scoring logic:
    - If value is within [min, max]: 100 points
    - If value is outside range: score decreases based on distance
    - Score = 100 * (1 - min(distance / max_distance, 1.0))

    Args:
        value: Student metric value
        range_min: Minimum of teacher range
        range_max: Maximum of teacher range
        max_distance: Maximum acceptable distance from range

    Returns:
        Score from 0-100
    """
    # Check if value is within range
    if range_min <= value <= range_max:
        return 100.0

    # Calculate distance from range
    if value < range_min:
        distance = range_min - value
    else:  # value > range_max
        distance = value - range_max

    # Score based on distance
    if distance >= max_distance:
        return 0.0

    # Linear decrease: score = 100 * (1 - distance / max_distance)
    score = 100.0 * (1.0 - distance / max_distance)
    return max(0.0, score)


def score_metric_percentile(
    value: float,
    teacher_values: list[float],
    percentile_low: float = 25.0,
    percentile_high: float = 75.0
) -> float:
    """
    Score based on percentile position within teacher distribution.

    Scoring logic:
    - If value is within [percentile_low, percentile_high]: 100 points
    - If value is at extremes: score decreases based on how far from percentiles

    Args:
        value: Student metric value
        teacher_values: List of teacher metric values
        percentile_low: Lower percentile threshold
        percentile_high: Upper percentile threshold

    Returns:
        Score from 0-100
    """
    if not teacher_values:
        return 0.0

    values_array = np.array(teacher_values)
    p_low = float(np.percentile(values_array, percentile_low))
    p_high = float(np.percentile(values_array, percentile_high))

    # Check if value is within percentile range
    if p_low <= value <= p_high:
        return 100.0

    # Calculate distance from percentile range
    if value < p_low:
        distance = p_low - value
        range_size = p_high - p_low
    else:  # value > p_high
        distance = value - p_high
        range_size = p_high - p_low

    # Score decreases based on distance relative to range size
    # If distance > range_size, score approaches 0
    if distance >= range_size:
        return 0.0

    score = 100.0 * (1.0 - distance / range_size)
    return max(0.0, score)
