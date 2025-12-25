"""
Teacher learning module for Phase 2 comparison system.

This module processes teacher pose examples and derives statistical ranges
(percentiles or normal distribution) for each metric.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..image_processor import process_single_image, process_three_positions
from .config_loader import (
    get_metrics_for_stroke,
    get_percentile_settings,
    get_single_pose_tolerance,
    load_comparison_config,
)
from .metric_extractor import extract_metrics


def learn_teacher_ranges(
    teacher_poses: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    stroke_type: str,
    config_path: str,
    output_path: str,
    estimator: Any | None = None
) -> dict[str, Any]:
    """
    Learn ideal ranges from teacher pose examples.

    Args:
        teacher_poses: Either:
            - List of pose data dicts with keys 'preparation', 'contact', 'finish' (three positions)
            - Dict with keys 'preparation', 'contact', 'finish', each containing list of pose data
        stroke_type: Name of stroke type (e.g., "serve")
        config_path: Path to configuration file
        output_path: Path to save learned ranges JSON
        estimator: Optional SAM3DBodyEstimator (required if teacher_poses contains image paths)

    Returns:
        Dictionary containing learned ranges for each position and metric

    Raises:
        ValueError: If teacher_poses format is invalid
        FileNotFoundError: If config_path doesn't exist
    """
    # Load configuration
    config = load_comparison_config(config_path)

    # Normalize teacher_poses format
    if isinstance(teacher_poses, list):
        # List of dicts with 'preparation', 'contact', 'finish' keys
        normalized_poses = {
            'preparation': [pose.get('preparation') for pose in teacher_poses if pose.get('preparation')],
            'contact': [pose.get('contact') for pose in teacher_poses if pose.get('contact')],
            'finish': [pose.get('finish') for pose in teacher_poses if pose.get('finish')],
        }
    elif isinstance(teacher_poses, dict):
        # Already in correct format
        normalized_poses = teacher_poses
    else:
        raise ValueError(
            "teacher_poses must be either a list of dicts or a dict with 'preparation', 'contact', 'finish' keys"
        )

    # Process images if needed (if teacher_poses contains image paths)
    processed_poses = _process_teacher_inputs(normalized_poses, estimator)

    # Learn ranges for each position
    learned_ranges = {
        'stroke_type': stroke_type,
        'preparation': {},
        'contact': {},
        'finish': {},
    }

    for position in ['preparation', 'contact', 'finish']:
        if position not in processed_poses or not processed_poses[position]:
            continue

        position_poses = processed_poses[position]
        metrics = get_metrics_for_stroke(config, stroke_type, position)

        # Extract metrics from all teacher poses for this position
        all_metric_values = {}
        for metric_key in metrics:
            all_metric_values[metric_key] = []

        for pose_data in position_poses:
            extracted = extract_metrics(pose_data, metrics, position)
            for metric_key, value in extracted.items():
                if value is not None:
                    all_metric_values[metric_key].append(value)

        # Calculate ranges for each metric
        for metric_key, values in all_metric_values.items():
            if not values:
                continue  # Skip if no valid values

            percentile_settings = get_percentile_settings(config)
            tolerance_percentage = get_single_pose_tolerance(config)

            range_data = calculate_metric_ranges(
                metric_values=values,
                method="percentile" if len(values) > 1 else "normal",
                percentile_low=percentile_settings['low'],
                percentile_high=percentile_settings['high'],
                tolerance_percentage=tolerance_percentage
            )

            learned_ranges[position][metric_key] = range_data

    # Save to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    serializable_ranges = _make_serializable(learned_ranges)

    with open(output_path_obj, 'w') as f:
        json.dump(serializable_ranges, f, indent=2)

    return learned_ranges


def _process_teacher_inputs(
    teacher_poses: dict[str, list[Any]],
    estimator: Any | None
) -> dict[str, list[dict[str, Any]]]:
    """
    Process teacher inputs - handle both image paths and pose data.

    Args:
        teacher_poses: Dict with position keys, values are lists of pose data or image paths
        estimator: Optional estimator for processing images

    Returns:
        Dict with processed pose data for each position
    """
    processed = {}

    for position, inputs in teacher_poses.items():
        processed[position] = []

        for input_item in inputs:
            if isinstance(input_item, str) or isinstance(input_item, Path):
                # Image path - need to process
                if estimator is None:
                    raise ValueError(
                        f"Estimator required to process image: {input_item}"
                    )
                pose_data = process_single_image(
                    estimator,
                    str(input_item),
                    position_name=position
                )
                processed[position].append(pose_data)
            elif isinstance(input_item, dict):
                # Already pose data
                processed[position].append(input_item)
            else:
                raise ValueError(
                    f"Invalid teacher input type: {type(input_item)}. "
                    "Expected image path (str) or pose data (dict)"
                )

    return processed


def calculate_metric_ranges(
    metric_values: list[float],
    method: str = "percentile",
    percentile_low: float = 25.0,
    percentile_high: float = 75.0,
    tolerance_percentage: float = 0.10
) -> dict[str, Any]:
    """
    Calculate statistical range for a metric.

    Args:
        metric_values: List of metric values from teacher poses
        method: "percentile" or "normal"
        percentile_low: Lower percentile (for percentile method)
        percentile_high: Upper percentile (for percentile method)
        tolerance_percentage: Percentage tolerance (for normal method with single value)

    Returns:
        Dictionary with:
        - min: Minimum value (or mean - tolerance for normal)
        - max: Maximum value (or mean + tolerance for normal)
        - mean: Mean value
        - std_dev: Standard deviation
        - method: Method used ("percentile" or "normal")
        - percentile_25, percentile_75: Percentile values (if percentile method)
        - sample_count: Number of samples
    """
    if not metric_values:
        raise ValueError("metric_values cannot be empty")

    values_array = np.array(metric_values)
    mean = float(np.mean(values_array))
    std_dev = float(np.std(values_array))
    sample_count = len(metric_values)

    if method == "percentile" and sample_count > 1:
        # Use percentiles for multiple samples
        min_val = float(np.percentile(values_array, percentile_low))
        max_val = float(np.percentile(values_array, percentile_high))

        return {
            'min': min_val,
            'max': max_val,
            'mean': mean,
            'std_dev': std_dev,
            'method': 'percentile',
            'percentile_low': percentile_low,
            'percentile_high': percentile_high,
            'percentile_25': float(np.percentile(values_array, 25.0)),
            'percentile_75': float(np.percentile(values_array, 75.0)),
            'sample_count': sample_count
        }
    else:
        # Use normal distribution for single sample
        # Range = mean ± (mean * tolerance_percentage)
        tolerance = mean * tolerance_percentage
        min_val = mean - tolerance
        max_val = mean + tolerance

        return {
            'min': min_val,
            'max': max_val,
            'mean': mean,
            'std_dev': std_dev,
            'method': 'normal',
            'tolerance_percentage': tolerance_percentage,
            'sample_count': sample_count
        }


def _make_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to make serializable

    Returns:
        Serializable version of object
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, np.integer | np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_teacher_ranges(ranges_path: str) -> dict[str, Any]:
    """
    Load learned teacher ranges from JSON file.

    Args:
        ranges_path: Path to learned ranges JSON file

    Returns:
        Dictionary containing learned ranges

    Raises:
        FileNotFoundError: If ranges_path doesn't exist
    """
    ranges_path_obj = Path(ranges_path)
    if not ranges_path_obj.exists():
        raise FileNotFoundError(f"Teacher ranges file not found: {ranges_path}")

    with open(ranges_path_obj) as f:
        ranges = json.load(f)

    return ranges
