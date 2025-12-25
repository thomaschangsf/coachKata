"""
Configuration loading module for Phase 2 comparison system.

This module handles loading and validating comparison configuration files
that define stroke types, metrics, scoring methods, and visualization settings.
"""

import json
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_comparison_config(config_path: str) -> dict[str, Any]:
    """
    Load comparison configuration from JSON or YAML file.

    Args:
        config_path: Path to configuration file (JSON or YAML)

    Returns:
        Configuration dictionary with:
        - stroke_types: dict mapping stroke names to metric configurations
        - scoring_methods: dict mapping metrics to scoring function configs
        - visualization_settings: dict with visualization options
        - single_pose_tolerance: dict with tolerance settings for single teacher pose
        - percentile_settings: dict with percentile configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid or missing required keys
    """
    config_path_obj = Path(config_path)
    if not config_path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load based on file extension
    if config_path_obj.suffix.lower() in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError(
                "YAML support requires PyYAML. Install with: pip install pyyaml"
            )
        # yaml is guaranteed to be available here due to HAS_YAML check
        assert yaml is not None  # type: ignore[reportUnnecessaryComparison]
        with open(config_path_obj) as f:
            config = yaml.safe_load(f)
    else:
        # Default to JSON
        with open(config_path_obj) as f:
            config = json.load(f)

    # Validate configuration
    validate_config(config)

    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate configuration structure.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['stroke_types']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Validate stroke_types
    if not isinstance(config['stroke_types'], dict):
        raise ValueError("'stroke_types' must be a dictionary")

    # Validate each stroke type
    for stroke_name, stroke_config in config['stroke_types'].items():
        if not isinstance(stroke_config, dict):
            raise ValueError(f"Stroke type '{stroke_name}' must be a dictionary")

        # Each stroke should have position configurations
        for position in ['preparation', 'contact', 'finish']:
            if position not in stroke_config:
                continue  # Position is optional

            position_config = stroke_config[position]
            if not isinstance(position_config, dict):
                raise ValueError(
                    f"Position '{position}' in stroke '{stroke_name}' must be a dictionary"
                )

            # Validate metrics list
            if 'metrics' not in position_config:
                raise ValueError(
                    f"Position '{position}' in stroke '{stroke_name}' missing 'metrics' key"
                )
            if not isinstance(position_config['metrics'], list):
                raise ValueError(
                    f"'metrics' in position '{position}' for stroke '{stroke_name}' must be a list"
                )

            # Validate weights if present
            if 'weights' in position_config:
                if not isinstance(position_config['weights'], dict):
                    raise ValueError(
                        f"'weights' in position '{position}' for stroke '{stroke_name}' must be a dictionary"
                    )
                # Check that weights sum to approximately 1.0 (allow small floating point errors)
                weight_sum = sum(position_config['weights'].values())
                if abs(weight_sum - 1.0) > 0.02:  # Allow 2% tolerance for floating point precision
                    raise ValueError(
                        f"Weights in position '{position}' for stroke '{stroke_name}' must sum to 1.0 (got {weight_sum:.6f})"
                    )


def get_metrics_for_stroke(
    config: dict[str, Any],
    stroke_type: str,
    position: str
) -> list[str]:
    """
    Get list of metrics to compare for a given stroke type and position.

    Args:
        config: Configuration dictionary
        stroke_type: Name of stroke (e.g., "serve", "forehand")
        position: Position name ("preparation", "contact", "finish")

    Returns:
        List of metric keys (e.g., ["shoulder_angle", "weight_distribution"])

    Raises:
        ValueError: If stroke_type or position not found in config
    """
    if stroke_type not in config['stroke_types']:
        raise ValueError(f"Stroke type '{stroke_type}' not found in configuration")

    stroke_config = config['stroke_types'][stroke_type]

    if position not in stroke_config:
        raise ValueError(
            f"Position '{position}' not found for stroke type '{stroke_type}'"
        )

    position_config = stroke_config[position]
    return position_config.get('metrics', [])


def get_weights_for_stroke(
    config: dict[str, Any],
    stroke_type: str,
    position: str
) -> dict[str, float]:
    """
    Get scoring weights for a given stroke type and position.

    Args:
        config: Configuration dictionary
        stroke_type: Name of stroke
        position: Position name

    Returns:
        Dictionary mapping metric names to weights
        Returns equal weights if not specified in config
    """
    if stroke_type not in config['stroke_types']:
        raise ValueError(f"Stroke type '{stroke_type}' not found in configuration")

    stroke_config = config['stroke_types'][stroke_type]

    if position not in stroke_config:
        raise ValueError(
            f"Position '{position}' not found for stroke type '{stroke_type}'"
        )

    position_config = stroke_config[position]
    weights = position_config.get('weights', {})

    # If no weights specified, return equal weights for all metrics
    if not weights:
        metrics = position_config.get('metrics', [])
        if metrics:
            equal_weight = 1.0 / len(metrics)
            return {metric: equal_weight for metric in metrics}

    return weights


def get_scoring_method(
    config: dict[str, Any],
    metric_name: str
) -> dict[str, Any]:
    """
    Get scoring method configuration for a metric.

    Args:
        config: Configuration dictionary
        metric_name: Name of metric (e.g., "shoulder_angle")

    Returns:
        Dictionary with scoring method configuration:
        - type: Scoring method type ("tolerance", "distance", "percentile")
        - Additional parameters specific to method type

    Returns default tolerance method if not specified.
    """
    scoring_methods = config.get('scoring_methods', {})
    method_config = scoring_methods.get(metric_name, {})

    # Default to tolerance method if not specified
    if not method_config:
        return {
            'type': 'tolerance',
            'tolerance': 0.2
        }

    return method_config


def get_visualization_settings(config: dict[str, Any]) -> dict[str, Any]:
    """
    Get visualization settings from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with visualization settings:
        - show_colors: bool
        - show_arrows: bool
        - color_thresholds: dict with 'good', 'fair', 'poor' thresholds
        - arrow_threshold: float (minimum difference to show arrow)
    """
    vis_config = config.get('visualization', {})

    # Default settings
    defaults = {
        'show_colors': True,
        'show_arrows': True,
        'color_thresholds': {
            'good': 80,
            'fair': 50,
            'poor': 0
        },
        'arrow_threshold': 5.0
    }

    # Merge with config (config overrides defaults)
    result = defaults.copy()
    result.update(vis_config)

    # Ensure color_thresholds is properly merged
    if 'color_thresholds' in vis_config:
        result['color_thresholds'] = {
            **defaults['color_thresholds'],
            **vis_config['color_thresholds']
        }

    return result


def get_single_pose_tolerance(config: dict[str, Any]) -> float:
    """
    Get tolerance percentage for single teacher pose (normal distribution).

    Args:
        config: Configuration dictionary

    Returns:
        Tolerance percentage (e.g., 0.10 for ±10%)
    """
    single_pose_config = config.get('single_pose_tolerance', {})
    return single_pose_config.get('percentage', 0.10)


def get_percentile_settings(config: dict[str, Any]) -> dict[str, float]:
    """
    Get percentile settings for range calculation.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with 'low' and 'high' percentile values
    """
    percentile_config = config.get('percentile_settings', {})
    return {
        'low': percentile_config.get('low', 25.0),
        'high': percentile_config.get('high', 75.0)
    }
