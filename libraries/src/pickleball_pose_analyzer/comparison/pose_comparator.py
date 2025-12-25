"""
Pose comparison module for Phase 2 comparison system.

This module provides the main PoseComparator class that compares student poses
against teacher-learned ranges.
"""

from typing import Any

from .config_loader import (
    get_metrics_for_stroke,
    get_scoring_method,
    get_weights_for_stroke,
    load_comparison_config,
)
from .metric_extractor import extract_metrics
from .scoring_v2 import score_metric
from .teacher_learner import load_teacher_ranges


class PoseComparator:
    """
    Compare student poses against teacher-learned ranges.

    Usage:
        estimator = load_sam3d_model(...)
        comparator = PoseComparator(
            teacher_ranges_path="path/to/ranges.json",
            config_path="path/to/config.json",
            estimator=estimator
        )
        results = comparator.compare(
            student_poses={
                'preparation': pose_data_dict,
                'contact': pose_data_dict,
                'finish': pose_data_dict
            },
            stroke_type="serve"
        )
    """

    def __init__(
        self,
        teacher_ranges_path: str,
        config_path: str,
        estimator: Any,
        stroke_type: str | None = None
    ):
        """
        Initialize comparator.

        Args:
            teacher_ranges_path: Path to learned teacher ranges JSON
            config_path: Path to comparison configuration JSON
            estimator: SAM 3D Body estimator (reused for all comparisons)
            stroke_type: Default stroke type (can be overridden in compare())
        """
        self.teacher_ranges_path = teacher_ranges_path
        self.config_path = config_path
        self.estimator = estimator
        self.default_stroke_type = stroke_type

        # Load configuration and teacher ranges
        self.config = load_comparison_config(config_path)
        self.teacher_ranges = load_teacher_ranges(teacher_ranges_path)

        # Validate that stroke_type exists in config
        if stroke_type and stroke_type not in self.config['stroke_types']:
            raise ValueError(
                f"Stroke type '{stroke_type}' not found in configuration. "
                f"Available: {list(self.config['stroke_types'].keys())}"
            )

    def compare(
        self,
        student_poses: dict[str, dict[str, Any]],
        stroke_type: str | None = None
    ) -> dict[str, Any]:
        """
        Compare student poses against teacher ranges.

        Args:
            student_poses: Dict with keys 'preparation', 'contact', 'finish'
                Each value is pose data dict from Phase 1
            stroke_type: Stroke type (uses default if None)

        Returns:
            Dictionary with comparison results:
            {
                'preparation': {
                    'scores': {...},
                    'preparation_score': float,
                    'metrics': {...},
                    'differences': {...}
                },
                'contact': {...},
                'finish': {...},
                'cumulative_score': float
            }
        """
        # Use provided stroke_type or default
        if stroke_type is None:
            stroke_type = self.default_stroke_type

        if stroke_type is None:
            raise ValueError(
                "stroke_type must be provided either in __init__ or compare()"
            )

        if stroke_type not in self.config['stroke_types']:
            raise ValueError(
                f"Stroke type '{stroke_type}' not found in configuration"
            )

        # Compare each position
        results: dict[str, Any] = {
            'preparation': {},
            'contact': {},
            'finish': {},
        }

        for position in ['preparation', 'contact', 'finish']:
            if position not in student_poses:
                continue

            student_pose = student_poses[position]
            position_results = self._compare_position(
                student_pose=student_pose,
                teacher_ranges=self.teacher_ranges.get(position, {}),
                position_name=position,
                stroke_type=stroke_type
            )
            results[position] = position_results

        # Calculate cumulative score
        cumulative_score = self._calculate_cumulative_score(results)

        results['cumulative_score'] = cumulative_score
        results['stroke_type'] = stroke_type

        return results

    def _compare_position(
        self,
        student_pose: dict[str, Any],
        teacher_ranges: dict[str, Any],
        position_name: str,
        stroke_type: str
    ) -> dict[str, Any]:
        """
        Compare a single position.

        Args:
            student_pose: Student pose data for this position
            teacher_ranges: Teacher ranges for this position
            position_name: Position name ("preparation", "contact", "finish")
            stroke_type: Stroke type

        Returns:
            Dictionary with comparison results for this position
        """
        # Get metrics to compare for this position
        metrics = get_metrics_for_stroke(self.config, stroke_type, position_name)

        # Extract student metrics
        student_metrics = extract_metrics(student_pose, metrics, position_name)

        # Compare each metric and calculate scores
        scores = {}
        differences = {}
        raw_metrics = {}

        for metric_key in metrics:
            student_value = student_metrics.get(metric_key)

            if student_value is None:
                # Metric unavailable - score 0
                scores[metric_key] = 0.0
                differences[metric_key] = None
                raw_metrics[metric_key] = None
                continue

            raw_metrics[metric_key] = student_value

            # Get teacher range for this metric
            teacher_range = teacher_ranges.get(metric_key)
            if teacher_range is None:
                # No teacher range available - score 0
                scores[metric_key] = 0.0
                differences[metric_key] = None
                continue

            # Get scoring method configuration
            scoring_config = get_scoring_method(self.config, metric_key)

            # Score the metric
            score = score_metric(student_value, teacher_range, scoring_config)
            scores[metric_key] = score

            # Calculate difference (for feedback)
            range_min = teacher_range.get('min', 0.0)
            range_max = teacher_range.get('max', 0.0)
            range_center = (range_min + range_max) / 2.0
            difference = student_value - range_center
            differences[metric_key] = difference

        # Calculate weighted overall score for this position
        weights = get_weights_for_stroke(self.config, stroke_type, position_name)
        overall_score = sum(
            scores.get(metric_key, 0.0) * weights.get(metric_key, 0.0)
            for metric_key in metrics
        )

        # Position-specific score key
        position_score_key = f'{position_name}_score'

        return {
            'scores': scores,
            position_score_key: overall_score,
            'metrics': raw_metrics,
            'differences': differences,
            'weights': weights,
        }

    def _calculate_cumulative_score(
        self,
        results: dict[str, dict[str, Any]]
    ) -> float:
        """
        Calculate cumulative score across all three positions.

        Uses equal weights for now (can be made configurable later).

        Args:
            results: Comparison results dictionary

        Returns:
            Cumulative score (0-100)
        """
        position_scores = []
        position_keys = ['preparation', 'contact', 'finish']

        for position in position_keys:
            if position not in results:
                continue

            position_result = results[position]
            # Get position-specific score key
            score_key = f'{position}_score'
            if score_key in position_result:
                position_scores.append(position_result[score_key])

        if not position_scores:
            return 0.0

        # Equal weights for now (can be made configurable)
        cumulative = sum(position_scores) / len(position_scores)
        return cumulative
