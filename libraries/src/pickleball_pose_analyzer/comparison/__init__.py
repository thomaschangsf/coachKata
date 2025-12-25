"""
Comparison module for Phase 2: Teacher-student pose comparison.

This module provides:
- Learning phase: Derive ideal ranges from teacher examples
- Comparison: Compare student poses against teacher ranges
- Feedback: Generate prioritized, actionable feedback
- Visualization: Visualize differences with color-coded keypoints and arrows
"""

from .config_loader import (
    get_metrics_for_stroke,
    get_percentile_settings,
    get_scoring_method,
    get_single_pose_tolerance,
    get_visualization_settings,
    get_weights_for_stroke,
    load_comparison_config,
    validate_config,
)
from .feedback_generator import generate_feedback
from .metric_extractor import extract_metrics
from .pose_comparator import PoseComparator
from .scoring_v2 import (
    score_metric,
    score_metric_distance,
    score_metric_percentile,
    score_metric_tolerance,
)
from .teacher_learner import (
    calculate_metric_ranges,
    learn_teacher_ranges,
    load_teacher_ranges,
)
from .visualizer import ComparisonVisualizer

__all__ = [
    # Configuration
    'load_comparison_config',
    'validate_config',
    'get_metrics_for_stroke',
    'get_weights_for_stroke',
    'get_scoring_method',
    'get_visualization_settings',
    'get_single_pose_tolerance',
    'get_percentile_settings',
    # Metric extraction
    'extract_metrics',
    # Teacher learning
    'learn_teacher_ranges',
    'load_teacher_ranges',
    'calculate_metric_ranges',
    # Scoring v2
    'score_metric',
    'score_metric_tolerance',
    'score_metric_distance',
    'score_metric_percentile',
    # Comparison
    'PoseComparator',
    # Feedback
    'generate_feedback',
    # Visualization
    'ComparisonVisualizer',
]
