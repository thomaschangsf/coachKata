"""
Pickleball Pose Analyzer - Phase 1: Single Image Analysis

This module provides tools for analyzing pickleball poses using SAM 3D Body.
"""

import os
import sys


def _setup_sam3d_paths():
    """Automatically add models/sam-3d-body to Python path"""
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '../../..'))
    sam3d_path = os.path.join(project_root, 'models', 'sam-3d-body')

    if os.path.exists(sam3d_path) and sam3d_path not in sys.path:
        sys.path.insert(0, sam3d_path)
        return True
    return False


# Setup paths before importing sam_3d_body
_setup_sam3d_paths()

# Import SAM 3D Body components (after path setup)
try:
    from sam_3d_body import (  # type: ignore[reportMissingImports]
        SAM3DBodyEstimator,
        load_sam_3d_body_hf,
    )
except ImportError as e:
    # Graceful handling if sam_3d_body is not available
    SAM3DBodyEstimator = None
    load_sam_3d_body_hf = None
    import warnings
    warnings.warn(f"Could not import sam_3d_body: {e}. Some functions may not work.", stacklevel=2)

# Import our modules
from .data_structures import (
    KEYPOINT_NAMES_TO_INDICES,
    PICKLEBALL_KEYPOINTS,
    PoseData,
    PositionScore,
)
from .image_processor import process_single_image, process_three_positions
from .keypoint_extractor import calculate_body_center, extract_keypoints
from .model_loader import load_sam3d_model
from .scoring import (
    calculate_3d_angle,
    score_contact_position,
    score_finish_position,
    score_preparation_position,
)

# Import Phase 2 comparison module
try:
    from .comparison import (
        ComparisonVisualizer,
        PoseComparator,
        generate_feedback,
        learn_teacher_ranges,
        load_teacher_ranges,
    )
    HAS_COMPARISON = True
except ImportError:
    # Graceful handling if comparison module not available
    HAS_COMPARISON = False
    import warnings
    warnings.warn(
        "Comparison module (Phase 2) not available. Some functions may not work.",
        stacklevel=2
    )

__all__ = [
    # Data structures
    'PoseData',
    'PositionScore',
    'PICKLEBALL_KEYPOINTS',
    'KEYPOINT_NAMES_TO_INDICES',
    # Model loading
    'load_sam3d_model',
    # Image processing
    'process_single_image',
    'process_three_positions',
    # Keypoint extraction
    'extract_keypoints',
    'calculate_body_center',
    # Scoring (Phase 1)
    'calculate_3d_angle',
    'score_preparation_position',
    'score_contact_position',
    'score_finish_position',
]

# Add Phase 2 exports if available
if HAS_COMPARISON:
    __all__.extend([
        # Comparison (Phase 2)
        'PoseComparator',
        'learn_teacher_ranges',
        'load_teacher_ranges',
        'generate_feedback',
        'ComparisonVisualizer',
    ])
