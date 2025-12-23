"""
Image processing module for pickleball pose analysis.

This module handles processing images through SAM 3D Body and extracting
structured pose data.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def process_single_image(
    estimator,
    image_path: str,
    position_name: str = "unknown",
    bbox_thr: float = 0.5,
    use_mask: bool = False,
) -> dict[str, Any]:
    """
    Process a single image and extract pose data.

    Args:
        estimator: Initialized SAM3DBodyEstimator
        image_path: Path to image file
        position_name: Name of position (e.g., "preparation", "contact", "finish")
        bbox_thr: Bounding box detection threshold
        use_mask: Whether to use mask-conditioned inference

    Returns:
        Dictionary containing:
        - position_name: str
        - pred_keypoints_3d: np.ndarray (70, 3)
        - pred_keypoints_2d: np.ndarray (70, 2)
        - pred_vertices: np.ndarray (18439, 3)
        - body_pose_params: np.ndarray
        - hand_pose_params: np.ndarray
        - shape_params: np.ndarray
        - scale_params: np.ndarray
        - pred_global_rots: np.ndarray (127, 3, 3)
        - bbox: np.ndarray
        - focal_length: float
        - metadata: dict (image_path, timestamp, etc.)
    """
    # Validate image file exists
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image not found: {image_path_obj}")

    # Load and validate image
    img = cv2.imread(str(image_path_obj))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Process image through estimator
    outputs = estimator.process_one_image(
        str(image_path_obj),
        bbox_thr=bbox_thr,
        use_mask=use_mask,
    )

    if not outputs or len(outputs) == 0:
        raise ValueError(f"No person detected in image: {image_path_obj}")

    # Use first detected person
    output = outputs[0]

    # Extract and structure pose data
    pose_data = {
        'position_name': position_name,
        'pred_keypoints_3d': output.get('pred_keypoints_3d', np.zeros((70, 3))),
        'pred_keypoints_2d': output.get('pred_keypoints_2d', np.zeros((70, 2))),
        'pred_vertices': output.get('pred_vertices', np.zeros((18439, 3))),
        'body_pose_params': output.get('body_pose_params', np.array([])),
        'hand_pose_params': output.get('hand_pose_params', np.array([])),
        'shape_params': output.get('shape_params', np.array([])),
        'scale_params': output.get('scale_params', np.array([])),
        'pred_global_rots': output.get('pred_global_rots', np.zeros((127, 3, 3))),
        'bbox': output.get('bbox', np.zeros(4)),
        'focal_length': float(output.get('focal_length', 800.0)),
        'metadata': {
            'image_path': str(image_path_obj),
            'timestamp': datetime.now().isoformat(),
            'bbox_thr': bbox_thr,
            'use_mask': use_mask,
            'image_shape': img.shape,
        },
    }

    return pose_data


def process_three_positions(
    estimator,
    preparation_path: str,
    contact_path: str,
    finish_path: str,
    bbox_thr: float = 0.5,
    use_mask: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Process all three pickleball positions.

    Args:
        estimator: Initialized SAM3DBodyEstimator
        preparation_path: Path to preparation position image
        contact_path: Path to contact position image
        finish_path: Path to finish position image
        bbox_thr: Bounding box detection threshold
        use_mask: Whether to use mask-conditioned inference

    Returns:
        Dictionary with keys: 'preparation', 'contact', 'finish'
        Each value is the output from process_single_image()
    """
    results = {}

    print("Processing preparation position...")
    results['preparation'] = process_single_image(
        estimator,
        preparation_path,
        position_name='preparation',
        bbox_thr=bbox_thr,
        use_mask=use_mask,
    )

    print("Processing contact position...")
    results['contact'] = process_single_image(
        estimator,
        contact_path,
        position_name='contact',
        bbox_thr=bbox_thr,
        use_mask=use_mask,
    )

    print("Processing finish position...")
    results['finish'] = process_single_image(
        estimator,
        finish_path,
        position_name='finish',
        bbox_thr=bbox_thr,
        use_mask=use_mask,
    )

    return results
