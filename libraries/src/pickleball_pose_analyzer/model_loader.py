"""
Model loading module for SAM 3D Body.

This module handles loading and initializing the SAM 3D Body model
from HuggingFace or local checkpoints.
"""

import os
import sys
from typing import Any

import torch

# Ensure sam_3d_body is in path (should be done by __init__.py, but just in case)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sam3d_path = os.path.join(project_root, 'models', 'sam-3d-body')
if sam3d_path not in sys.path:
    sys.path.insert(0, sam3d_path)

try:
    from sam_3d_body import (  # type: ignore[reportMissingImports]
        SAM3DBodyEstimator,
        load_sam_3d_body_hf,
    )
    from tools.build_detector import HumanDetector  # type: ignore[reportMissingImports]
    from tools.build_fov_estimator import (  # type: ignore[reportMissingImports]
        FOVEstimator,
    )
except ImportError as e:
    raise ImportError(
        f"Could not import sam_3d_body. Make sure models/sam-3d-body is available. Error: {e}"
    ) from e


def load_sam3d_model(
    checkpoint_path: str | None = None,
    hf_repo_id: str = "facebook/sam-3d-body-dinov3",
    device: str = "auto",
    use_detector: bool = True,
    use_segmentor: bool = False,
    use_fov_estimator: bool = True,
    detector_name: str = "vitdet",
    fov_name: str = "moge2",
) -> tuple[SAM3DBodyEstimator, dict[str, Any]]:
    """
    Load and initialize SAM 3D Body model.

    Args:
        checkpoint_path: Local path to model checkpoint (optional)
        hf_repo_id: HuggingFace repository ID (used if checkpoint_path is None)
        device: "auto", "cuda", or "cpu"
        use_detector: Whether to use human detector
        use_segmentor: Whether to use segmentation (SAM2) - not implemented yet
        use_fov_estimator: Whether to use FOV estimator (MoGe2)
        detector_name: Name of detector to use (default: "vitdet")
        fov_name: Name of FOV estimator to use (default: "moge2")

    Returns:
        estimator: Initialized SAM3DBodyEstimator
        config: Model configuration dictionary
    """
    # Auto-detect device if set to "auto" or None
    if device == "auto" or device is None:
        try:
            if torch.cuda.is_available():
                device = "cuda"
                print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
        except (AssertionError, RuntimeError) as e:
            device = "cpu"
            print(f"CUDA not available ({type(e).__name__}), using CPU")
    elif device == "cuda":
        try:
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                device = "cpu"
        except (AssertionError, RuntimeError):
            print("Warning: PyTorch not compiled with CUDA support, falling back to CPU")
            device = "cpu"

    print(f"Loading SAM 3D Body model from {hf_repo_id}...")
    print(f"Using device: {device}")

    # Load core model from HuggingFace
    if checkpoint_path:
        # TODO: Implement local checkpoint loading if needed
        raise NotImplementedError("Local checkpoint loading not yet implemented")
    else:
        model, model_cfg = load_sam_3d_body_hf(hf_repo_id, device=device)

    # Initialize optional components
    human_detector = None
    if use_detector:
        print(f"Loading human detector: {detector_name}...")
        human_detector = HumanDetector(name=detector_name, device=device)

    human_segmentor = None
    if use_segmentor:
        # TODO: Implement segmentor loading if needed
        print("Warning: Segmentor not yet implemented")

    fov_estimator = None
    if use_fov_estimator:
        print(f"Loading FOV estimator: {fov_name}...")
        fov_estimator = FOVEstimator(name=fov_name, device=device)

    # Create estimator
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    # Create config dictionary
    config = {
        'hf_repo_id': hf_repo_id,
        'device': device,
        'use_detector': use_detector,
        'use_segmentor': use_segmentor,
        'use_fov_estimator': use_fov_estimator,
        'detector_name': detector_name,
        'fov_name': fov_name,
    }

    print("Model loaded successfully!")
    return estimator, config
