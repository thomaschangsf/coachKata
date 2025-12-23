# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import torch

from .models.meta_arch import SAM3DBody
from .utils.checkpoint import load_state_dict
from .utils.config import get_config


def load_sam_3d_body(checkpoint_path: str = "", device: str = "cpu", mhr_path: str = ""):
    print("Loading SAM 3D Body model...")

    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()

    # Initialze the model
    model = SAM3DBody(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)

    # Safely move model to device
    try:
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, using CPU")
            device = "cpu"
        model = model.to(device)
    except (AssertionError, RuntimeError) as e:
        print(f"Warning: Could not use device '{device}': {e}, falling back to CPU")
        device = "cpu"
        model = model.to(device)

    model.eval()
    return model, model_cfg


def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "model.ckpt"), os.path.join(local_dir, "assets", "mhr_model.pt")


def load_sam_3d_body_hf(repo_id, device: str = "cpu", **kwargs):
    ckpt_path, mhr_path = _hf_download(repo_id)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path, device=device)
