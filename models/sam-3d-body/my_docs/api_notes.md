# API Notes: `use_detector` and `use_fov_estimator` Parameters

This document explains the importance and trade-offs of the `use_detector` and `use_fov_estimator` parameters in `load_sam3d_model()`.

## Overview

Both `use_detector` and `use_fov_estimator` are optional components that can significantly impact inference speed and accuracy. Understanding when to enable or disable them is crucial for optimizing your use case.

---

## 1. `use_detector` (Human Detector)

### What It Does

The human detector (default: ViTDet) automatically detects and localizes humans in images by:
- Running object detection to find bounding boxes around people
- Handling multiple people in a single image
- Filtering detections by confidence threshold

### When `use_detector=True` (Default)

**Behavior:**
- The detector runs on each image before pose estimation
- Automatically finds bounding boxes for all detected humans
- Returns pose estimates for each detected person

**Performance Impact:**
- **GPU**: Adds ~100-200ms per image
- **CPU**: Adds ~1000-2000ms (1-2 seconds) per image
- **Total speedup cost**: ~27-41% slower inference

**Use Cases:**
- Images with multiple people (need to detect and isolate each person)
- Uncropped images with background/other objects
- Unknown image content (don't know where the person is)
- Batch processing diverse images

### When `use_detector=False`

**Behavior:**
- No detection step runs
- If `bboxes` parameter is provided: Uses those bounding boxes directly
- If `bboxes` is `None`: Assumes the entire image is the person region
  ```python
  boxes = np.array([0, 0, width, height]).reshape(1, 4)  # Full image
  ```

**Performance Impact:**
- **Speedup**: ~27-41% faster inference
- **Memory**: Saves detector model memory (~500MB-1GB)

**Use Cases:**
- Pre-cropped images (person already centered/framed)
- Single person per image
- You can provide bounding boxes manually via `bboxes` parameter
- Speed is critical and accuracy can be slightly lower

**Trade-offs:**
- ✅ Faster inference
- ✅ Lower memory usage
- ❌ Requires manual bounding boxes or pre-cropped images
- ❌ Less accurate if image contains multiple people or background
- ❌ May fail if person is not well-centered in the image

---

## 2. `use_fov_estimator` (FOV Estimator)

### What It Does

The FOV estimator (default: MoGe2) estimates camera focal length from the image by analyzing:
- Perspective cues (parallel lines, vanishing points)
- Geometric patterns and object scale relationships
- Scene structure

**Why Focal Length Matters:**

Focal length is critical for accurate 3D reconstruction because:

1. **Depth Calculation**: The model computes depth as `depth = 2 * focal_length / bounding_box_scale`
   - Wrong focal length → Wrong depth → Wrong 3D coordinates
   - All 3D measurements (distances, angles, proportions) become inaccurate

2. **3D to 2D Projection**: The model projects 3D keypoints back to 2D for validation
   - Uses focal length in the pinhole camera model: `x_2d = (f * x_3d) / z_3d`
   - Incorrect focal length causes projection errors

3. **Decoder Conditioning**: The model normalizes inputs using focal length
   - Affects how the decoder interprets scale and perspective

**Visual Analogy:**
Think of focal length like a zoom level:
- **Short focal length (wide FOV)**: Person appears smaller, farther away
- **Long focal length (narrow FOV)**: Person appears larger, closer

If the model assumes the wrong zoom level, it misinterprets:
- How far the person is from the camera
- How large the person is in real 3D space

### When `use_fov_estimator=True` (Default)

**Behavior:**
- MoGe2 model analyzes each image to estimate actual camera focal length
- Provides camera intrinsics matrix with estimated focal length
- More accurate 3D reconstruction, especially for diverse camera setups

**Performance Impact:**
- **GPU**: Adds ~30-80ms per image
- **CPU**: Adds ~200-500ms per image
- **Total speedup cost**: ~7-15% slower inference

**Use Cases:**
- Images from unknown cameras (phone, DSLR, webcam)
- Different camera-to-subject distances
- Need accurate metric measurements (actual distances in meters)
- Need accurate 3D scale and proportions
- Diverse image sources with varying camera parameters

### When `use_fov_estimator=False`

**Behavior:**
- Uses default focal length (typically ~800 pixels)
- Assumes a standard camera FOV (~55-60 degrees)
- Assumes standard image size and camera distance

**Performance Impact:**
- **Speedup**: ~7-15% faster inference
- **Memory**: Saves FOV estimator model memory (~500MB-1GB)

**Use Cases:**
- All images from the same camera with known/similar settings
- Only need relative pose comparisons (not absolute measurements)
- Speed is critical and approximate 3D scale is acceptable
- Images are from a controlled setup (same camera, same distance)

**Trade-offs:**
- ✅ Faster inference
- ✅ Lower memory usage
- ❌ Less accurate 3D depth/scale if camera differs from default assumptions
- ❌ Metric measurements (distances, sizes) may be incorrect
- ❌ 3D proportions may be distorted

**When Default FOV Works Well:**
- Images from similar cameras (e.g., all iPhone photos)
- Consistent camera-to-subject distance
- Only comparing relative poses (not measuring absolute distances)
- Controlled studio/indoor environment

**When Default FOV Fails:**
- Mix of different cameras (phone vs DSLR vs webcam)
- Varying distances (close-up vs far away)
- Different lenses (wide-angle vs telephoto)
- Need accurate metric measurements

---

## Performance Summary

### Speed Improvements (Disabling Components)

| Configuration | GPU Speedup | CPU Speedup |
|--------------|------------|-------------|
| Disable detector only | ~27-36% faster | ~31-41% faster |
| Disable FOV estimator only | ~9-15% faster | ~7-10% faster |
| Disable both | ~36-45% faster | ~38-50% faster |

### Accuracy Trade-offs

| Configuration | Accuracy Impact |
|--------------|----------------|
| Disable detector | ⚠️ May fail with multiple people or uncropped images |
| Disable FOV estimator | ⚠️ 3D scale/depth may be inaccurate if camera differs from default |

---

## Recommendations

### For Maximum Accuracy
```python
estimator, config = load_sam3d_model(
    use_detector=True,      # Handle multiple people, unknown image content
    use_fov_estimator=True, # Accurate 3D reconstruction for diverse cameras
)
```

### For Maximum Speed
```python
estimator, config = load_sam3d_model(
    use_detector=False,      # If you have pre-cropped images or manual bboxes
    use_fov_estimator=False, # If all images from same camera setup
)
```

### For Balanced Performance
```python
# Fast detection, accurate 3D
estimator, config = load_sam3d_model(
    use_detector=False,      # Pre-cropped images
    use_fov_estimator=True,  # Accurate 3D for diverse cameras
)

# OR

# Automatic detection, approximate 3D
estimator, config = load_sam3d_model(
    use_detector=True,       # Handle unknown image content
    use_fov_estimator=False, # Same camera setup, approximate 3D OK
)
```

### For Pickleball Pose Analysis

**Recommended Configuration:**
```python
estimator, config = load_sam3d_model(
    use_detector=True,      # Handle multiple people in court images
    use_fov_estimator=True, # Accurate 3D measurements for pose scoring
)
```

**Rationale:**
- Pickleball images may contain multiple people (players, spectators)
- Accurate 3D measurements are needed for pose scoring (angles, distances, proportions)
- The speed cost (~40-50% slower) is acceptable for accuracy gains

**Alternative (if speed is critical):**
```python
estimator, config = load_sam3d_model(
    use_detector=False,      # Pre-crop images to single person
    use_fov_estimator=True,  # Still need accurate 3D for scoring
)
```

---

## Technical Details

### Detector Implementation

- **Model**: ViTDet (Vision Transformer-based detector)
- **Input**: Full image (resized to 1024x1024)
- **Output**: Bounding boxes `[x1, y1, x2, y2]` for detected humans
- **Location**: `models/sam-3d-body/tools/build_detector.py`

### FOV Estimator Implementation

- **Model**: MoGe2 (Monocular Geometry Estimation)
- **Input**: RGB image
- **Output**: Camera intrinsics matrix with estimated focal length
- **Location**: `models/sam-3d-body/tools/build_fov_estimator.py`

### How Focal Length Affects 3D Reconstruction

The critical relationship in `camera_head.py`:

```python
# Depth calculation (line 86)
tz = 2 * focal_length / bs  # depth ~= f / scale

# 3D to 2D projection (line 102)
j2d = perspective_projection(j3d_cam, cam_int)
```

If focal length is wrong:
- Depth (`tz`) is wrong → All 3D coordinates are scaled incorrectly
- Projection errors → 2D keypoint validation fails
- Metric measurements become inaccurate

---

## See Also

- `architecture.md`: Detailed SAM 3D Body architecture
- `impl-phase1-single-image-analysis.md`: Implementation details for Phase 1
- `models/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py`: Main estimator implementation
