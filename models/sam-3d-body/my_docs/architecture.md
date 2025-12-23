# SAM 3D Body Architecture for Pickleball Pose Comparison

## Overview

This document describes the SAM 3D Body architecture and its potential for comparing and quantifying differences between pickleball poses, as an alternative to the current MediaPipe-based approach in `notebooks/mediapipe/pose_comparison_example.ipynb`.

## Core Components and Classes

### 1. Main Entry Point: `SAM3DBodyEstimator`

**Location:** `sam_3d_body/sam_3d_body_estimator.py`

**Purpose:** The `SAM3DBodyEstimator` is like a **complete NLP pipeline** (think HuggingFace's `pipeline()` function) that takes raw input and returns structured, usable output. It orchestrates the entire pose estimation process from image preprocessing to final 3D pose prediction.

**NLP Analogy:** 
- In NLP, you might use `pipeline("text-classification")` which handles tokenization, model inference, and post-processing automatically
- Similarly, `SAM3DBodyEstimator` handles image loading, human detection, model inference, and result formatting

**Key Method: `process_one_image()`**

**Example Input:**
```python
# Input: Raw image (numpy array or file path)
img = cv2.imread("pickleball_serve.jpg")  # Shape: (480, 640, 3) - RGB image
# OR
img = "path/to/image.jpg"  # String path

# Optional inputs:
bboxes = None  # Pre-detected bounding boxes (if you already know where person is)
masks = None   # Segmentation masks (if you want to focus on specific regions)
cam_int = None # Camera intrinsics (focal length, etc.)
```

**Example Output:**
```python
outputs = estimator.process_one_image(img)
# Returns: List of dictionaries (one per detected person)

output = outputs[0]  # First detected person
{
    'bbox': array([100, 50, 300, 500]),  # Bounding box [x1, y1, x2, y2]
    'focal_length': array([800.5]),      # Camera focal length
    'pred_keypoints_3d': array([          # 70 keypoints in 3D space
        [0.12, -0.45, 1.2],   # nose (x, y, z in meters)
        [0.15, -0.43, 1.21],  # left_eye
        [0.09, -0.43, 1.21],  # right_eye
        # ... 67 more keypoints
    ]),  # Shape: (70, 3)
    'pred_keypoints_2d': array([         # 2D projections (for visualization)
        [320, 150],  # nose in pixel coordinates
        [330, 145],  # left_eye
        # ... 68 more
    ]),  # Shape: (70, 2)
    'pred_vertices': array([             # Full 3D mesh (~18,000 vertices)
        [0.1, -0.4, 1.15],
        [0.11, -0.41, 1.16],
        # ... ~18,000 more vertices
    ]),  # Shape: (18439, 3)
    'pred_cam_t': array([0.0, 0.1, 2.5]),  # Camera translation
    'body_pose_params': array([...]),      # Body pose (260-dim vector)
    'hand_pose_params': array([...]),      # Hand pose (108-dim: 54 per hand)
    'shape_params': array([...]),          # Body shape (45-dim)
    'scale_params': array([...]),          # Scale parameters (28-dim)
    'pred_joint_coords': array([...]),     # Joint coordinates (127, 3)
    'pred_global_rots': array([...])       # Rotation matrices (127, 3, 3)
}
```

**What it does internally:**
1. Loads/preprocesses image (like tokenization in NLP)
2. Detects humans in image (optional, if detector provided)
3. Crops image to person bounding box
4. Runs SAM3DBody model inference
5. Formats and returns structured results

---

### 2. Model Architecture: `SAM3DBody`

**Location:** `sam_3d_body/models/meta_arch/sam3d_body.py`

**Purpose:** `SAM3DBody` is the **core neural network model** - think of it as the equivalent of BERT or GPT in NLP. It's an encoder-decoder architecture that transforms image features into a "pose representation" (like how BERT transforms tokens into embeddings).

**NLP Analogy:**
- **Encoder (Backbone)**: Like BERT's encoder - takes image patches (like tokens) and creates feature embeddings
- **Decoder**: Like GPT's decoder - takes embeddings and generates pose tokens (like generating text tokens)
- **Heads**: Like classification heads in NLP - convert embeddings to specific outputs (pose parameters, camera params)

**What SAM3DBody Actually Predicts:**

The model predicts **pose tokens** - high-dimensional vectors (typically 1024-dim) that encode the entire body pose. Think of these like **sentence embeddings** in NLP that capture the meaning of a sentence in a single vector.

**Example Input:**
```python
# Internal representation - you typically don't call this directly
batch = {
    'img': tensor([...]),        # Preprocessed image crops (B, N, 3, 256, 256)
    'bbox_center': tensor([...]), # Bounding box centers
    'bbox_scale': tensor([...]),  # Bounding box scales
    'cam_int': tensor([...]),     # Camera intrinsics
    # ... other metadata
}

# The model processes this through:
# 1. Backbone (ViT/DINOv3) → image_embeddings (B, C, H, W)
# 2. Decoder → pose_tokens (B, 1, 1024) - the "pose embedding"
```

**Example Output:**
```python
# Internal output from forward_pose_branch()
output = {
    'mhr': {
        'pred_pose_raw': tensor([...]),      # Raw pose parameters (404-dim)
        'global_rot': tensor([...]),          # Global rotation (3-dim Euler angles)
        'body_pose': tensor([...]),           # Body joint rotations (130-dim)
        'shape': tensor([...]),               # Body shape (45-dim)
        'scale': tensor([...]),               # Scale parameters (28-dim)
        'hand': tensor([...]),                # Hand pose (108-dim)
        'pred_keypoints_3d': tensor([...]),   # 3D keypoints (70, 3)
        'pred_vertices': tensor([...]),       # 3D mesh vertices (18439, 3)
        'joint_global_rots': tensor([...])    # Rotation matrices (127, 3, 3)
    },
    'image_embeddings': tensor([...]),        # Image features from backbone
    'condition_info': tensor([...])           # Camera/bbox conditioning info
}
```

**How This Relates to 3D Visualization and Rotation:**

The key outputs for 3D visualization are:
1. **`pred_vertices`**: The 3D mesh vertices - these are the actual 3D points that form the body surface
2. **`pred_global_rots`**: Rotation matrices for each joint - these tell you how each body part is oriented
3. **`pred_keypoints_3d`**: Skeleton keypoints - these are the "bones" of the skeleton

**To rotate the body in 3D:**
```python
# You can apply rotation matrices to the vertices
vertices_3d = output['mhr']['pred_vertices']  # (18439, 3)
rotation_matrix = np.array([...])  # (3, 3) rotation matrix

# Rotate all vertices
rotated_vertices = vertices_3d @ rotation_matrix.T

# Now you can render rotated_vertices from any angle!
```

The rotation matrices (`pred_global_rots`) are like **part-of-speech tags** in NLP - they describe the "state" of each joint (how it's rotated), which allows you to reconstruct the full 3D pose and view it from any angle.

---

### 3. MHR Head: `MHRHead`

**Location:** `sam_3d_body/models/heads/mhr_head.py`

**Purpose:** The MHR Head is like a **task-specific output layer** in NLP. Just as a BERT model might have a classification head that converts embeddings to class probabilities, the MHR Head converts pose tokens to MHR (Momentum Human Rig) parameters.

**Relationship Between Pose Tokens and Pose Parameters:**

**Key Point:** Pose tokens and pose parameters are **both derived from the same decoder embeddings**, but they represent different stages of the transformation pipeline:

1. **Pose Tokens** = Direct output from decoder (intermediate representation)
2. **Pose Parameters** = Transformed output from MHRHead (task-specific representation)

**NLP Analogy:**
- **Pose Tokens** are like the `[CLS]` token embedding from BERT - a learned representation that captures the pose information
- **Pose Parameters** are like the output from a classification head - a transformed version optimized for the specific task (generating 3D mesh)

**The Flow:**

```
Decoder Embeddings (image features)
    ↓
Decoder Transformer Layers
    ↓
Pose Token (1024-dim) ← Intermediate representation
    ↓
MHRHead.proj (Linear/MLP layer)
    ↓
Pose Parameters (404-dim) ← Task-specific representation
    ↓
Parameter Decomposition
    ↓
MHR Model → 3D Mesh
```

**Example Input (Pose Token):**
```python
# Input: Pose token from decoder
pose_token = tensor([
    [0.12, -0.45, 0.89, ..., 0.23],  # 1024 values
    [0.15, -0.43, 0.91, ..., 0.25],  # For batch_size=2
])  # Shape: (batch_size, 1024)
# This is a learned embedding that encodes the entire body pose
# It's the FIRST token from the decoder output (tokens[:, 0])
```

**Example Intermediate Output (Pose Parameters):**
```python
# After MHRHead.proj(pose_token)
pose_params = tensor([
    [0.1, 0.2, 0.3, ..., 0.5],  # 404 values
    [0.12, 0.22, 0.32, ..., 0.52],
])  # Shape: (batch_size, 404)
# This is the "raw" pose parameters before decomposition
```

**Example Final Output (Decomposed Parameters):**
```python
# Output from MHRHead.forward() - after decomposition
output = {
    # Raw parameters (what you get directly from pose_token)
    'pred_pose_raw': tensor([...]),      # (B, 404) - Raw continuous pose params
    
    # Decomposed parameters (split from pred_pose_raw)
    'global_rot': tensor([...]),         # (B, 3) - Global body rotation (Euler angles)
    'body_pose': tensor([...]),          # (B, 130) - Body joint rotations (Euler)
    'shape': tensor([...]),               # (B, 45) - Body shape parameters
    'scale': tensor([...]),               # (B, 28) - Scale parameters
    'hand': tensor([...]),               # (B, 108) - Hand pose (54 per hand)
    'face': tensor([...]),                # (B, 72) - Face expression (usually zeroed)
    
    # Generated 3D outputs (from MHR forward kinematics)
    'pred_keypoints_3d': tensor([...]),   # (B, 70, 3) - 3D keypoint coordinates
    'pred_vertices': tensor([...]),      # (B, 18439, 3) - Full 3D mesh vertices
    'pred_joint_coords': tensor([...]),  # (B, 127, 3) - Joint positions
    'joint_global_rots': tensor([...])   # (B, 127, 3, 3) - Rotation matrices
}
```

**The Conversion Process (Step-by-Step):**

1. **Pose Token Extraction**: 
   ```python
   # In decoder: tokens[:, 0] extracts the first token
   pose_token = decoder_output[:, 0]  # (B, 1024)
   ```

2. **Linear Projection**: 
   ```python
   # In MHRHead.forward():
   pred = self.proj(pose_token)  # (B, 1024) → (B, 404)
   # This is a simple MLP: FFN(input_dim=1024, output_dim=404)
   ```

3. **Parameter Decomposition**: The 404-dim vector is split into interpretable components:
   ```python
   # Decomposition (from pred_pose_raw):
   global_rot_6d = pred[:, :6]           # First 6 dims → 3D rotation matrix
   body_pose_cont = pred[:, 6:266]        # Next 260 dims → 130 joint rotations
   shape = pred[:, 266:311]               # Next 45 dims → body shape
   scale = pred[:, 311:339]               # Next 28 dims → scale parameters
   hand = pred[:, 339:447]                # Next 108 dims → hand pose
   face = pred[:, 447:519]                # Next 72 dims → face expression
   ```

4. **Forward Kinematics**: Parameters → 3D Mesh
   ```python
   # Uses MHR model (like a "grammar" for human bodies)
   verts, keypoints_3d = mhr_forward(
       global_rot, body_pose, shape, scale, hand, face
   )
   ```

**Are They Both Derived from the Same Decoder Embeddings?**

**Yes, but at different stages:**

- **Pose Tokens**: Direct output from the decoder transformer (the first token: `tokens[:, 0]`)
  - Shape: `(batch_size, 1024)`
  - This is the "raw" learned representation

- **Pose Parameters**: Derived from pose tokens via MHRHead's projection layer
  - Shape: `(batch_size, 404)`
  - This is a transformed version optimized for MHR model input

**Think of it like this (NLP analogy):**
- **Decoder Embeddings** = All token embeddings from BERT encoder
- **Pose Token** = `[CLS]` token embedding (summary of the pose)
- **Pose Parameters** = Output from a task-specific head (like classification logits, but for pose)

**Why Both Exist:**

- **Pose Tokens** are useful for:
  - Iterative refinement (can be fed back into decoder)
  - Multi-task learning (can be used for other tasks)
  - Intermediate debugging/analysis

- **Pose Parameters** are useful for:
  - Direct interpretation (each dimension has meaning)
  - Editing poses (modify specific parameters)
  - Compatibility with MHR model (required format)

**Concrete Example: Same Pose in Both Formats**

```python
# After decoder forward pass:
decoder_output = decoder(image_embeddings, ...)  # Returns tokens
pose_token = decoder_output[0][:, 0]  # Extract first token
# pose_token shape: (batch_size=1, 1024)
# Example values: [0.12, -0.45, 0.89, 0.23, ..., 0.67]  # 1024 floats

# After MHRHead.proj:
pose_params = mhr_head.proj(pose_token)
# pose_params shape: (batch_size=1, 404)
# Example values: [0.1, 0.2, 0.3, ..., 0.5]  # 404 floats

# After decomposition:
decomposed = {
    'global_rot': pose_params[:, :6],      # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    'body_pose': pose_params[:, 6:266],    # [0.7, 0.8, ..., 0.9]  # 260 values
    'shape': pose_params[:, 266:311],      # [0.11, 0.12, ..., 0.15]  # 45 values
    'scale': pose_params[:, 311:339],      # [0.16, 0.17, ..., 0.18]  # 28 values
    'hand': pose_params[:, 339:447],       # [0.19, 0.20, ..., 0.21]  # 108 values
    'face': pose_params[:, 447:519],       # [0.0, 0.0, ..., 0.0]  # 72 values (zeroed)
}

# Both represent the SAME pose, just in different formats:
# - pose_token: Learned embedding (not directly interpretable)
# - pose_params: Structured parameters (each dimension has meaning)
```

**Why MHR Parameters Matter:**

MHR parameters are **interpretable** and **editable**:
- You can directly modify `body_pose[0]` to change the left shoulder rotation
- You can adjust `shape` to change body proportions
- Similar to how you can modify word embeddings or attention weights in NLP models

---

### 4. Component Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAM3DBodyEstimator                          │
│                  (High-level API / Pipeline)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Input: Image (numpy array or path)
                             │
                             ▼
                    ┌─────────────────┐
                    │  Image Loading  │
                    │  & Preprocessing│
                    └────────┬────────┘
                             │
                             │ Preprocessed image
                             │ + Optional bboxes/masks
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SAM3DBody Model                            │
│                   (Core Neural Network)                         │
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │   Backbone   │────────▶│   Decoder    │                     │
│  │  (Encoder)   │         │ (Transformer)│                     │
│  │              │         │              │                     │
│  │ ViT/DINOv3   │         │ Generates    │                     │
│  │              │         │ Pose Tokens │                     │
│  │ Image →      │         │              │                     │
│  │ Embeddings   │         │ Embeddings → │                     │
│  │ (B,C,H,W)    │         │ Tokens       │                     │
│  │              │         │ (B,1,1024)   │                     │
│  └──────────────┘         └──────┬───────┘                     │
│                                   │                             │
│                                   │ Pose Token                  │
│                                   │ (1024-dim vector)           │
│                                   │                             │
└───────────────────────────────────┼─────────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────┐
                    │      MHRHead           │
                    │  (Output Layer/Head)   │
                    │                        │
                    │  Pose Token →         │
                    │  MHR Parameters       │
                    │                        │
                    │  (1024) → (404)       │
                    │                        │
                    │  Then:                 │
                    │  Parameters →          │
                    │  3D Mesh via MHR      │
                    │                        │
                    │  Output:               │
                    │  - Keypoints 3D        │
                    │  - Vertices 3D         │
                    │  - Rotation matrices   │
                    └───────────┬────────────┘
                                │
                                │ Structured 3D Pose Data
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Formatted Output     │
                    │  Dictionary with:     │
                    │  - pred_keypoints_3d  │
                    │  - pred_vertices      │
                    │  - pred_global_rots   │
                    │  - body_pose_params   │
                    │  - etc.               │
                    └───────────────────────┘
                                │
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Visualization       │
                    │   (Optional)          │
                    │                       │
                    │   Can render:         │
                    │   - 3D mesh           │
                    │   - Skeleton overlay  │
                    │   - From any angle    │
                    └───────────────────────┘
```

**Data Flow Summary:**

1. **Image** → `SAM3DBodyEstimator.process_one_image()`
2. **Preprocessed Image** → `SAM3DBody.backbone` → **Image Embeddings** (B, C, H, W)
3. **Image Embeddings** → `SAM3DBody.decoder` → **Pose Tokens** (B, 1024-dim)
   - Note: Pose token is extracted as `tokens[:, 0]` - the first token from decoder
4. **Pose Tokens** → `MHRHead.proj` (MLP) → **Pose Parameters** (B, 404-dim)
   - This is a linear transformation: `pose_params = MLP(pose_token)`
5. **Pose Parameters** → Decomposition → **Structured Parameters** (global_rot, body_pose, shape, scale, hand)
6. **Structured Parameters** → `MHRHead.mhr_forward()` → **3D Mesh & Keypoints**
7. **3D Data** → Formatted dictionary → **Visualization/Use**

**Key Relationship:**
- **Pose Tokens** and **Pose Parameters** are both derived from the same decoder output
- **Pose Token** = Direct decoder output (intermediate representation)
- **Pose Parameters** = Transformed via MHRHead.proj (task-specific representation)
- Both represent the same pose, but in different formats optimized for different purposes

**NLP Analogy for the Full Flow:**
- **Image** = Raw text
- **Backbone** = Tokenizer + BERT encoder
- **Decoder** = GPT decoder (generates pose "tokens")
- **Pose Token** = `[CLS]` token embedding (summary representation)
- **MHRHead.proj** = Task-specific head (like classification layer)
- **Pose Parameters** = Structured output (like named entities or parse trees)
- **3D Mesh** = Final interpretable result (like generated text or extracted information)

---

### 5. Keypoint Metadata: `mhr70.py`

**Location:** `sam_3d_body/metadata/mhr70.py`

Defines 70 keypoints including:
- Body: 17 keypoints (nose, eyes, shoulders, elbows, wrists, hips, knees, ankles, feet)
- Hands: 40 keypoints (20 per hand, including all finger joints)
- Extra: 7 keypoints (neck, olecranon, cubital fossa, acromion)

### 6. Visualization Components

**Location:** `sam_3d_body/visualization/`

- `skeleton_visualizer.py`: Draws 2D skeleton overlays
- `renderer.py`: 3D mesh rendering
- `utils.py`: Visualization utilities

## Key Libraries and Dependencies

### Core Dependencies

- **PyTorch**: Deep learning framework
- **OpenCV (cv2)**: Image processing
- **NumPy**: Numerical operations
- **Roma**: Rotation matrix utilities
- **PyTorch Lightning**: Training framework (for model loading)

### Model-Specific Dependencies

- **MHR (Momentum Human Rig)**: Parametric human mesh model
- **Detectron2**: Object detection (for human bounding boxes)
- **SAM3** (optional): Segmentation for mask prompting
- **MoGe** (optional): Field-of-view estimation

### Comparison with MediaPipe

| Feature | MediaPipe | SAM 3D Body |
|---------|-----------|-------------|
| Keypoints | 33 landmarks | 70 keypoints |
| Output | 2D coordinates | 3D coordinates + mesh |
| Hand detail | Basic | Full finger joints (20/hand) |
| Accuracy | Good for 2D | State-of-the-art 3D |
| Speed | Fast | Slower (GPU recommended) |
| Dependencies | Lightweight | Heavy (PyTorch, MHR) |

## Potential Capabilities for Pickleball Pose Comparison

### 1. Enhanced 3D Joint Angle Analysis

**Current MediaPipe Approach:**
- Calculates angles from 2D projections
- Limited to visible joints
- Subject to perspective distortion

**SAM 3D Body Advantages:**
- True 3D joint angles from 3D coordinates
- More accurate angle calculations
- Handles occlusions better
- Can compare poses from different camera angles

**Implementation:**
```python
# Extract 3D keypoints
teacher_kp3d = teacher_output['pred_keypoints_3d']  # (70, 3)
student_kp3d = student_output['pred_keypoints_3d']  # (70, 3)

# Calculate 3D angles (e.g., elbow angle)
def angle_3d(a, b, c):
    """Calculate angle at point b between vectors ba and bc"""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

elbow_angle_teacher = angle_3d(
    teacher_kp3d[6],   # right_shoulder
    teacher_kp3d[8],   # right_elbow
    teacher_kp3d[41]   # right_wrist
)
```

### 2. 3D Distance and Position Metrics

**New Capabilities:**
- Compare 3D joint positions directly
- Measure distances between corresponding joints
- Analyze body segment lengths
- Compare spatial relationships (e.g., racket-to-body distance)

**Example Metrics:**
- **Joint-to-joint distances**: Compare distances between key joints
- **Body segment lengths**: Compare arm/leg lengths (normalized by body size)
- **Spatial alignment**: Compare overall body orientation in 3D space

### 3. Hand and Racket Analysis

**Advantages:**
- 20 keypoints per hand (vs MediaPipe's basic hand detection)
- Full finger joint tracking
- Can analyze grip, wrist angle, finger positions
- Better for analyzing racket technique

**Key Hand Keypoints:**
- Wrist (1 per hand)
- Thumb: 4 joints
- Each finger: 4 joints (tip, 3 phalanges)

### 4. Full Body Mesh Comparison

**Capabilities:**
- Compare full 3D body shape
- Analyze posture and body alignment
- Visualize differences as 3D mesh overlays
- Calculate surface-to-surface distances

**Use Cases:**
- Overall posture comparison
- Body alignment analysis
- Weight distribution visualization

### 5. Temporal and Multi-View Analysis

**Potential Extensions:**
- Process video sequences
- Compare poses across different camera angles
- Analyze pose transitions (e.g., serve motion)
- Generate 3D motion paths

### 6. Advanced Comparison Metrics

**Beyond Simple Angle Differences:**

1. **Rotation Matrix Comparison**: Compare joint rotations directly
   ```python
   # Available in output: pred_global_rots
   # Shape: (127, 3, 3) rotation matrices
   ```

2. **Pose Parameter Comparison**: Compare MHR parameters directly
   - Body pose parameters
   - Shape parameters
   - Scale parameters

3. **Mesh-based Metrics**: 
   - Vertex-to-vertex distances
   - Surface area differences
   - Volume differences

## Integration Strategy

### Step 1: Replace MediaPipe Detection

Replace `PoseComparator.detect_pose()` to use SAM 3D Body:

```python
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

class PoseComparator3D:
    def __init__(self, checkpoint_path, mhr_path):
        model, cfg = load_sam_3d_body(checkpoint_path, device="cuda")
        self.estimator = SAM3DBodyEstimator(
            model, cfg,
            human_detector=...,  # Optional
            human_segmentor=...,  # Optional
            fov_estimator=...    # Optional
        )
    
    def detect_pose(self, image):
        outputs = self.estimator.process_one_image(image)
        return outputs[0]  # First detected person
```

### Step 2: Extend Comparison Metrics

Add 3D-specific comparison methods:

```python
def compare_poses_3d(self, teacher_output, student_output):
    """Compare poses using 3D data"""
    results = {}
    
    # 3D angle comparison
    results['angles_3d'] = self.compare_angles_3d(
        teacher_output['pred_keypoints_3d'],
        student_output['pred_keypoints_3d']
    )
    
    # 3D distance comparison
    results['distances_3d'] = self.compare_distances_3d(
        teacher_output['pred_keypoints_3d'],
        student_output['pred_keypoints_3d']
    )
    
    # Joint rotation comparison
    results['rotations'] = self.compare_rotations(
        teacher_output['pred_global_rots'],
        student_output['pred_global_rots']
    )
    
    return results
```

### Step 3: Enhanced Visualization

Use 3D visualization capabilities:

```python
from sam_3d_body.visualization import renderer, skeleton_visualizer

def visualize_comparison_3d(self, teacher_output, student_output):
    # Render 3D meshes side-by-side
    # Overlay skeletons
    # Highlight differences
    pass
```

## Performance Considerations

### Advantages
- More accurate 3D pose estimation
- Better handling of occlusions
- More detailed hand tracking
- Can compare poses from different viewpoints

### Trade-offs
- Slower inference (requires GPU for real-time)
- Higher memory usage
- More complex setup (requires model checkpoints)
- Larger dependency footprint

## Pickleball-Specific Analysis: Three-Position Scoring System

### Position 1: Preparation Phase

**Key Metrics:**
- **Shoulder Angle**: Angle between shoulder-elbow-wrist (both arms)
  - Target: Proper shoulder rotation for power generation
  - Measurement: 3D angle from `pred_keypoints_3d` using indices:
    - Right: shoulder(6) → elbow(8) → wrist(41)
    - Left: shoulder(5) → elbow(7) → wrist(62)
  
- **Weight Distribution**: Analyze hip/knee angles and body center of mass
  - Measurement: Compare hip heights and knee angles
  - Keypoints: left_hip(9), right_hip(10), left_knee(11), right_knee(12)
  - Calculate: Hip height difference, knee flexion angles
  - Formula: `weight_on_leg = f(hip_height_diff, knee_angles)`

**Scoring Criteria:**
```python
def score_preparation(teacher_pose, student_pose):
    scores = {}
    
    # Shoulder angle (0-100 points)
    shoulder_angle_diff = compare_3d_angle(
        teacher_pose, student_pose, 
        joint_indices=[6, 8, 41]  # right shoulder-elbow-wrist
    )
    scores['shoulder_angle'] = max(0, 100 - abs(shoulder_angle_diff) * 2)
    
    # Weight distribution (0-100 points)
    weight_dist_diff = compare_weight_distribution(
        teacher_pose, student_pose
    )
    scores['weight_distribution'] = max(0, 100 - abs(weight_dist_diff) * 3)
    
    # Overall preparation score (weighted average)
    preparation_score = (
        scores['shoulder_angle'] * 0.6 + 
        scores['weight_distribution'] * 0.4
    )
    
    return {
        'preparation_score': preparation_score,
        'components': scores
    }
```

### Position 2: Point of Contact

**Key Metrics:**
- **Paddle Position**: 3D position of wrist/hand relative to body
  - Measurement: Distance from wrist to body center
  - Keypoints: right_wrist(41), left_wrist(62), body_center (midpoint of hips)
  - Calculate: `paddle_position = wrist_3d - body_center_3d`
  
- **Body Alignment**: Torso orientation and hip position
  - Measurement: Torso angle (shoulder-to-hip line)
  - Keypoints: left_shoulder(5), right_shoulder(6), left_hip(9), right_hip(10)
  
- **Contact Point Height**: Vertical position of paddle at contact
  - Measurement: Z-coordinate (depth) of wrist keypoint
  - Compare: Teacher vs student paddle height

**Scoring Criteria:**
```python
def score_contact(teacher_pose, student_pose):
    scores = {}
    
    # Paddle position accuracy (0-100 points)
    paddle_pos_diff = np.linalg.norm(
        teacher_pose['pred_keypoints_3d'][41] -  # right_wrist
        student_pose['pred_keypoints_3d'][41]
    )
    scores['paddle_position'] = max(0, 100 - paddle_pos_diff * 10)
    
    # Body alignment (0-100 points)
    torso_angle_diff = compare_torso_angle(teacher_pose, student_pose)
    scores['body_alignment'] = max(0, 100 - abs(torso_angle_diff) * 2)
    
    # Contact height (0-100 points)
    height_diff = abs(
        teacher_pose['pred_keypoints_3d'][41][2] -  # Z-coordinate
        student_pose['pred_keypoints_3d'][41][2]
    )
    scores['contact_height'] = max(0, 100 - height_diff * 20)
    
    # Overall contact score
    contact_score = (
        scores['paddle_position'] * 0.5 +
        scores['body_alignment'] * 0.3 +
        scores['contact_height'] * 0.2
    )
    
    return {
        'contact_score': contact_score,
        'components': scores
    }
```

### Position 3: Finish Point

**Key Metrics:**
- **Paddle Finish Position**: Final paddle location after contact
  - Measurement: 3D position of wrist at finish
  - Compare: Teacher finish position vs student
  
- **Follow-Through Angle**: Arm extension and rotation
  - Measurement: Shoulder-elbow-wrist angle at finish
  - Keypoints: Same as preparation but at finish position
  
- **Body Rotation**: Hip and shoulder rotation through the shot
  - Measurement: Change in body orientation from contact to finish
  - Calculate: Rotation matrix difference using `pred_global_rots`

**Scoring Criteria:**
```python
def score_finish(teacher_pose, student_pose):
    scores = {}
    
    # Finish position (0-100 points)
    finish_pos_diff = np.linalg.norm(
        teacher_pose['pred_keypoints_3d'][41] -  # right_wrist
        student_pose['pred_keypoints_3d'][41]
    )
    scores['finish_position'] = max(0, 100 - finish_pos_diff * 10)
    
    # Follow-through angle (0-100 points)
    follow_through_diff = compare_3d_angle(
        teacher_pose, student_pose,
        joint_indices=[6, 8, 41]  # shoulder-elbow-wrist
    )
    scores['follow_through'] = max(0, 100 - abs(follow_through_diff) * 2)
    
    # Body rotation (0-100 points)
    rotation_diff = compare_body_rotation(
        teacher_pose['pred_global_rots'],
        student_pose['pred_global_rots']
    )
    scores['body_rotation'] = max(0, 100 - rotation_diff * 5)
    
    # Overall finish score
    finish_score = (
        scores['finish_position'] * 0.4 +
        scores['follow_through'] * 0.4 +
        scores['body_rotation'] * 0.2
    )
    
    return {
        'finish_score': finish_score,
        'components': scores
    }
```

### Cumulative Scoring System

**Overall Score Calculation:**
```python
def calculate_cumulative_score(preparation_score, contact_score, finish_score):
    """
    Calculate weighted cumulative score across all three positions.
    
    Weights can be adjusted based on coaching priorities:
    - Preparation: Foundation for good technique
    - Contact: Most critical moment
    - Finish: Ensures proper follow-through
    """
    weights = {
        'preparation': 0.3,  # 30%
        'contact': 0.5,      # 50% (most important)
        'finish': 0.2        # 20%
    }
    
    cumulative_score = (
        preparation_score * weights['preparation'] +
        contact_score * weights['contact'] +
        finish_score * weights['finish']
    )
    
    return {
        'cumulative_score': cumulative_score,
        'breakdown': {
            'preparation': preparation_score,
            'contact': contact_score,
            'finish': finish_score
        },
        'weights': weights
    }
```

### Implementation with SAM 3D Body

**Key Advantages for Three-Position Analysis:**

1. **3D Accuracy**: True 3D measurements eliminate perspective distortion
2. **Hand Detail**: 20 keypoints per hand for precise paddle tracking
3. **Body Alignment**: Full body mesh for weight distribution analysis
4. **Rotation Matrices**: Direct access to joint rotations via `pred_global_rots`

**Required Keypoint Mappings:**

```python
# Key keypoint indices for pickleball analysis
KEYPOINT_INDICES = {
    # Upper body
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 62,
    'right_wrist': 41,
    
    # Lower body
    'left_hip': 9,
    'right_hip': 10,
    'left_knee': 11,
    'right_knee': 12,
    'left_ankle': 13,
    'right_ankle': 14,
    
    # Body center (calculated)
    'body_center': None  # (left_hip + right_hip) / 2
}
```

## Recommended Implementation Approach

### Phase 1: Single Image Analysis
1. Load SAM 3D Body model
2. Process three images (preparation, contact, finish)
3. Extract 3D keypoints for each position
4. Implement scoring functions for each position

### Phase 2: Comparison System
1. Load teacher reference poses (3 positions)
2. Compare student poses against teacher
3. Calculate individual position scores
4. Compute cumulative score

### Phase 3: Visualization
1. Overlay 3D skeletons on images
2. Highlight differences between teacher/student
3. Display scores and feedback

### Phase 4: Video Analysis (Future)
1. Extract frames at key moments
2. Track pose across sequence
3. Analyze motion quality

## Next Steps

1. **Set up SAM 3D Body**: Load model and test on pickleball images
2. **Implement position detection**: Identify preparation/contact/finish frames
3. **Create scoring functions**: Implement the three scoring methods above
4. **Build comparison pipeline**: Integrate with existing MediaPipe workflow
5. **Add visualization**: Show scores and pose differences
6. **Calibrate thresholds**: Adjust scoring weights based on coaching feedback
