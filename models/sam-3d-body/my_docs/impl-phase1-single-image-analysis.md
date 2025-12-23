# Phase 1: Single Image Analysis - Implementation Plan

## Goal

Implement a system that can:
1. Load and initialize the SAM 3D Body model
2. Process three pickleball pose images (preparation, contact, finish positions)
3. Extract 3D keypoints and pose data for each position
4. Implement scoring functions for each position (without comparison yet - just individual position analysis)

## Overview

Phase 1 focuses on **single image analysis** - processing individual images and extracting meaningful pose data. This is the foundation before we can compare poses in Phase 2.

## Architecture Components

### 1. Model Initialization Module

**File:** `pickleball_pose_analyzer/model_loader.py`

**Purpose:** Handle SAM 3D Body model loading and initialization

**Responsibilities:**
- Load model checkpoint from HuggingFace or local path
- Initialize SAM3DBodyEstimator with optional components (detector, segmentor, FOV estimator)
- Handle device selection (CUDA/CPU)
- Provide model configuration access

**Key Functions:**
```python
# Path setup handled in __init__.py
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body, load_sam_3d_body_hf

def load_sam3d_model(
    checkpoint_path: Optional[str] = None,
    hf_repo_id: str = "facebook/sam-3d-body-dinov3",
    device: str = "auto",
    use_detector: bool = True,
    use_segmentor: bool = False,
    use_fov_estimator: bool = True
) -> Tuple[SAM3DBodyEstimator, Dict]:
    """
    Load and initialize SAM 3D Body model.
    
    Args:
        checkpoint_path: Local path to model checkpoint (optional)
        hf_repo_id: HuggingFace repository ID (used if checkpoint_path is None)
        device: "auto", "cuda", or "cpu"
        use_detector: Whether to use human detector
        use_segmentor: Whether to use segmentation (SAM2)
        use_fov_estimator: Whether to use FOV estimator (MoGe2)
    
    Returns:
        estimator: Initialized SAM3DBodyEstimator
        config: Model configuration dictionary
    """
    pass
```

### 2. Image Processing Module

**File:** `pickleball_pose_analyzer/image_processor.py`

**Purpose:** Process individual images and extract pose data

**Responsibilities:**
- Load images from file paths
- Validate image format and quality
- Process images through SAM 3D Body estimator
- Extract and structure pose data (keypoints, parameters, mesh)

**Key Functions:**
```python
def process_single_image(
    estimator: SAM3DBodyEstimator,
    image_path: str,
    position_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Process a single image and extract pose data.
    
    Args:
        estimator: Initialized SAM3DBodyEstimator
        image_path: Path to image file
        position_name: Name of position (e.g., "preparation", "contact", "finish")
    
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
    pass

def process_three_positions(
    estimator: SAM3DBodyEstimator,
    preparation_path: str,
    contact_path: str,
    finish_path: str
) -> Dict[str, Dict[str, Any]]:
    """
    Process all three pickleball positions.
    
    Returns:
        Dictionary with keys: 'preparation', 'contact', 'finish'
        Each value is the output from process_single_image()
    """
    pass
```

### 3. Keypoint Extraction Module

**File:** `pickleball_pose_analyzer/keypoint_extractor.py`

**Purpose:** Extract and structure keypoints for pickleball analysis

**Responsibilities:**
- Extract specific keypoints needed for pickleball analysis
- Calculate derived keypoints (e.g., body center from hips)
- Provide keypoint access by name (not just index)
- Validate keypoint visibility/quality

**Key Functions:**
```python
# Keypoint indices for pickleball analysis
PICKLEBALL_KEYPOINTS = {
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
}

def extract_keypoints(pose_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Extract keypoints by name from pose data.
    
    Returns:
        Dictionary mapping keypoint names to 3D coordinates
        Example: {'left_shoulder': array([x, y, z]), ...}
    """
    pass

def calculate_body_center(pose_data: Dict[str, Any]) -> np.ndarray:
    """
    Calculate body center as midpoint between left and right hips.
    
    Returns:
        3D coordinates of body center: array([x, y, z])
    """
    pass
```

### 4. Scoring Functions Module

**File:** `pickleball_pose_analyzer/scoring.py`

**Purpose:** Implement scoring functions for each position (without comparison)

**Responsibilities:**
- Calculate metrics for each position independently
- Score preparation position (shoulder angle, weight distribution)
- Score contact position (paddle position, body alignment, contact height)
- Score finish position (finish position, follow-through angle, body rotation)
- Return structured scores (0-100 scale)

**Key Functions:**
```python
def calculate_3d_angle(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray
) -> float:
    """
    Calculate 3D angle at point_b between vectors ba and bc.
    
    Returns:
        Angle in degrees
    """
    pass

def score_preparation_position(pose_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score preparation position.
    
    Returns:
        {
            'shoulder_angle_right': float,  # degrees
            'shoulder_angle_left': float,
            'weight_distribution': float,  # 0-1 ratio
            'hip_height_diff': float,  # meters
            'knee_angles': dict,  # left and right knee angles
            'scores': {
                'shoulder_angle': float,  # 0-100
                'weight_distribution': float,  # 0-100
            },
            'preparation_score': float  # 0-100 (weighted average)
        }
    """
    pass

def score_contact_position(pose_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score contact position.
    
    Returns:
        {
            'paddle_position': np.ndarray,  # 3D position relative to body
            'paddle_height': float,  # Z-coordinate
            'torso_angle': float,  # degrees
            'body_alignment': float,  # alignment metric
            'scores': {
                'paddle_position': float,  # 0-100
                'body_alignment': float,  # 0-100
                'contact_height': float,  # 0-100
            },
            'contact_score': float  # 0-100 (weighted average)
        }
    """
    pass

def score_finish_position(pose_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score finish position.
    
    Returns:
        {
            'finish_position': np.ndarray,  # 3D wrist position
            'follow_through_angle': float,  # degrees
            'body_rotation': float,  # rotation metric
            'scores': {
                'finish_position': float,  # 0-100
                'follow_through': float,  # 0-100
                'body_rotation': float,  # 0-100
            },
            'finish_score': float  # 0-100 (weighted average)
        }
    """
    pass
```

### 5. Data Structures Module

**File:** `pickleball_pose_analyzer/data_structures.py`

**Purpose:** Define data structures and constants

**Responsibilities:**
- Define keypoint mappings
- Define scoring weights and thresholds
- Define data format specifications
- Type hints and validation

**Key Definitions:**
```python
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class PoseData:
    """Structured pose data container"""
    position_name: str
    keypoints_3d: np.ndarray  # (70, 3)
    keypoints_2d: np.ndarray  # (70, 2)
    vertices: np.ndarray  # (18439, 3)
    body_pose_params: np.ndarray
    hand_pose_params: np.ndarray
    shape_params: np.ndarray
    scale_params: np.ndarray
    global_rots: np.ndarray  # (127, 3, 3)
    bbox: np.ndarray
    focal_length: float
    metadata: Dict[str, Any]

@dataclass
class PositionScore:
    """Score data for a single position"""
    position_name: str
    metrics: Dict[str, float]  # Raw metrics (angles, distances, etc.)
    component_scores: Dict[str, float]  # Individual component scores (0-100)
    overall_score: float  # Weighted overall score (0-100)
```

### 6. Main Analysis Script

**File:** `pickleball_pose_analyzer/analyze_positions.py`

**Purpose:** Main entry point for Phase 1 analysis

**Responsibilities:**
- Command-line interface
- Orchestrate model loading, image processing, and scoring
- Output results (JSON, console, files)
- Basic visualization

**Usage:**

**From Notebook (models/sam-3d-body/notebook/):**
```python
import sys
import os

# Setup paths
project_root = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
sys.path.insert(0, project_root)
sam3d_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, sam3d_root)

# Import pickleball analyzer
from libraries.src.pickleball_pose_analyzer import (
    load_sam3d_model,
    process_three_positions,
    score_preparation_position,
    score_contact_position,
    score_finish_position
)

# Use it
estimator, config = load_sam3d_model()
results = process_three_positions(
    estimator,
    "preparation.jpg",
    "contact.jpg", 
    "finish.jpg"
)
```

**From Command Line:**
```bash
# From project root (coachKata/):
python -m libraries.src.pickleball_pose_analyzer.analyze_positions \
    --preparation data/pickelball/preparation.jpg \
    --contact data/pickelball/contact.jpg \
    --finish data/pickelball/finish.jpg \
    --output results.json \
    --visualize
```

## Implementation Steps

### Step 1: Project Structure Setup

```
coachKata/
├── libraries/
│   └── src/
│       ├── coackata/              # Existing
│       │   ├── __init__.py
│       │   └── compare_image.py
│       └── pickleball_pose_analyzer/  # NEW
│           ├── __init__.py         # Handles path setup for sam_3d_body imports
│           ├── model_loader.py
│           ├── image_processor.py
│           ├── keypoint_extractor.py
│           ├── scoring.py
│           ├── data_structures.py
│           └── analyze_positions.py
└── models/
    └── sam-3d-body/
        ├── sam_3d_body/           # SAM 3D Body code
        ├── tools/                  # Existing tools
        ├── notebook/               # Notebooks (will import pickleball_pose_analyzer)
        └── my_docs/
            └── impl-phase1-single-image-analysis.md
```

**Path Setup Implementation:**

The `__init__.py` will automatically configure paths so imports work seamlessly:
```python
# libraries/src/pickleball_pose_analyzer/__init__.py
import sys
import os

def _setup_sam3d_paths():
    """Automatically add models/sam-3d-body to Python path"""
    # Get path to this file
    current_file = os.path.abspath(__file__)
    # Navigate: libraries/src/pickleball_pose_analyzer/__init__.py -> ../../.. -> coachKata/
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), '../../..'))
    sam3d_path = os.path.join(project_root, 'models', 'sam-3d-body')
    
    if os.path.exists(sam3d_path) and sam3d_path not in sys.path:
        sys.path.insert(0, sam3d_path)
        return True
    return False

# Auto-setup on import
_setup_sam3d_paths()

# Now imports work
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf

# Export main functions
from .model_loader import load_sam3d_model
from .image_processor import process_single_image, process_three_positions
from .keypoint_extractor import extract_keypoints, calculate_body_center
from .scoring import (
    calculate_3d_angle,
    score_preparation_position,
    score_contact_position,
    score_finish_position
)

__all__ = [
    'load_sam3d_model',
    'process_single_image',
    'process_three_positions',
    'extract_keypoints',
    'calculate_body_center',
    'calculate_3d_angle',
    'score_preparation_position',
    'score_contact_position',
    'score_finish_position',
]
```

**Notebook Usage Example:**

In `models/sam-3d-body/notebook/pickleball_analysis.ipynb`:
```python
import sys
import os

# Add project root to path (to access libraries.src.pickleball_pose_analyzer)
project_root = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add sam-3d-body to path (for sam_3d_body imports - already done by pickleball_pose_analyzer)
# But we can also do it here for direct sam_3d_body access
sam3d_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if sam3d_root not in sys.path:
    sys.path.insert(0, sam3d_root)

# Now import works seamlessly!
from libraries.src.pickleball_pose_analyzer import (
    load_sam3d_model,
    process_three_positions,
    score_preparation_position,
    score_contact_position,
    score_finish_position
)

# Use it
estimator, config = load_sam3d_model(hf_repo_id="facebook/sam-3d-body-dinov3")

# Process images
results = process_three_positions(
    estimator,
    "../data/pickelball/preparation.jpg",
    "../data/pickelball/contact.jpg",
    "../data/pickelball/finish.jpg"
)

# Score each position
prep_score = score_preparation_position(results['preparation'])
contact_score = score_contact_position(results['contact'])
finish_score = score_finish_position(results['finish'])

print(f"Preparation: {prep_score['overall_score']:.1f}/100")
print(f"Contact: {contact_score['overall_score']:.1f}/100")
print(f"Finish: {finish_score['overall_score']:.1f}/100")
```

**Feasibility Confirmation:**

✅ **Yes, this is feasible!** The path setup in `__init__.py` automatically handles the `sam_3d_body` import, and notebooks just need to add the project root to access `libraries.src.pickleball_pose_analyzer`.

**Import Strategy:**

Since `pickleball_pose_analyzer` is in `libraries/src/` but needs to import `sam_3d_body` from `models/sam-3d-body/`, we'll handle path setup:

1. **In notebooks** (models/sam-3d-body/notebook/):
   ```python
   import sys
   import os
   
   # Add project root to path
   project_root = os.path.abspath(os.path.join(os.getcwd(), '../../..'))
   sys.path.insert(0, project_root)
   
   # Add models/sam-3d-body to path (for sam_3d_body imports)
   sam3d_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
   sys.path.insert(0, sam3d_root)
   
   # Now can import both:
   from libraries.src.pickleball_pose_analyzer import ...
   from sam_3d_body import ...
   ```

2. **In pickleball_pose_analyzer modules**:
   ```python
   # model_loader.py
   import sys
   import os
   
   # Add models/sam-3d-body to path if not already there
   # This allows importing sam_3d_body
   project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
   sam3d_path = os.path.join(project_root, 'models', 'sam-3d-body')
   if sam3d_path not in sys.path:
       sys.path.insert(0, sam3d_path)
   
   from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body_hf
   ```

**Alternative: Helper function for path setup**

Create a utility function to handle path setup automatically:
```python
# pickleball_pose_analyzer/__init__.py
import sys
import os

def _setup_paths():
    """Add necessary paths for imports"""
    # Get project root (assuming we're in libraries/src/pickleball_pose_analyzer)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
    sam3d_path = os.path.join(project_root, 'models', 'sam-3d-body')
    
    if sam3d_path not in sys.path:
        sys.path.insert(0, sam3d_path)
    
    return project_root, sam3d_path

# Auto-setup on import
_setup_paths()
```

### Step 2: Model Loading (Week 1)

**Tasks:**
1. Create `model_loader.py`
2. Implement model loading from HuggingFace
3. Handle device selection (CUDA/CPU)
4. Add error handling for missing dependencies
5. Test with sample images

**Deliverables:**
- Working model loader that can initialize SAM3DBodyEstimator
- Unit tests for model loading

### Step 3: Image Processing (Week 1)

**Tasks:**
1. Create `image_processor.py`
2. Implement `process_single_image()`
3. Implement `process_three_positions()`
4. Add image validation (format, size, quality checks)
5. Handle errors (no person detected, multiple people, etc.)

**Deliverables:**
- Image processor that can extract pose data from images
- Error handling for edge cases

### Step 4: Keypoint Extraction (Week 1-2)

**Tasks:**
1. Create `keypoint_extractor.py`
2. Implement keypoint name mapping
3. Implement `extract_keypoints()`
4. Implement `calculate_body_center()`
5. Add keypoint validation (visibility, confidence)

**Deliverables:**
- Keypoint extractor with named access
- Helper functions for derived keypoints

### Step 5: Scoring Functions - Preparation (Week 2)

**Tasks:**
1. Create `scoring.py`
2. Implement `calculate_3d_angle()`
3. Implement `score_preparation_position()`
4. Calculate shoulder angles (left and right)
5. Calculate weight distribution metrics
6. Test with sample preparation images

**Deliverables:**
- Working preparation position scorer
- Test results on sample images

### Step 6: Scoring Functions - Contact (Week 2)

**Tasks:**
1. Implement `score_contact_position()`
2. Calculate paddle position (wrist relative to body center)
3. Calculate torso alignment angle
4. Calculate contact height
5. Test with sample contact images

**Deliverables:**
- Working contact position scorer
- Test results on sample images

### Step 7: Scoring Functions - Finish (Week 2)

**Tasks:**
1. Implement `score_finish_position()`
2. Calculate finish position (wrist 3D location)
3. Calculate follow-through angle
4. Calculate body rotation metric
5. Test with sample finish images

**Deliverables:**
- Working finish position scorer
- Test results on sample images

### Step 8: Main Script and Integration (Week 2-3)

**Tasks:**
1. Create `analyze_positions.py`
2. Implement command-line interface
3. Integrate all modules
4. Add JSON output format
5. Add basic visualization (optional)
6. Add logging and error reporting

**Deliverables:**
- Complete working script
- Documentation and usage examples

### Step 9: Testing and Validation (Week 3)

**Tasks:**
1. Test with real pickleball images
2. Validate scoring functions produce reasonable results
3. Test error cases (missing images, no person detected, etc.)
4. Performance testing (inference time)
5. Document known limitations

**Deliverables:**
- Test suite
- Validation report
- Performance benchmarks

## Dependencies

### Required
- `sam_3d_body` (from `models/sam-3d-body/sam_3d_body/` - path setup required)
- `torch` (PyTorch)
- `numpy`
- `opencv-python` (cv2)
- `typing` (for type hints - built-in)

### Optional
- `matplotlib` (for visualization)
- `json` (for output formatting - built-in)
- `argparse` (for CLI - built-in)

**Path Setup Note:** 
Since `pickleball_pose_analyzer` is in `libraries/src/` but needs to import `sam_3d_body` from `models/sam-3d-body/`, the `__init__.py` will automatically add the necessary paths to `sys.path` when the module is imported. This allows seamless imports from notebooks.

## Output Format

### JSON Output Structure

```json
{
  "preparation": {
    "position_name": "preparation",
    "image_path": "preparation.jpg",
    "keypoints_3d": {
      "left_shoulder": [x, y, z],
      "right_shoulder": [x, y, z],
      ...
    },
    "metrics": {
      "shoulder_angle_right": 145.3,
      "shoulder_angle_left": 142.1,
      "weight_distribution": 0.65,
      "hip_height_diff": 0.02
    },
    "scores": {
      "shoulder_angle": 85.0,
      "weight_distribution": 78.0
    },
    "overall_score": 82.2
  },
  "contact": { ... },
  "finish": { ... }
}
```

## Success Criteria

1. ✅ Can load SAM 3D Body model successfully
2. ✅ Can process three images (preparation, contact, finish)
3. ✅ Can extract 3D keypoints for each position
4. ✅ Can calculate scores for each position independently
5. ✅ Output is structured and usable for Phase 2
6. ✅ Error handling for common failure cases
7. ✅ Basic documentation and examples

## Known Limitations (Phase 1)

- No comparison with teacher/reference poses (Phase 2)
- No cumulative scoring across positions (Phase 2)
- Scoring thresholds are placeholders (need calibration)
- No video processing (Phase 4)
- Limited visualization (Phase 3)

## Test Organization

### Test Structure

Following Python testing best practices and your existing project patterns, tests will be organized as follows:

```
libraries/
└── src/
    └── pickleball_pose_analyzer/
        ├── __init__.py
        ├── model_loader.py
        ├── image_processor.py
        ├── keypoint_extractor.py
        ├── scoring.py
        ├── data_structures.py
        └── analyze_positions.py
└── tests/                                    # NEW: Test directory
    └── pickleball_pose_analyzer/
        ├── __init__.py
        ├── conftest.py                       # pytest fixtures and shared test utilities
        ├── test_model_loader.py
        ├── test_image_processor.py
        ├── test_keypoint_extractor.py
        ├── test_scoring.py
        ├── test_data_structures.py
        ├── test_analyze_positions.py
        ├── integration/
        │   ├── __init__.py
        │   ├── test_full_pipeline.py        # End-to-end tests
        │   └── test_three_positions.py       # Test all three positions together
        └── fixtures/                         # Test data and fixtures
            ├── __init__.py
            ├── sample_images/                # Sample pickleball images for testing
            │   ├── preparation_sample.jpg
            │   ├── contact_sample.jpg
            │   └── finish_sample.jpg
            └── mock_pose_data.py            # Mock pose data for unit tests
```

### Test File Organization

**1. Unit Tests (One test file per module)**

Each module has a corresponding test file that tests its functions in isolation:

- `test_model_loader.py` - Tests model loading, device selection, error handling
- `test_image_processor.py` - Tests image processing, validation, pose extraction
- `test_keypoint_extractor.py` - Tests keypoint extraction, body center calculation
- `test_scoring.py` - Tests scoring functions for each position
- `test_data_structures.py` - Tests data structure validation and serialization
- `test_analyze_positions.py` - Tests CLI interface and main script

**2. Integration Tests**

- `test_full_pipeline.py` - Tests complete workflow from image to score
- `test_three_positions.py` - Tests processing all three positions together

**3. Test Fixtures (`conftest.py`)**

Shared pytest fixtures for common test setup:

```python
# tests/pickleball_pose_analyzer/conftest.py
import pytest
import numpy as np
import sys
import os

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sam3d_path = os.path.join(project_root, 'models', 'sam-3d-body')
if sam3d_path not in sys.path:
    sys.path.insert(0, sam3d_path)

@pytest.fixture
def mock_estimator():
    """Mock SAM3DBodyEstimator for unit tests"""
    from unittest.mock import MagicMock
    estimator = MagicMock()
    # Setup mock return values
    return estimator

@pytest.fixture
def sample_pose_data():
    """Sample pose data dictionary for testing"""
    return {
        'pred_keypoints_3d': np.random.rand(70, 3),
        'pred_keypoints_2d': np.random.rand(70, 2),
        'pred_vertices': np.random.rand(18439, 3),
        'body_pose_params': np.random.rand(260),
        'hand_pose_params': np.random.rand(108),
        'shape_params': np.random.rand(45),
        'scale_params': np.random.rand(28),
        'pred_global_rots': np.random.rand(127, 3, 3),
        'bbox': np.array([100, 50, 300, 500]),
        'focal_length': 800.5,
    }

@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary sample image for testing"""
    import cv2
    import numpy as np
    
    # Create a simple test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (128, 128, 128)  # Gray background
    
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img)
    return str(img_path)
```

### Example Test File Structure

**Example: `test_scoring.py`**

```python
# tests/pickleball_pose_analyzer/test_scoring.py
import pytest
import numpy as np
import sys
import os

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(project_root, 'libraries', 'src'))

from pickleball_pose_analyzer.scoring import (
    calculate_3d_angle,
    score_preparation_position,
    score_contact_position,
    score_finish_position
)

class TestCalculate3DAngle:
    """Tests for 3D angle calculation"""
    
    def test_right_angle(self, sample_pose_data):
        """Test 90-degree angle calculation"""
        point_a = np.array([1, 0, 0])
        point_b = np.array([0, 0, 0])
        point_c = np.array([0, 1, 0])
        
        angle = calculate_3d_angle(point_a, point_b, point_c)
        assert abs(angle - 90.0) < 0.1
    
    def test_straight_line(self, sample_pose_data):
        """Test 180-degree angle (straight line)"""
        point_a = np.array([-1, 0, 0])
        point_b = np.array([0, 0, 0])
        point_c = np.array([1, 0, 0])
        
        angle = calculate_3d_angle(point_a, point_b, point_c)
        assert abs(angle - 180.0) < 0.1

class TestScorePreparationPosition:
    """Tests for preparation position scoring"""
    
    def test_score_preparation_basic(self, sample_pose_data):
        """Test basic preparation scoring"""
        result = score_preparation_position(sample_pose_data)
        
        assert 'shoulder_angle_right' in result
        assert 'shoulder_angle_left' in result
        assert 'weight_distribution' in result
        assert 'scores' in result
        assert 'preparation_score' in result
        assert 0 <= result['preparation_score'] <= 100
    
    def test_score_preparation_missing_keypoints(self):
        """Test handling of missing keypoints"""
        incomplete_data = {'pred_keypoints_3d': np.random.rand(70, 3)}
        # Should handle gracefully or raise appropriate error
        with pytest.raises(KeyError):
            score_preparation_position(incomplete_data)

class TestScoreContactPosition:
    """Tests for contact position scoring"""
    # Similar structure...

class TestScoreFinishPosition:
    """Tests for finish position scoring"""
    # Similar structure...
```

### Running Tests

**From project root:**
```bash
# Run all tests
pytest libraries/tests/

# Run specific test file
pytest libraries/tests/pickleball_pose_analyzer/test_scoring.py

# Run with coverage
pytest libraries/tests/ --cov=libraries.src.pickleball_pose_analyzer --cov-report=html

# Run only unit tests
pytest libraries/tests/pickleball_pose_analyzer/ -k "not integration"

# Run only integration tests
pytest libraries/tests/pickleball_pose_analyzer/integration/
```

**From notebook:**
```python
# Can also run tests programmatically
import pytest
pytest.main(['libraries/tests/pickleball_pose_analyzer/test_scoring.py', '-v'])
```

### Test Data Management

**Fixtures Directory:**
- `fixtures/sample_images/` - Small sample images for testing (can be git-ignored if large)
- `fixtures/mock_pose_data.py` - Functions to generate mock pose data for unit tests

**Test Images:**
- Use small, representative images for unit tests
- Integration tests can use actual pickleball images from `data/pickelball/`
- Consider using synthetic/test images to avoid large file sizes in repo

### Test Coverage Goals

- **Unit Tests**: 80%+ coverage for core functions
- **Integration Tests**: Cover main workflows
- **Edge Cases**: Test error handling, missing data, invalid inputs

### Dependencies for Testing

```python
# Add to libraries/pyproject.toml or requirements-test.txt
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
```

## Next Steps After Phase 1

- Phase 2: Add teacher/student comparison
- Phase 3: Enhanced visualization
- Phase 4: Video analysis
