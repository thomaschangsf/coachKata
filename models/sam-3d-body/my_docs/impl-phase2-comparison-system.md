# Phase 2: Comparison System - Implementation Plan

## Goal

Implement a system that can:
1. Learn ideal pose ranges from multiple teacher examples (learning phase)
2. Compare student poses against teacher-learned ranges
3. Generate scores and feedback for improvement
4. Visualize differences between student and teacher poses
5. Provide actionable feedback to help users improve their stroke

## Overview

Phase 2 builds on Phase 1 by adding **comparison capabilities** - comparing student poses against teacher reference poses. The system learns from multiple teacher examples to derive statistical ranges, then compares student metrics against these ranges to provide scores and feedback.

## Key Design Decisions

### 1. Teacher Reference Storage
- **Format**: Cached pose data (JSON) or images (processed on-the-fly)
- **Rationale**: Flexibility to use pre-processed data for speed or raw images for convenience

### 2. Range Derivation
- **Multiple teacher poses**: Percentiles (25th-75th) for robust range calculation
- **Single teacher pose**: Normal distribution with fixed percentage (±10% default)
- **Rationale**: Handles both single-example and multi-example scenarios

### 3. Comparison Approach
- **Method**: Compare raw metrics (shoulder angle, weight distribution, etc.)
- **Configurability**: External config file defines which metrics to compare per stroke type
- **Rationale**: Flexible system that can adapt to different pickleball strokes

### 4. API Design
- **Structure**: Class-based API (`PoseComparator` class)
- **Estimator Management**: Pass estimator to constructor (caller manages model lifecycle)
- **Rationale**: Efficient resource usage - one estimator for all comparisons

### 5. Visualization & Feedback
- **Visualization**: Difference visualization with color-coded keypoints and arrows (configurable)
- **Feedback**: Prioritized action items + textual feedback with specific values
- **Rationale**: Clear, actionable guidance for users

## Architecture Components

### 1. Configuration Module

**File:** `libraries/src/pickleball_pose_analyzer/comparison/config_loader.py`

**Purpose:** Load and validate comparison configuration from external files

**Responsibilities:**
- Load configuration from JSON/YAML files
- Validate configuration structure
- Provide default configurations
- Map stroke types to metric keys

**Key Functions:**
```python
def load_comparison_config(config_path: str) -> dict[str, Any]:
    """
    Load comparison configuration from file.
    
    Args:
        config_path: Path to JSON/YAML config file
    
    Returns:
        Configuration dictionary with:
        - stroke_types: dict mapping stroke names to metric configurations
        - scoring_methods: dict mapping metrics to scoring functions
        - visualization_settings: dict with visualization options
    """
    pass

def get_metrics_for_stroke(config: dict, stroke_type: str) -> list[str]:
    """
    Get list of metrics to compare for a given stroke type.
    
    Args:
        config: Configuration dictionary
        stroke_type: Name of stroke (e.g., "serve", "forehand")
    
    Returns:
        List of metric keys (e.g., ["shoulder_angle", "weight_distribution"])
    """
    pass
```

**Configuration File Structure:**
```json
{
  "stroke_types": {
    "serve": {
      "preparation": ["shoulder_angle", "weight_distribution", "knee_bend"],
      "contact": ["paddle_position", "body_alignment", "contact_height"],
      "finish": ["finish_position", "follow_through_angle", "body_rotation"]
    },
    "forehand": {
      "preparation": ["shoulder_angle", "weight_distribution"],
      "contact": ["paddle_position", "contact_height"],
      "finish": ["follow_through_angle"]
    }
  },
  "scoring_methods": {
    "shoulder_angle": {
      "type": "tolerance",
      "tolerance": 0.2
    },
    "weight_distribution": {
      "type": "distance",
      "max_distance": 0.1
    }
  },
  "visualization": {
    "show_colors": true,
    "show_arrows": true,
    "color_thresholds": {
      "good": 80,
      "fair": 50,
      "poor": 0
    }
  },
  "single_pose_tolerance": {
    "percentage": 0.10,
    "default_std_dev": 0.05
  }
}
```

### 2. Learning Phase Module

**File:** `libraries/src/pickleball_pose_analyzer/comparison/teacher_learner.py`

**Purpose:** Process teacher examples and derive statistical ranges

**Responsibilities:**
- Process teacher pose data (from images or cached data)
- Calculate statistical ranges (percentiles or normal distribution)
- Save learned ranges to JSON files
- Handle single vs. multiple teacher poses

**Key Functions:**
```python
def learn_teacher_ranges(
    teacher_poses: list[dict[str, Any]] | dict[str, list[dict[str, Any]]],
    stroke_type: str,
    config_path: str,
    output_path: str,
    estimator: SAM3DBodyEstimator | None = None
) -> dict[str, Any]:
    """
    Learn ideal ranges from teacher pose examples.
    
    Args:
        teacher_poses: Either:
            - List of pose data dicts (single position)
            - Dict with keys 'preparation', 'contact', 'finish' (three positions)
        stroke_type: Name of stroke type (e.g., "serve")
        config_path: Path to configuration file
        output_path: Path to save learned ranges JSON
        estimator: Optional estimator (if teacher_poses contains image paths)
    
    Returns:
        Dictionary containing learned ranges for each position and metric
    """
    pass

def calculate_metric_ranges(
    metric_values: list[float],
    method: str = "percentile",
    percentile_low: float = 25.0,
    percentile_high: float = 75.0,
    tolerance_percentage: float = 0.10
) -> dict[str, float]:
    """
    Calculate statistical range for a metric.
    
    Args:
        metric_values: List of metric values from teacher poses
        method: "percentile" or "normal"
        percentile_low: Lower percentile (for percentile method)
        percentile_high: Upper percentile (for percentile method)
        tolerance_percentage: Percentage tolerance (for normal method with single value)
    
    Returns:
        Dictionary with 'min', 'max', 'mean', 'std_dev', 'method'
    """
    pass
```

**Output Format (Learned Ranges JSON):**
```json
{
  "stroke_type": "serve",
  "preparation": {
    "shoulder_angle": {
      "min": 85.0,
      "max": 105.0,
      "mean": 95.0,
      "std_dev": 5.2,
      "method": "percentile",
      "percentile_25": 85.0,
      "percentile_75": 105.0,
      "sample_count": 10
    },
    "weight_distribution": {
      "min": 0.45,
      "max": 0.60,
      "mean": 0.52,
      "std_dev": 0.03,
      "method": "percentile",
      "percentile_25": 0.45,
      "percentile_75": 0.60,
      "sample_count": 10
    }
  },
  "contact": {...},
  "finish": {...}
}
```

### 3. Comparison Module

**File:** `libraries/src/pickleball_pose_analyzer/comparison/pose_comparator.py`

**Purpose:** Main comparison class that compares student poses against teacher ranges

**Responsibilities:**
- Load teacher learned ranges
- Extract metrics from student poses
- Compare student metrics against teacher ranges
- Calculate scores using configurable scoring methods
- Generate overall position scores

**Key Class:**
```python
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
        estimator: SAM3DBodyEstimator,
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
        pass
    
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
        pass
    
    def _compare_position(
        self,
        student_pose: dict[str, Any],
        teacher_ranges: dict[str, Any],
        position_name: str,
        stroke_type: str
    ) -> dict[str, Any]:
        """Compare a single position."""
        pass
    
    def _score_metric(
        self,
        student_value: float,
        teacher_range: dict[str, float],
        scoring_config: dict[str, Any]
    ) -> float:
        """Score a single metric against teacher range."""
        pass
```

### 4. Metric Extraction Module

**File:** `libraries/src/pickleball_pose_analyzer/comparison/metric_extractor.py`

**Purpose:** Extract raw metrics from pose data (reusable from Phase 1, but may need v2)

**Responsibilities:**
- Extract metrics from pose data dictionaries
- Support configurable metric keys per stroke type
- Handle missing keypoints gracefully

**Key Functions:**
```python
def extract_metrics(
    pose_data: dict[str, Any],
    metric_keys: list[str],
    position_name: str
) -> dict[str, float | None]:
    """
    Extract raw metrics from pose data.
    
    Args:
        pose_data: Pose data dictionary from Phase 1
        metric_keys: List of metric keys to extract (e.g., ["shoulder_angle", "weight_distribution"])
        position_name: Position name ("preparation", "contact", "finish")
    
    Returns:
        Dictionary mapping metric keys to values (or None if unavailable)
    """
    pass

def extract_shoulder_angle(pose_data: dict[str, Any]) -> float | None:
    """Extract shoulder angle metric."""
    pass

def extract_weight_distribution(pose_data: dict[str, Any]) -> float | None:
    """Extract weight distribution metric."""
    pass

# ... similar functions for other metrics
```

### 5. Scoring Module (v2)

**File:** `libraries/src/pickleball_pose_analyzer/comparison/scoring_v2.py`

**Purpose:** Configurable scoring functions that work with teacher-derived ranges

**Responsibilities:**
- Implement multiple scoring methods (tolerance-based, distance-based, etc.)
- Support configurable scoring per metric
- Handle single vs. multiple teacher pose scenarios

**Key Functions:**
```python
def score_metric_tolerance(
    value: float,
    range_min: float,
    range_max: float,
    tolerance: float = 0.2
) -> float:
    """
    Score using tolerance-based method (similar to Phase 1 _score_metric).
    
    Args:
        value: Student metric value
        range_min: Minimum of teacher range
        range_max: Maximum of teacher range
        tolerance: Fraction of range to allow outside ideal
    
    Returns:
        Score from 0-100
    """
    pass

def score_metric_distance(
    value: float,
    range_min: float,
    range_max: float,
    max_distance: float
) -> float:
    """
    Score using distance-based method.
    
    Args:
        value: Student metric value
        range_min: Minimum of teacher range
        range_max: Maximum of teacher range
        max_distance: Maximum acceptable distance from range
    
    Returns:
        Score from 0-100
    """
    pass

def score_metric_percentile(
    value: float,
    teacher_values: list[float],
    percentile_low: float = 25.0,
    percentile_high: float = 75.0
) -> float:
    """
    Score based on percentile position within teacher distribution.
    
    Args:
        value: Student metric value
        teacher_values: List of teacher metric values
        percentile_low: Lower percentile threshold
        percentile_high: Upper percentile threshold
    
    Returns:
        Score from 0-100
    """
    pass
```

### 6. Feedback Generation Module

**File:** `libraries/src/pickleball_pose_analyzer/comparison/feedback_generator.py`

**Purpose:** Generate prioritized action items and textual feedback

**Responsibilities:**
- Analyze comparison results
- Prioritize issues by score impact
- Generate specific, actionable feedback
- Format feedback text with values and targets

**Key Functions:**
```python
def generate_feedback(
    comparison_results: dict[str, Any],
    stroke_type: str,
    top_n: int = 3
) -> dict[str, Any]:
    """
    Generate prioritized feedback from comparison results.
    
    Args:
        comparison_results: Output from PoseComparator.compare()
        stroke_type: Stroke type being analyzed
        top_n: Number of top issues to highlight
    
    Returns:
        Dictionary with:
        {
            'prioritized_issues': [
                {
                    'position': 'preparation',
                    'metric': 'shoulder_angle',
                    'priority': 1,
                    'student_value': 75.0,
                    'target_range': (85.0, 105.0),
                    'feedback_text': 'Shoulder angle is 10° too low. Aim for 85-105° range.',
                    'correction': 'Rotate right shoulder 10° more'
                },
                ...
            ],
            'overall_feedback': 'Focus on improving shoulder angle and weight distribution...',
            'strengths': [...],
            'improvements': [...]
        }
    """
    pass

def format_feedback_text(
    metric_name: str,
    student_value: float,
    target_range: tuple[float, float],
    difference: float
) -> str:
    """Format human-readable feedback text."""
    pass
```

### 7. Visualization Module

**File:** `libraries/src/pickleball_pose_analyzer/comparison/visualizer.py`

**Purpose:** Visualize differences between student and teacher poses

**Responsibilities:**
- Render side-by-side images (student vs. teacher)
- Color-code keypoints based on score
- Draw arrows showing needed corrections
- Support configurable visualization options

**Key Class:**
```python
class ComparisonVisualizer:
    """
    Visualize pose comparisons.
    
    Usage:
        visualizer = ComparisonVisualizer(
            show_colors=True,
            show_arrows=True,
            color_thresholds={'good': 80, 'fair': 50, 'poor': 0}
        )
        fig = visualizer.visualize_comparison(
            student_poses={...},
            teacher_poses={...},  # Optional - can use ranges only
            comparison_results={...},
            output_mode='return'  # 'return', 'save', 'display'
        )
    """
    
    def __init__(
        self,
        show_colors: bool = True,
        show_arrows: bool = True,
        color_thresholds: dict[str, int] | None = None
    ):
        """Initialize visualizer with configuration."""
        pass
    
    def visualize_comparison(
        self,
        student_poses: dict[str, dict[str, Any]],
        teacher_poses: dict[str, dict[str, Any]] | None,
        comparison_results: dict[str, Any],
        output_mode: str = 'return',
        output_path: str | None = None
    ) -> Any:
        """
        Create visualization of comparison.
        
        Args:
            student_poses: Student pose data (with image paths in metadata)
            teacher_poses: Optional teacher pose data (for side-by-side display)
            comparison_results: Output from PoseComparator.compare()
            output_mode: 'return', 'save', or 'display'
            output_path: Path to save image (if output_mode='save')
        
        Returns:
            Matplotlib figure (if output_mode='return'), None otherwise
        """
        pass
    
    def _draw_keypoints_with_colors(
        self,
        image: np.ndarray,
        keypoints_2d: np.ndarray,
        scores: dict[str, float],
        keypoint_mapping: dict[str, int]
    ) -> np.ndarray:
        """Draw keypoints with color coding based on scores."""
        pass
    
    def _draw_correction_arrows(
        self,
        image: np.ndarray,
        student_keypoints: np.ndarray,
        teacher_keypoints: np.ndarray,
        differences: dict[str, float]
    ) -> np.ndarray:
        """Draw arrows showing needed corrections."""
        pass
```

## File Structure

```
libraries/src/pickleball_pose_analyzer/
├── __init__.py                          # Update exports
├── comparison/                          # NEW: Phase 2 comparison module
│   ├── __init__.py
│   ├── config_loader.py                 # Configuration loading
│   ├── teacher_learner.py               # Learning phase functions
│   ├── pose_comparator.py               # Main comparison class
│   ├── metric_extractor.py              # Metric extraction (v2)
│   ├── scoring_v2.py                     # Configurable scoring (v2)
│   ├── feedback_generator.py            # Feedback generation
│   └── visualizer.py                    # Visualization module
├── data_structures.py                   # May need updates for comparison results
├── scoring.py                           # Phase 1 (will be refactored to use scoring_v2)
├── keypoint_extractor.py                # Phase 1 (may need v2)
└── ... (other Phase 1 files)
```

## Implementation Steps

### Step 1: Configuration System (Week 1)

**Tasks:**
1. Create `comparison/config_loader.py`
2. Define configuration file schema
3. Implement configuration loading and validation
4. Create example configuration files for different stroke types
5. Add unit tests

**Deliverables:**
- Working configuration loader
- Example config files
- Configuration validation

### Step 2: Metric Extraction (Week 1)

**Tasks:**
1. Create `comparison/metric_extractor.py`
2. Refactor metric extraction from Phase 1 scoring functions
3. Make metric extraction configurable (support different metrics per stroke)
4. Handle missing keypoints gracefully
5. Add unit tests

**Deliverables:**
- Reusable metric extraction functions
- Support for configurable metrics

### Step 3: Learning Phase (Week 1-2)

**Tasks:**
1. Create `comparison/teacher_learner.py`
2. Implement percentile range calculation
3. Implement normal distribution range for single pose
4. Implement pose data processing (from images or cached)
5. Implement JSON serialization of learned ranges
6. Add unit tests

**Deliverables:**
- Working learning function
- Learned ranges saved to JSON
- Support for single and multiple teacher poses

### Step 4: Scoring v2 (Week 2)

**Tasks:**
1. Create `comparison/scoring_v2.py`
2. Implement multiple scoring methods (tolerance, distance, percentile)
3. Make scoring configurable per metric
4. Refactor Phase 1 scoring to use scoring_v2 internally
5. Update Phase 1 code to use scoring_v2
6. Add unit tests

**Deliverables:**
- Configurable scoring system
- Phase 1 refactored to use new scoring

### Step 5: Comparison Class (Week 2)

**Tasks:**
1. Create `comparison/pose_comparator.py`
2. Implement `PoseComparator` class
3. Implement range loading from JSON
4. Implement metric comparison logic
5. Implement score calculation using configurable methods
6. Add unit tests

**Deliverables:**
- Working comparison class
- Comparison results matching Phase 1 output structure

### Step 6: Feedback Generation (Week 2-3)

**Tasks:**
1. Create `comparison/feedback_generator.py`
2. Implement issue prioritization algorithm
3. Implement feedback text generation
4. Implement correction suggestions
5. Add unit tests

**Deliverables:**
- Feedback generation system
- Prioritized action items
- Textual feedback with specific values

### Step 7: Visualization (Week 3)

**Tasks:**
1. Create `comparison/visualizer.py`
2. Implement `ComparisonVisualizer` class
3. Implement color-coded keypoint rendering
4. Implement correction arrow drawing
5. Implement side-by-side image display
6. Add configurable options (show/hide colors, arrows)
7. Add unit tests

**Deliverables:**
- Working visualization system
- Configurable visualization options

### Step 8: Integration and Testing (Week 3)

**Tasks:**
1. Integrate all modules
2. Create end-to-end test with real data
3. Test learning phase with multiple teacher examples
4. Test comparison with various student poses
5. Test visualization and feedback generation
6. Performance testing

**Deliverables:**
- Integrated system
- End-to-end tests
- Performance benchmarks

### Step 9: Jupyter Notebook (Week 3)

**Tasks:**
1. Create `notebooks/sam3dbody/phase2_comparison_verification.ipynb`
2. Demonstrate learning phase step-by-step
3. Demonstrate comparison step-by-step
4. Show visualization examples
5. Show feedback examples
6. Document key APIs

**Deliverables:**
- Complete verification notebook
- API documentation examples

### Step 10: Documentation and Cleanup (Week 3-4)

**Tasks:**
1. Update main `__init__.py` exports
2. Remove old file versions after refactoring
3. Update documentation
4. Create usage examples
5. Final testing and bug fixes

**Deliverables:**
- Clean codebase
- Complete documentation
- Usage examples

## Dependencies

### New Dependencies
- `scipy` (for percentile calculations) - or use `numpy.percentile`
- `matplotlib` (for visualization) - already used in Phase 1 notebook

### Existing Dependencies
- All Phase 1 dependencies (sam_3d_body, torch, numpy, cv2, etc.)

## Configuration File Example

**File:** `configs/pickleball_comparison.json`

```json
{
  "stroke_types": {
    "serve": {
      "preparation": {
        "metrics": ["shoulder_angle", "weight_distribution", "knee_bend"],
        "weights": {
          "shoulder_angle": 0.4,
          "weight_distribution": 0.3,
          "knee_bend": 0.2
        }
      },
      "contact": {
        "metrics": ["paddle_position", "body_alignment", "contact_height", "torso_angle"],
        "weights": {
          "paddle_position": 0.35,
          "body_alignment": 0.3,
          "contact_height": 0.25,
          "torso_angle": 0.1
        }
      },
      "finish": {
        "metrics": ["finish_position", "follow_through_angle", "body_rotation"],
        "weights": {
          "finish_position": 0.4,
          "follow_through": 0.35,
          "body_rotation": 0.25
        }
      }
    }
  },
  "scoring_methods": {
    "shoulder_angle": {
      "type": "tolerance",
      "tolerance": 0.2
    },
    "weight_distribution": {
      "type": "tolerance",
      "tolerance": 0.2
    },
    "knee_bend": {
      "type": "tolerance",
      "tolerance": 0.2
    }
  },
  "visualization": {
    "show_colors": true,
    "show_arrows": true,
    "color_thresholds": {
      "good": 80,
      "fair": 50,
      "poor": 0
    },
    "arrow_threshold": 5.0
  },
  "single_pose_tolerance": {
    "percentage": 0.10
  },
  "percentile_settings": {
    "low": 25.0,
    "high": 75.0
  }
}
```

## API Usage Examples

### Learning Phase

```python
from pickleball_pose_analyzer.comparison import learn_teacher_ranges
from pickleball_pose_analyzer import load_sam3d_model

# Load model (once)
estimator, config = load_sam3d_model(...)

# Option 1: Learn from cached pose data
teacher_poses = [
    {'preparation': prep_pose1, 'contact': contact_pose1, 'finish': finish_pose1},
    {'preparation': prep_pose2, 'contact': contact_pose2, 'finish': finish_pose2},
    # ... more teacher examples
]

learned_ranges = learn_teacher_ranges(
    teacher_poses=teacher_poses,
    stroke_type="serve",
    config_path="configs/pickleball_comparison.json",
    output_path="data/teacher_ranges/serve_ranges.json",
    estimator=None  # Not needed if using cached data
)

# Option 2: Learn from images
teacher_images = {
    'preparation': ['teacher_prep1.jpg', 'teacher_prep2.jpg', ...],
    'contact': ['teacher_contact1.jpg', ...],
    'finish': ['teacher_finish1.jpg', ...]
}

learned_ranges = learn_teacher_ranges(
    teacher_poses=teacher_images,
    stroke_type="serve",
    config_path="configs/pickleball_comparison.json",
    output_path="data/teacher_ranges/serve_ranges.json",
    estimator=estimator  # Needed to process images
)
```

### Comparison Phase

```python
from pickleball_pose_analyzer.comparison import PoseComparator
from pickleball_pose_analyzer import load_sam3d_model, process_three_positions

# Load model (once, reuse for all comparisons)
estimator, config = load_sam3d_model(...)

# Initialize comparator
comparator = PoseComparator(
    teacher_ranges_path="data/teacher_ranges/serve_ranges.json",
    config_path="configs/pickleball_comparison.json",
    estimator=estimator,
    stroke_type="serve"
)

# Process student images
student_poses = process_three_positions(
    estimator,
    preparation_path="student_prep.jpg",
    contact_path="student_contact.jpg",
    finish_path="student_finish.jpg"
)

# Compare against teacher
comparison_results = comparator.compare(
    student_poses=student_poses,
    stroke_type="serve"
)

# Access results
print(f"Preparation Score: {comparison_results['preparation']['preparation_score']}")
print(f"Contact Score: {comparison_results['contact']['contact_score']}")
print(f"Overall Score: {comparison_results['cumulative_score']}")
```

### Feedback Generation

```python
from pickleball_pose_analyzer.comparison import generate_feedback

feedback = generate_feedback(
    comparison_results=comparison_results,
    stroke_type="serve",
    top_n=3
)

print("Top Issues:")
for issue in feedback['prioritized_issues']:
    print(f"{issue['priority']}. {issue['feedback_text']}")
    print(f"   Correction: {issue['correction']}")
```

### Visualization

```python
from pickleball_pose_analyzer.comparison import ComparisonVisualizer

visualizer = ComparisonVisualizer(
    show_colors=True,
    show_arrows=True
)

# Visualize comparison
fig = visualizer.visualize_comparison(
    student_poses=student_poses,
    teacher_poses=teacher_poses,  # Optional
    comparison_results=comparison_results,
    output_mode='display'  # or 'save' or 'return'
)
```

## Testing Strategy

### Unit Tests
- Configuration loading and validation
- Metric extraction for each metric type
- Range calculation (percentiles, normal distribution)
- Scoring methods (tolerance, distance, percentile)
- Feedback generation
- Visualization rendering

### Integration Tests
- End-to-end learning phase
- End-to-end comparison phase
- Learning + comparison workflow
- Visualization + feedback workflow

### Test Data
- Sample teacher pose data (multiple examples)
- Sample student pose data
- Edge cases (single teacher pose, missing metrics, etc.)

## Success Criteria

1. ✅ Can learn ranges from multiple teacher examples
2. ✅ Can learn ranges from single teacher example (normal distribution)
3. ✅ Can compare student poses against learned ranges
4. ✅ Generates scores matching Phase 1 structure
5. ✅ Provides prioritized, actionable feedback
6. ✅ Visualizes differences clearly (colors + arrows)
7. ✅ Configurable per stroke type
8. ✅ Efficient resource usage (one estimator for all operations)
9. ✅ Complete verification notebook
10. ✅ All Phase 1 code refactored to use new versions

## Future Enhancements (Out of Scope for Phase 2)

- Video sequence analysis
- Temporal pose tracking
- Multiple teacher comparison (ensemble)
- Adaptive learning (update ranges as more teacher data is added)
- 3D mesh visualization
- Interactive feedback interface
