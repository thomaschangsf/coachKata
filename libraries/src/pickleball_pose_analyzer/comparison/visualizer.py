"""
Visualization module for Phase 2 comparison system.

This module provides visualization of pose comparisons with color-coded
keypoints and correction arrows.
"""

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ..data_structures import PICKLEBALL_KEYPOINTS


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
            teacher_poses={...},  # Optional
            comparison_results={...},
            output_mode='return'  # 'return', 'save', 'display'
        )
    """

    def __init__(
        self,
        show_colors: bool = True,
        show_arrows: bool = True,
        color_thresholds: dict[str, int] | None = None,
        arrow_threshold: float = 5.0
    ):
        """
        Initialize visualizer with configuration.

        Args:
            show_colors: Whether to color-code keypoints based on scores
            show_arrows: Whether to draw correction arrows
            color_thresholds: Dict with 'good', 'fair', 'poor' score thresholds
            arrow_threshold: Minimum score difference to show arrow
        """
        self.show_colors = show_colors
        self.show_arrows = show_arrows
        self.color_thresholds = color_thresholds or {
            'good': 80,
            'fair': 50,
            'poor': 0
        }
        self.arrow_threshold = arrow_threshold

        # Color definitions
        self.colors = {
            'good': (0, 255, 0),      # Green
            'fair': (255, 165, 0),    # Orange
            'poor': (255, 0, 0),      # Red
            'teacher': (0, 0, 255),   # Blue
            'default': (255, 255, 255)  # White
        }

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
        positions = ['preparation', 'contact', 'finish']
        n_positions = sum(1 for p in positions if p in student_poses)

        if n_positions == 0:
            raise ValueError("No valid positions found in student_poses")

        # Create figure with subplots
        fig, axes = plt.subplots(
            n_positions,
            2 if teacher_poses else 1,
            figsize=(12 if teacher_poses else 6, 4 * n_positions)
        )

        if n_positions == 1:
            axes = axes.reshape(1, -1) if teacher_poses else axes.reshape(1, -1)

        row = 0
        for position in positions:
            if position not in student_poses:
                continue

            student_pose = student_poses[position]
            position_result = comparison_results.get(position, {})

            # Load student image
            student_img = self._load_image(student_pose)
            if student_img is None:
                continue

            # Draw student visualization
            student_vis = self._draw_keypoints_with_colors(
                student_img.copy(),
                student_pose,
                position_result,
                position,
                label='Student'
            )

            # Show student image
            if teacher_poses:
                ax_student = axes[row, 0] if n_positions > 1 else axes[0]
            else:
                ax_student = axes[row] if n_positions > 1 else axes
            ax_student.imshow(cv2.cvtColor(student_vis, cv2.COLOR_BGR2RGB))
            ax_student.set_title(f'{position.title()} - Student', fontsize=12, fontweight='bold')
            ax_student.axis('off')

            # Draw teacher image if available
            if teacher_poses and position in teacher_poses:
                teacher_pose = teacher_poses[position]
                teacher_img = self._load_image(teacher_pose)
                if teacher_img is not None:
                    teacher_vis = self._draw_teacher_keypoints(
                        teacher_img.copy(),
                        teacher_pose,
                        label='Teacher'
                    )

                    ax_teacher = axes[row, 1] if n_positions > 1 else axes[1]
                    ax_teacher.imshow(cv2.cvtColor(teacher_vis, cv2.COLOR_BGR2RGB))
                    ax_teacher.set_title(f'{position.title()} - Teacher', fontsize=12, fontweight='bold')
                    ax_teacher.axis('off')

                    # Draw arrows if enabled
                    if self.show_arrows:
                        self._draw_correction_arrows(
                            student_vis.copy(),
                            student_pose,
                            teacher_pose,
                            position_result
                        )
                        # Could show combined image in a third column or overlay

            row += 1

        plt.tight_layout()

        # Handle output mode
        if output_mode == 'save':
            if output_path is None:
                raise ValueError("output_path required when output_mode='save'")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            return None
        elif output_mode == 'display':
            plt.show()
            return None
        else:  # 'return'
            return fig

    def _load_image(self, pose_data: dict[str, Any]) -> np.ndarray | None:
        """Load image from pose data metadata."""
        metadata = pose_data.get('metadata', {})
        image_path = metadata.get('image_path')

        if image_path is None:
            return None

        img = cv2.imread(str(image_path))
        if img is None:
            return None

        return img

    def _draw_keypoints_with_colors(
        self,
        image: np.ndarray,
        pose_data: dict[str, Any],
        position_result: dict[str, Any],
        position_name: str,
        label: str = 'Student'
    ) -> np.ndarray:
        """
        Draw keypoints with color coding based on scores.

        Args:
            image: Image to draw on
            pose_data: Pose data dictionary
            position_result: Comparison results for this position
            position_name: Position name
            label: Label for this pose ('Student' or 'Teacher')

        Returns:
            Image with keypoints drawn
        """
        keypoints_2d = pose_data.get('pred_keypoints_2d')
        if keypoints_2d is None:
            return image

        # Map metrics to keypoints (simplified - would need full mapping)
        # For now, we'll color based on overall position score
        overall_score = position_result.get(f'{position_name}_score', 0.0)

        # Determine color based on score
        if self.show_colors:
            if overall_score >= self.color_thresholds['good']:
                base_color = self.colors['good']
            elif overall_score >= self.color_thresholds['fair']:
                base_color = self.colors['fair']
            else:
                base_color = self.colors['poor']
        else:
            base_color = self.colors['default']

        # Draw keypoints
        for kpt in keypoints_2d:
            if np.isnan(kpt).any() or (kpt < 0).any():
                continue

            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(image, (x, y), 5, base_color, -1)

        # Draw bounding box if available
        bbox = pose_data.get('bbox')
        if bbox is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(image, (x1, y1), (x2, y2), base_color, 2)

            # Add label
            cv2.putText(
                image,
                f'{label} ({overall_score:.0f})',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                base_color,
                2
            )

        return image

    def _draw_teacher_keypoints(
        self,
        image: np.ndarray,
        pose_data: dict[str, Any],
        label: str = 'Teacher'
    ) -> np.ndarray:
        """Draw teacher keypoints in blue."""
        keypoints_2d = pose_data.get('pred_keypoints_2d')
        if keypoints_2d is None:
            return image

        teacher_color = self.colors['teacher']

        # Draw keypoints
        for kpt in keypoints_2d:
            if np.isnan(kpt).any() or (kpt < 0).any():
                continue

            x, y = int(kpt[0]), int(kpt[1])
            cv2.circle(image, (x, y), 5, teacher_color, -1)

        # Draw bounding box
        bbox = pose_data.get('bbox')
        if bbox is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(image, (x1, y1), (x2, y2), teacher_color, 2)

            # Add label
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                teacher_color,
                2
            )

        return image

    def _draw_correction_arrows(
        self,
        image: np.ndarray,
        student_pose: dict[str, Any],
        teacher_pose: dict[str, Any],
        position_result: dict[str, Any]
    ) -> np.ndarray:
        """
        Draw arrows showing needed corrections.

        Args:
            image: Image to draw on
            student_pose: Student pose data
            teacher_pose: Teacher pose data
            position_result: Comparison results

        Returns:
            Image with arrows drawn
        """
        if not self.show_arrows:
            return image

        student_kpts_2d = student_pose.get('pred_keypoints_2d')
        teacher_kpts_2d = teacher_pose.get('pred_keypoints_2d')

        if student_kpts_2d is None or teacher_kpts_2d is None:
            return image

        # Draw arrows for key keypoints that need correction
        # Focus on keypoints relevant to metrics with low scores
        key_keypoint_indices = [
            PICKLEBALL_KEYPOINTS.get('right_shoulder'),
            PICKLEBALL_KEYPOINTS.get('right_elbow'),
            PICKLEBALL_KEYPOINTS.get('right_wrist'),
            PICKLEBALL_KEYPOINTS.get('left_shoulder'),
            PICKLEBALL_KEYPOINTS.get('left_elbow'),
            PICKLEBALL_KEYPOINTS.get('left_wrist'),
            PICKLEBALL_KEYPOINTS.get('right_hip'),
            PICKLEBALL_KEYPOINTS.get('left_hip'),
            PICKLEBALL_KEYPOINTS.get('right_knee'),
            PICKLEBALL_KEYPOINTS.get('left_knee'),
        ]

        arrow_color = (255, 255, 0)  # Yellow for arrows

        for kpt_idx in key_keypoint_indices:
            if kpt_idx is None or kpt_idx >= len(student_kpts_2d):
                continue

            student_kpt = student_kpts_2d[kpt_idx]
            teacher_kpt = teacher_kpts_2d[kpt_idx]

            if np.isnan(student_kpt).any() or np.isnan(teacher_kpt).any():
                continue

            # Calculate direction vector
            direction = teacher_kpt - student_kpt
            distance = np.linalg.norm(direction)

            # Only draw arrow if distance is significant
            if distance < self.arrow_threshold:
                continue

            # Normalize direction
            if distance > 0:
                direction = direction / distance

            # Draw arrow
            start_point = (int(student_kpt[0]), int(student_kpt[1]))
            end_point = (
                int(student_kpt[0] + direction[0] * min(distance, 50)),
                int(student_kpt[1] + direction[1] * min(distance, 50))
            )

            cv2.arrowedLine(
                image,
                start_point,
                end_point,
                arrow_color,
                2,
                tipLength=0.3
            )

        return image
