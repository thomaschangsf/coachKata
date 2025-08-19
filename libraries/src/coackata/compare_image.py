#!/usr/bin/env python3
"""
Pose comparison module for Coach Kata using MediaPipe.

This module compares student and teacher poses by analyzing joint angles
and providing feedback for coaching purposes.

Can be used as both a module and a standalone script.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
import math

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PoseComparator:
    """Class for comparing poses using MediaPipe pose detection."""
    
    def __init__(self):
        """Initialize the PoseComparator with MediaPipe pose detection."""
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for pose comparison")
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required for pose comparison")
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Define key joints for angle analysis
        self.JOINTS = {
            "elbow_right": (
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                self.mp_pose.PoseLandmark.RIGHT_WRIST.value
            ),
            "elbow_left": (
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
                self.mp_pose.PoseLandmark.LEFT_WRIST.value
            ),
            "shoulder_right": (
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
            ),
            "shoulder_left": (
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_ELBOW.value
            ),
            "hip_right": (
                self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            ),
            "hip_left": (
                self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            ),
            "knee_right": (
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
            ),
            "knee_left": (
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                self.mp_pose.PoseLandmark.LEFT_ANKLE.value
            ),
            "torso_tilt": (
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            ),
            "ankle_right": (
                self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
            ),
            "ankle_left": (
                self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
            )
        }
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        return img
    
    def detect_pose(self, image: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        Detect pose landmarks in an image.
        
        Args:
            image: Input image
            
        Returns:
            List of landmark coordinates (x, y) or None if no pose detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks is None:
            return None
        
        # Extract landmark coordinates
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y))
        
        return landmarks
    
    def calculate_angle(self, point_a: Tuple[float, float], 
                       point_b: Tuple[float, float], 
                       point_c: Tuple[float, float]) -> float:
        """
        Calculate the angle between three points.
        
        Args:
            point_a: First point (x, y)
            point_b: Middle point (x, y)
            point_c: Third point (x, y)
            
        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        a = np.array(point_a)
        b = np.array(point_b)
        c = np.array(point_c)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate cosine of angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Clamp to valid range
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle = np.arccos(cosine_angle) * 180.0 / np.pi
        
        return angle
    
    def compare_poses(self, teacher_img: np.ndarray, student_img: np.ndarray) -> Dict[str, Any]:
        """
        Compare teacher and student poses.
        
        Args:
            teacher_img: Teacher's pose image
            student_img: Student's pose image
            
        Returns:
            Dictionary containing comparison results
        """
        # Detect poses
        teacher_landmarks = self.detect_pose(teacher_img)
        student_landmarks = self.detect_pose(student_img)
        
        if teacher_landmarks is None:
            raise ValueError("No pose detected in teacher image")
        if student_landmarks is None:
            raise ValueError("No pose detected in student image")
        
        # Compute angles
        angle_diffs = {}
        for name, (a_idx, b_idx, c_idx) in self.JOINTS.items():
            try:
                angle_teacher = self.calculate_angle(
                    teacher_landmarks[a_idx], 
                    teacher_landmarks[b_idx], 
                    teacher_landmarks[c_idx]
                )
                angle_student = self.calculate_angle(
                    student_landmarks[a_idx], 
                    student_landmarks[b_idx], 
                    student_landmarks[c_idx]
                )
                angle_diffs[name] = {
                    'teacher_angle': angle_teacher,
                    'student_angle': angle_student,
                    'difference': angle_student - angle_teacher,
                    'absolute_diff': abs(angle_student - angle_teacher)
                }
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not calculate angle for {name}: {e}")
                angle_diffs[name] = None
        
        return {
            'teacher_landmarks': teacher_landmarks,
            'student_landmarks': student_landmarks,
            'angle_differences': angle_diffs
        }
    
    def draw_pose_landmarks(self, image: np.ndarray, landmarks: List[Tuple[float, float]], 
                          connections: List[Tuple[int, int]], color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw pose landmarks and connections on an image.
        
        Args:
            image: Input image
            landmarks: List of landmark coordinates
            connections: List of landmark connections
            color: Color for drawing (BGR)
            
        Returns:
            Image with pose landmarks drawn
        """
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # Draw connections
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx][0] * w), int(landmarks[start_idx][1] * h))
                end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))
                
                cv2.line(img_copy, start_point, end_point, color, 2)
        
        # Draw landmarks
        for landmark in landmarks:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            cv2.circle(img_copy, (x, y), 5, color, -1)
        
        return img_copy
    
    def generate_feedback(self, angle_diffs: Dict[str, Any]) -> List[str]:
        """
        Generate coaching feedback based on angle differences.
        
        Args:
            angle_diffs: Dictionary of angle differences
            
        Returns:
            List of feedback messages
        """
        feedback = []
        threshold = 10.0  # degrees
        
        for joint_name, diff_data in angle_diffs.items():
            if diff_data is None:
                continue
                
            abs_diff = diff_data['absolute_diff']
            diff = diff_data['difference']
            
            if abs_diff > threshold:
                if diff > 0:
                    feedback.append(f"🔴 {joint_name.replace('_', ' ').title()}: Increase angle by {abs_diff:.1f}°")
                else:
                    feedback.append(f"🔴 {joint_name.replace('_', ' ').title()}: Decrease angle by {abs_diff:.1f}°")
            else:
                feedback.append(f"✅ {joint_name.replace('_', ' ').title()}: Good form (±{abs_diff:.1f}°)")
        
        return feedback
    
    def visualize_comparison(self, teacher_img: np.ndarray, student_img: np.ndarray,
                           teacher_landmarks: List[Tuple[float, float]], 
                           student_landmarks: List[Tuple[float, float]],
                           output_path: Optional[str] = None) -> np.ndarray:
        """
        Create a side-by-side visualization of teacher and student poses.
        
        Args:
            teacher_img: Teacher's image
            student_img: Student's image
            teacher_landmarks: Teacher's pose landmarks
            student_landmarks: Student's pose landmarks
            output_path: Optional path to save the visualization
            
        Returns:
            Combined visualization image
        """
        # Draw pose landmarks
        teacher_vis = self.draw_pose_landmarks(
            teacher_img, teacher_landmarks, 
            self.mp_pose.POSE_CONNECTIONS, color=(0, 255, 0)  # Green for teacher
        )
        student_vis = self.draw_pose_landmarks(
            student_img, student_landmarks, 
            self.mp_pose.POSE_CONNECTIONS, color=(0, 0, 255)  # Red for student
        )
        
        # Resize images to same height for side-by-side comparison
        h1, w1 = teacher_vis.shape[:2]
        h2, w2 = student_vis.shape[:2]
        
        target_height = min(h1, h2)
        scale1 = target_height / h1
        scale2 = target_height / h2
        
        new_w1 = int(w1 * scale1)
        new_w2 = int(w2 * scale2)
        
        teacher_resized = cv2.resize(teacher_vis, (new_w1, target_height))
        student_resized = cv2.resize(student_vis, (new_w2, target_height))
        
        # Combine images side by side
        combined = np.hstack((teacher_resized, student_resized))
        
        # Add labels
        cv2.putText(combined, "Teacher (Green)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Student (Red)", (new_w1 + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, combined)
            print(f"Visualization saved to: {output_path}")
        
        return combined


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Compare teacher and student poses using MediaPipe')
    parser.add_argument('teacher_image', help='Path to teacher/expert image')
    parser.add_argument('student_image', help='Path to student image')
    parser.add_argument('--output', '-o', help='Save comparison visualization to file')
    parser.add_argument('--feedback', '-f', action='store_true', 
                       help='Generate coaching feedback')
    parser.add_argument('--threshold', '-t', type=float, default=10.0,
                       help='Angle difference threshold for feedback (degrees)')
    
    args = parser.parse_args()
    
    try:
        # Initialize comparator
        comparator = PoseComparator()
        
        # Load images
        print(f"Loading images...")
        teacher_img = comparator.load_image(args.teacher_image)
        student_img = comparator.load_image(args.student_image)
        print(f"Teacher image: {teacher_img.shape}")
        print(f"Student image: {student_img.shape}")
        
        # Compare poses
        print("Detecting poses and comparing angles...")
        results = comparator.compare_poses(teacher_img, student_img)
        
        # Print angle comparisons
        print("\n📐 Joint Angle Comparisons:")
        print("-" * 60)
        for joint_name, diff_data in results['angle_differences'].items():
            if diff_data is not None:
                teacher_angle = diff_data['teacher_angle']
                student_angle = diff_data['student_angle']
                difference = diff_data['difference']
                abs_diff = diff_data['absolute_diff']
                
                status = "✅" if abs_diff <= args.threshold else "🔴"
                print(f"{status} {joint_name.replace('_', ' ').title():}")
                print(f"   Teacher: {teacher_angle:.1f}° | Student: {student_angle:.1f}° | Diff: {difference:+.1f}°")
        
        # Generate feedback
        if args.feedback:
            print("\n💡 Coaching Feedback:")
            print("-" * 60)
            feedback = comparator.generate_feedback(results['angle_differences'])
            for message in feedback:
                print(message)
        
        # Create visualization
        if args.output:
            print(f"\n📊 Creating visualization...")
            combined = comparator.visualize_comparison(
                teacher_img, student_img,
                results['teacher_landmarks'], results['student_landmarks'],
                args.output
            )
        
        # Summary
        total_joints = len(results['angle_differences'])
        good_joints = sum(1 for diff_data in results['angle_differences'].values() 
                         if diff_data is not None and diff_data['absolute_diff'] <= args.threshold)
        
        print(f"\n📈 Summary:")
        print(f"Total joints analyzed: {total_joints}")
        print(f"Joints with good form: {good_joints}")
        print(f"Form accuracy: {good_joints/total_joints*100:.1f}%")
            
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required dependencies:")
        print("  pip install opencv-python mediapipe matplotlib")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 