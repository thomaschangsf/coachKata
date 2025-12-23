#!/usr/bin/env python3
"""
Test script for the compare_image module (now PoseComparator).
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from coackata.compare_image import PoseComparator
    print("✅ Successfully imported PoseComparator")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_pose_comparator():
    """Test the PoseComparator functionality."""
    try:
        # Initialize comparator
        comparator = PoseComparator()
        print("✅ PoseComparator initialized successfully")

        # Show available joints
        print(f"\n📐 Available joints for analysis ({len(comparator.JOINTS)}):")
        for joint_name in comparator.JOINTS.keys():
            print(f"   - {joint_name.replace('_', ' ').title()}")

        # Test with sample images (you'll need to provide actual images)
        print("\n📝 Usage examples:")
        print("1. As a module:")
        print("   from coackata.compare_image import PoseComparator")
        print("   comparator = PoseComparator()")
        print("   teacher_img = comparator.load_image('teacher.jpg')")
        print("   student_img = comparator.load_image('student.jpg')")
        print("   results = comparator.compare_poses(teacher_img, student_img)")
        print("   feedback = comparator.generate_feedback(results['angle_differences'])")

        print("\n2. As a script:")
        print("   python compare_image.py teacher.jpg student.jpg")
        print("   python compare_image.py teacher.jpg student.jpg --feedback")
        print("   python compare_image.py teacher.jpg student.jpg --output comparison.png")
        print("   python compare_image.py teacher.jpg student.jpg --threshold 15.0")

        print("\n3. Available joints for angle analysis:")
        print("   - Elbow (right/left): Shoulder-Elbow-Wrist")
        print("   - Shoulder (right/left): Hip-Shoulder-Elbow")
        print("   - Hip (right/left): Knee-Hip-Shoulder")
        print("   - Knee (right/left): Hip-Knee-Ankle")
        print("   - Ankle (right/left): Knee-Ankle-Foot")
        print("   - Torso tilt: Left hip-Right hip-Right shoulder")

        print("\n4. Output includes:")
        print("   - Angle comparisons for each joint")
        print("   - Coaching feedback with specific recommendations")
        print("   - Side-by-side visualization with pose landmarks")
        print("   - Form accuracy percentage")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_pose_comparator()
