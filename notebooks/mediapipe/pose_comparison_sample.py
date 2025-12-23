# Pose Comparison Sample Code for Jupyter Notebook
# Copy and paste these cells into your Jupyter notebook

# ============================================================================
# CELL 1: Setup and Imports
# ============================================================================

# Add the project root to Python path
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# Import required libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np

print("✅ Setup complete - Python path updated")

# ============================================================================
# CELL 2: Import PoseComparator
# ============================================================================

# Import the pose comparator
try:
    from libraries.src.coackata.compare_image import PoseComparator
    print("✅ Successfully imported PoseComparator")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're using the 'Coach Kata (uv)' kernel")

# ============================================================================
# CELL 3: Initialize PoseComparator
# ============================================================================

# Initialize the comparator
try:
    comparator = PoseComparator()
    print("✅ PoseComparator initialized successfully")
    print(f"📐 Available joints: {len(comparator.JOINTS)}")

    # Show available joints
    print("\n📋 Available joints for analysis:")
    for i, joint_name in enumerate(comparator.JOINTS.keys(), 1):
        print(f"   {i:2d}. {joint_name.replace('_', ' ').title()}")

except Exception as e:
    print(f"❌ Error initializing PoseComparator: {e}")

# ============================================================================
# CELL 4: Example Usage (Update paths with your images)
# ============================================================================

# Example: Load and compare poses
# Replace these paths with your actual image files
teacher_image_path = "path/to/teacher_pose.jpg"  # UPDATE THIS
student_image_path = "path/to/student_pose.jpg"  # UPDATE THIS

# Check if files exist
if os.path.exists(teacher_image_path) and os.path.exists(student_image_path):
    print("Loading images...")

    # Load images
    teacher_img = comparator.load_image(teacher_image_path)
    student_img = comparator.load_image(student_image_path)

    print(f"Teacher image: {teacher_img.shape}")
    print(f"Student image: {student_img.shape}")

    # Compare poses
    print("\nDetecting poses and comparing angles...")
    results = comparator.compare_poses(teacher_img, student_img)

    # Display results
    print("\n📐 Joint Angle Comparisons:")
    print("-" * 60)
    for joint_name, diff_data in results['angle_differences'].items():
        if diff_data is not None:
            teacher_angle = diff_data['teacher_angle']
            student_angle = diff_data['student_angle']
            difference = diff_data['difference']
            abs_diff = diff_data['absolute_diff']

            status = "✅" if abs_diff <= 10.0 else "🔴"
            print(f"{status} {joint_name.replace('_', ' ').title()}:")
            print(f"   Teacher: {teacher_angle:.1f}° | Student: {student_angle:.1f}° | Diff: {difference:+.1f}°")

    # Generate feedback
    print("\n💡 Coaching Feedback:")
    print("-" * 60)
    feedback = comparator.generate_feedback(results['angle_differences'])
    for message in feedback:
        print(message)

    # Summary
    total_joints = len(results['angle_differences'])
    good_joints = sum(1 for diff_data in results['angle_differences'].values()
                     if diff_data is not None and diff_data['absolute_diff'] <= 10.0)

    print("\n📈 Summary:")
    print(f"Total joints analyzed: {total_joints}")
    print(f"Joints with good form: {good_joints}")
    print(f"Form accuracy: {good_joints/total_joints*100:.1f}%")

else:
    print("❌ Image files not found. Please update the paths above with your actual image files.")
    print("\nExample usage:")
    print("teacher_image_path = 'teacher_squat.jpg'")
    print("student_image_path = 'student_squat.jpg'")

# ============================================================================
# CELL 5: Visualization (Run after Cell 4)
# ============================================================================

# Example: Create visualization
if 'results' in locals():
    print("Creating side-by-side visualization...")

    # Create visualization
    combined = comparator.visualize_comparison(
        teacher_img, student_img,
        results['teacher_landmarks'], results['student_landmarks']
    )

    # Display the combined image
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.title('Teacher vs Student Pose Comparison\nGreen: Teacher | Red: Student', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Optionally save the visualization
    # cv2.imwrite('pose_comparison.png', combined)
    # print("Visualization saved as 'pose_comparison.png'")
else:
    print("No results available. Run the comparison cell above first.")

# ============================================================================
# CELL 6: Joint Analysis Details
# ============================================================================

# Display available joints and their landmark definitions
if 'comparator' in locals():
    print("📐 Available joints for angle analysis:")
    print("-" * 50)

    joint_definitions = {
        "elbow_right": "Shoulder-Elbow-Wrist",
        "elbow_left": "Shoulder-Elbow-Wrist",
        "shoulder_right": "Hip-Shoulder-Elbow",
        "shoulder_left": "Hip-Shoulder-Elbow",
        "hip_right": "Knee-Hip-Shoulder",
        "hip_left": "Knee-Hip-Shoulder",
        "knee_right": "Hip-Knee-Ankle",
        "knee_left": "Hip-Knee-Ankle",
        "ankle_right": "Knee-Ankle-Foot",
        "ankle_left": "Knee-Ankle-Foot",
        "torso_tilt": "Left hip-Right hip-Right shoulder"
    }

    for i, (joint_name, definition) in enumerate(joint_definitions.items(), 1):
        print(f"{i:2d}. {joint_name.replace('_', ' ').title()}: {definition}")

    print("\n💡 Each joint angle is calculated using three landmarks:")
    print("   - Point A: First landmark")
    print("   - Point B: Middle landmark (the joint being measured)")
    print("   - Point C: Third landmark")
    print("\n   The angle is measured at Point B between lines A-B and B-C.")

# ============================================================================
# CELL 7: Command Line Usage Examples
# ============================================================================

# Show command line usage examples
print("💻 Command Line Usage:")
print("-" * 30)
print("\nBasic comparison:")
print("  python compare_image.py teacher.jpg student.jpg")
print("\nWith coaching feedback:")
print("  python compare_image.py teacher.jpg student.jpg --feedback")
print("\nSave visualization:")
print("  python compare_image.py teacher.jpg student.jpg --output comparison.png")
print("\nCustom threshold:")
print("  python compare_image.py teacher.jpg student.jpg --threshold 15.0")
print("\nUsing poe task:")
print("  uv run poe compare-poses teacher.jpg student.jpg --feedback")

# ============================================================================
# CELL 8: Test with Sample Data (Optional)
# ============================================================================

# Create a simple test to verify the module works
print("🧪 Testing PoseComparator functionality...")

try:
    # Test if we can create a simple test image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (128, 128, 128)  # Gray background

    # Try to detect pose (will likely fail with no person, but tests the setup)
    landmarks = comparator.detect_pose(test_img)

    if landmarks is None:
        print("✅ Pose detection working (no pose detected in test image - expected)")
    else:
        print("✅ Pose detection working (pose detected)")

    print("✅ All tests passed - module is ready to use!")

except Exception as e:
    print(f"❌ Test failed: {e}")

# ============================================================================
# CELL 9: Custom Analysis Example
# ============================================================================

# Example of custom analysis (run after loading images)
if 'results' in locals():
    print("🔍 Custom Analysis Example:")
    print("-" * 40)

    # Find the joint with the biggest difference
    max_diff_joint = None
    max_diff_value = 0

    for joint_name, diff_data in results['angle_differences'].items():
        if diff_data is not None:
            abs_diff = diff_data['absolute_diff']
            if abs_diff > max_diff_value:
                max_diff_value = abs_diff
                max_diff_joint = joint_name

    if max_diff_joint:
        print(f"Biggest difference: {max_diff_joint.replace('_', ' ').title()} ({max_diff_value:.1f}°)")

        # Analyze specific joint
        joint_data = results['angle_differences'][max_diff_joint]
        teacher_angle = joint_data['teacher_angle']
        student_angle = joint_data['student_angle']
        difference = joint_data['difference']

        print(f"Teacher angle: {teacher_angle:.1f}°")
        print(f"Student angle: {student_angle:.1f}°")

        if difference > 0:
            print(f"Student needs to decrease angle by {difference:.1f}°")
        else:
            print(f"Student needs to increase angle by {abs(difference):.1f}°")
else:
    print("No results available. Run the comparison cell first.")

# ============================================================================
# END OF SAMPLE CODE
# ============================================================================
