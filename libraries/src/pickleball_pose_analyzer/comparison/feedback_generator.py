"""
Feedback generation module for Phase 2 comparison system.

This module generates prioritized, actionable feedback to help users improve
their pickleball stroke technique.
"""

from typing import Any


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
                    'difference': -10.0,
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
    # Collect all issues across all positions
    all_issues = []

    for position in ['preparation', 'contact', 'finish']:
        if position not in comparison_results:
            continue

        position_result = comparison_results[position]
        scores = position_result.get('scores', {})
        metrics = position_result.get('metrics', {})
        differences = position_result.get('differences', {})

        for metric_key, score in scores.items():
                if score >= 80:
                    continue  # Skip good scores

                student_value = metrics.get(metric_key)
                difference = differences.get(metric_key)

                if student_value is None or difference is None:
                    continue

                # Calculate priority based on score impact and position importance
                # Lower score = higher priority
                priority_score = 100 - score

                # Position weights (contact is most important)
                position_weights = {
                    'preparation': 0.3,
                    'contact': 0.5,
                    'finish': 0.2,
                }
                position_weight = position_weights.get(position, 0.33)

                # Combined priority
                priority = priority_score * position_weight

                # Generate feedback text
                feedback_text = format_feedback_text(
                    metric_name=metric_key,
                    student_value=student_value,
                    difference=difference,
                    score=score
                )

                # Generate correction suggestion
                correction = generate_correction(
                    metric_key=metric_key,
                    difference=difference,
                    position=position
                )

                all_issues.append({
                    'position': position,
                    'metric': metric_key,
                    'priority': priority,
                    'score': score,
                    'student_value': student_value,
                    'difference': difference,
                    'feedback_text': feedback_text,
                    'correction': correction,
                })

    # Sort by priority (highest first)
    all_issues.sort(key=lambda x: x['priority'], reverse=True)

    # Get top N issues
    prioritized_issues = all_issues[:top_n]

    # Re-number priorities
    for i, issue in enumerate(prioritized_issues, 1):
        issue['priority'] = i

    # Generate overall feedback
    overall_feedback = generate_overall_feedback(prioritized_issues, comparison_results)

    # Identify strengths (scores >= 80)
    strengths = []
    for position in ['preparation', 'contact', 'finish']:
        if position not in comparison_results:
            continue
        position_result = comparison_results[position]
        scores = position_result.get('scores', {})
        for metric_key, score in scores.items():
            if score >= 80:
                strengths.append({
                    'position': position,
                    'metric': metric_key,
                    'score': score
                })

    return {
        'prioritized_issues': prioritized_issues,
        'overall_feedback': overall_feedback,
        'strengths': strengths,
        'improvements': prioritized_issues,  # Same as prioritized_issues
    }


def format_feedback_text(
    metric_name: str,
    student_value: float,
    difference: float,
    score: float
) -> str:
    """
    Format human-readable feedback text.

    Args:
        metric_name: Name of metric
        student_value: Student's metric value
        difference: Difference from target (student_value - target_center)
        score: Score (0-100)

    Returns:
        Formatted feedback text
    """
    # Metric name formatting
    metric_display = metric_name.replace('_', ' ').title()

    # Determine direction
    if abs(difference) < 0.01:  # Essentially zero
        direction = "matches"
        action = "maintain"
    elif difference > 0:
        direction = "too high"
        action = "decrease"
    else:
        direction = "too low"
        action = "increase"

    # Format value based on metric type
    if 'angle' in metric_name:
        value_str = f"{abs(difference):.1f}°"
    elif 'height' in metric_name or 'position' in metric_name:
        value_str = f"{abs(difference):.2f}m"
    elif 'distribution' in metric_name or 'alignment' in metric_name or 'rotation' in metric_name:
        value_str = f"{abs(difference):.2f}"
    else:
        value_str = f"{abs(difference):.2f}"

    # Generate feedback text
    if score >= 80:
        feedback = f"{metric_display} is good ({student_value:.1f}{'°' if 'angle' in metric_name else ''})"
    elif score >= 50:
        feedback = (
            f"{metric_display} is {direction} by {value_str}. "
            f"Current value: {student_value:.1f}{'°' if 'angle' in metric_name else ''}. "
            f"Try to {action} it slightly."
        )
    else:
        feedback = (
            f"{metric_display} needs significant improvement. "
            f"It is {direction} by {value_str} "
            f"(current: {student_value:.1f}{'°' if 'angle' in metric_name else ''}). "
            f"Focus on {action}ing this metric."
        )

    return feedback


def generate_correction(
    metric_key: str,
    difference: float,
    position: str
) -> str:
    """
    Generate specific correction suggestion.

    Args:
        metric_key: Name of metric
        difference: Difference from target
        position: Position name

    Returns:
        Correction suggestion text
    """
    # Metric-specific corrections
    corrections_map = {
        'shoulder_angle': {
            'preparation': lambda d: (
                f"Rotate {'left' if d < 0 else 'right'} shoulder "
                f"{abs(d):.1f}° more to open your shoulders"
            ),
            'contact': lambda d: (
                f"Adjust shoulder rotation by {abs(d):.1f}°"
            ),
            'finish': lambda d: (
                f"Complete follow-through with {abs(d):.1f}° more shoulder rotation"
            ),
        },
        'weight_distribution': {
            'preparation': lambda d: (
                f"Shift {'more' if d < 0 else 'less'} weight to your back leg"
            ),
            'contact': lambda d: (
                f"Adjust weight distribution - {'more' if d < 0 else 'less'} on back leg"
            ),
        },
        'knee_bend': {
            'preparation': lambda d: (
                f"{'Bend' if d < 0 else 'Straighten'} your knees {abs(d):.1f}° more"
            ),
        },
        'paddle_position': {
            'contact': lambda d: (
                f"Move paddle {'forward' if d < 0 else 'back'} by {abs(d):.2f}m"
            ),
        },
        'contact_height': {
            'contact': lambda d: (
                f"Adjust paddle height - {'raise' if d < 0 else 'lower'} by {abs(d):.2f}m"
            ),
        },
        'torso_angle': {
            'contact': lambda d: (
                f"{'Lean forward' if d < 0 else 'Straighten up'} by {abs(d):.1f}°"
            ),
        },
        'follow_through_angle': {
            'finish': lambda d: (
                f"Extend follow-through by {abs(d):.1f}° more"
            ),
        },
    }

    # Get correction for this metric and position
    if metric_key in corrections_map:
        position_corrections = corrections_map[metric_key]
        if position in position_corrections:
            return position_corrections[position](difference)

    # Generic correction
    direction = "increase" if difference < 0 else "decrease"
    return f"{direction.capitalize()} {metric_key.replace('_', ' ')} by {abs(difference):.2f}"


def generate_overall_feedback(
    prioritized_issues: list[dict[str, Any]],
    comparison_results: dict[str, Any]
) -> str:
    """
    Generate overall feedback summary.

    Args:
        prioritized_issues: List of prioritized issues
        comparison_results: Full comparison results

    Returns:
        Overall feedback text
    """
    if not prioritized_issues:
        cumulative_score = comparison_results.get('cumulative_score', 0.0)
        if cumulative_score >= 80:
            return "Excellent technique! Your pose matches the teacher's ideal ranges very well."
        elif cumulative_score >= 60:
            return "Good technique overall. Minor adjustments would improve your performance."
        else:
            return "There are several areas for improvement. Focus on the prioritized issues below."

    # Get top issues by position
    top_issues_by_position = {}
    for issue in prioritized_issues[:3]:  # Top 3
        position = issue['position']
        if position not in top_issues_by_position:
            top_issues_by_position[position] = []
        top_issues_by_position[position].append(issue['metric'])

    # Build feedback text
    feedback_parts = []

    cumulative_score = comparison_results.get('cumulative_score', 0.0)
    if cumulative_score >= 80:
        feedback_parts.append("Overall, your technique is strong.")
    elif cumulative_score >= 60:
        feedback_parts.append("Your technique is good, with room for improvement.")
    else:
        feedback_parts.append("Your technique needs improvement in several areas.")

    # Add position-specific guidance
    if 'preparation' in top_issues_by_position:
        metrics = ', '.join(top_issues_by_position['preparation'])
        feedback_parts.append(f"Focus on {metrics} during preparation.")

    if 'contact' in top_issues_by_position:
        metrics = ', '.join(top_issues_by_position['contact'])
        feedback_parts.append(f"At contact, improve {metrics}.")

    if 'finish' in top_issues_by_position:
        metrics = ', '.join(top_issues_by_position['finish'])
        feedback_parts.append(f"During finish, work on {metrics}.")

    return " ".join(feedback_parts)
