"""Tests for feedback_generator module."""

import pytest
from pickleball_pose_analyzer.comparison.feedback_generator import (
    format_feedback_text,
    generate_correction,
    generate_feedback,
    generate_overall_feedback,
)


@pytest.fixture
def sample_comparison_results():
    """Sample comparison results for testing."""
    return {
        'preparation': {
            'scores': {
                'shoulder_angle': 65.0,
                'weight_distribution': 75.0,
                'knee_bend': 80.0,
            },
            'metrics': {
                'shoulder_angle': 85.0,
                'weight_distribution': 0.45,
                'knee_bend': 25.0,
            },
            'differences': {
                'shoulder_angle': -10.0,
                'weight_distribution': -0.05,
                'knee_bend': 5.0,
            },
            'preparation_score': 73.3,
        },
        'contact': {
            'scores': {
                'paddle_position': 90.0,
                'contact_height': 85.0,
            },
            'metrics': {
                'paddle_position': 0.5,
                'contact_height': 1.1,
            },
            'differences': {
                'paddle_position': 0.0,
                'contact_height': 0.1,
            },
            'contact_score': 87.5,
        },
        'finish': {
            'scores': {
                'follow_through_angle': 70.0,
            },
            'metrics': {
                'follow_through_angle': 45.0,
            },
            'differences': {
                'follow_through_angle': -5.0,
            },
            'finish_score': 70.0,
        },
        'cumulative_score': 76.9,
    }


def test_generate_feedback_basic(sample_comparison_results):
    """Test basic feedback generation."""
    feedback = generate_feedback(sample_comparison_results, stroke_type='serve', top_n=3)

    assert 'prioritized_issues' in feedback
    assert 'overall_feedback' in feedback
    assert 'strengths' in feedback
    assert 'improvements' in feedback


def test_generate_feedback_prioritized_issues(sample_comparison_results):
    """Test that prioritized issues are sorted correctly."""
    feedback = generate_feedback(sample_comparison_results, stroke_type='serve', top_n=3)

    issues = feedback['prioritized_issues']
    assert len(issues) <= 3

    # Check priorities are numbered correctly
    for i, issue in enumerate(issues, 1):
        assert issue['priority'] == i

    # Check issues are sorted by priority (highest first)
    if len(issues) > 1:
        priorities = [issue['priority'] for issue in issues]
        assert priorities == sorted(priorities)


def test_generate_feedback_strengths(sample_comparison_results):
    """Test that strengths are identified correctly."""
    feedback = generate_feedback(sample_comparison_results, stroke_type='serve', top_n=3)

    strengths = feedback['strengths']
    assert isinstance(strengths, list)

    # All strengths should have score >= 80
    for strength in strengths:
        assert strength['score'] >= 80
        assert 'position' in strength
        assert 'metric' in strength


def test_format_feedback_text_good_score():
    """Test formatting feedback text for good scores."""
    text = format_feedback_text('shoulder_angle', 95.0, 0.0, 85.0)

    assert 'good' in text.lower() or '95' in text


def test_format_feedback_text_poor_score():
    """Test formatting feedback text for poor scores."""
    text = format_feedback_text('shoulder_angle', 75.0, -20.0, 30.0)

    assert 'improvement' in text.lower() or 'too low' in text.lower() or 'too high' in text.lower()


def test_format_feedback_text_angle_metric():
    """Test formatting feedback text for angle metrics."""
    text = format_feedback_text('shoulder_angle', 85.0, -10.0, 60.0)

    assert '°' in text  # Should include degree symbol


def test_format_feedback_text_height_metric():
    """Test formatting feedback text for height metrics."""
    text = format_feedback_text('contact_height', 1.0, 0.1, 70.0)

    assert 'm' in text  # Should include meter unit


def test_generate_correction_shoulder_angle():
    """Test generating correction for shoulder angle."""
    correction = generate_correction('shoulder_angle', -10.0, 'preparation')

    assert isinstance(correction, str)
    assert len(correction) > 0
    assert 'shoulder' in correction.lower()


def test_generate_correction_weight_distribution():
    """Test generating correction for weight distribution."""
    correction = generate_correction('weight_distribution', -0.05, 'preparation')

    assert isinstance(correction, str)
    assert 'weight' in correction.lower() or 'leg' in correction.lower()


def test_generate_correction_unknown_metric():
    """Test generating correction for unknown metric."""
    correction = generate_correction('unknown_metric', 5.0, 'preparation')

    assert isinstance(correction, str)
    assert 'unknown_metric' in correction.lower() or 'increase' in correction.lower() or 'decrease' in correction.lower()


def test_generate_overall_feedback(sample_comparison_results):
    """Test generating overall feedback."""
    prioritized_issues = [
        {
            'position': 'preparation',
            'metric': 'shoulder_angle',
            'priority': 1,
        },
        {
            'position': 'contact',
            'metric': 'contact_height',
            'priority': 2,
        }
    ]

    feedback = generate_overall_feedback(prioritized_issues, sample_comparison_results)

    assert isinstance(feedback, str)
    assert len(feedback) > 0


def test_generate_overall_feedback_no_issues():
    """Test generating overall feedback when there are no issues."""
    feedback = generate_overall_feedback([], {'cumulative_score': 85.0})

    assert isinstance(feedback, str)
    assert 'excellent' in feedback.lower() or 'strong' in feedback.lower() or 'good' in feedback.lower()
