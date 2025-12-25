"""Tests for config_loader module."""

import json
import tempfile
from pathlib import Path

import pytest
from pickleball_pose_analyzer.comparison.config_loader import (
    get_metrics_for_stroke,
    get_percentile_settings,
    get_scoring_method,
    get_single_pose_tolerance,
    get_visualization_settings,
    get_weights_for_stroke,
    load_comparison_config,
    validate_config,
)


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        'stroke_types': {
            'serve': {
                'preparation': {
                    'metrics': ['shoulder_angle', 'weight_distribution', 'knee_bend'],
                    'weights': {
                        'shoulder_angle': 0.4,
                        'weight_distribution': 0.35,
                        'knee_bend': 0.25,
                    }
                },
                'contact': {
                    'metrics': ['paddle_position', 'contact_height'],
                    'weights': {
                        'paddle_position': 0.6,
                        'contact_height': 0.4,
                    }
                },
                'finish': {
                    'metrics': ['follow_through_angle'],
                    'weights': {
                        'follow_through_angle': 1.0,
                    }
                }
            }
        },
        'scoring_methods': {
            'shoulder_angle': {
                'type': 'tolerance',
                'tolerance': 0.2
            },
            'paddle_position': {
                'type': 'distance',
                'max_distance': 0.1
            }
        },
        'visualization': {
            'show_colors': True,
            'show_arrows': True,
            'color_thresholds': {
                'good': 80,
                'fair': 50,
                'poor': 0
            }
        },
        'single_pose_tolerance': {
            'percentage': 0.10
        },
        'percentile_settings': {
            'low': 25.0,
            'high': 75.0
        }
    }


@pytest.fixture
def config_file(sample_config):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config, f)
        config_path = f.name

    yield config_path

    # Cleanup
    Path(config_path).unlink(missing_ok=True)


def test_load_comparison_config(config_file, sample_config):
    """Test loading configuration from file."""
    config = load_comparison_config(config_file)

    assert 'stroke_types' in config
    assert 'serve' in config['stroke_types']
    assert config['stroke_types']['serve']['preparation']['metrics'] == \
        sample_config['stroke_types']['serve']['preparation']['metrics']


def test_load_comparison_config_file_not_found():
    """Test loading non-existent config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_comparison_config('/nonexistent/path/config.json')


def test_validate_config_valid(sample_config):
    """Test validation of valid configuration."""
    # Should not raise
    validate_config(sample_config)


def test_validate_config_missing_stroke_types():
    """Test validation fails when stroke_types is missing."""
    config = {}
    with pytest.raises(ValueError, match="Missing required configuration key: stroke_types"):
        validate_config(config)


def test_validate_config_invalid_stroke_types():
    """Test validation fails when stroke_types is not a dict."""
    config = {'stroke_types': 'not a dict'}
    with pytest.raises(ValueError, match="'stroke_types' must be a dictionary"):
        validate_config(config)


def test_validate_config_weights_sum_not_one():
    """Test validation fails when weights don't sum to 1.0."""
    config = {
        'stroke_types': {
            'serve': {
                'preparation': {
                    'metrics': ['shoulder_angle', 'weight_distribution'],
                    'weights': {
                        'shoulder_angle': 0.3,
                        'weight_distribution': 0.3,  # Sum = 0.6, not 1.0
                    }
                }
            }
        }
    }
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_config(config)


def test_get_metrics_for_stroke(sample_config):
    """Test getting metrics for a stroke type and position."""
    metrics = get_metrics_for_stroke(sample_config, 'serve', 'preparation')

    assert isinstance(metrics, list)
    assert 'shoulder_angle' in metrics
    assert 'weight_distribution' in metrics
    assert 'knee_bend' in metrics


def test_get_metrics_for_stroke_not_found(sample_config):
    """Test getting metrics for non-existent stroke type."""
    with pytest.raises(ValueError, match="Stroke type 'forehand' not found"):
        get_metrics_for_stroke(sample_config, 'forehand', 'preparation')


def test_get_weights_for_stroke(sample_config):
    """Test getting weights for a stroke type and position."""
    weights = get_weights_for_stroke(sample_config, 'serve', 'preparation')

    assert isinstance(weights, dict)
    assert 'shoulder_angle' in weights
    assert weights['shoulder_angle'] == 0.4
    assert abs(sum(weights.values()) - 1.0) < 0.01


def test_get_weights_for_stroke_equal_weights():
    """Test getting equal weights when not specified."""
    config = {
        'stroke_types': {
            'serve': {
                'preparation': {
                    'metrics': ['shoulder_angle', 'weight_distribution'],
                    # No weights specified
                }
            }
        }
    }
    weights = get_weights_for_stroke(config, 'serve', 'preparation')

    assert len(weights) == 2
    assert all(w == 0.5 for w in weights.values())


def test_get_scoring_method(sample_config):
    """Test getting scoring method configuration."""
    method = get_scoring_method(sample_config, 'shoulder_angle')

    assert method['type'] == 'tolerance'
    assert method['tolerance'] == 0.2


def test_get_scoring_method_default():
    """Test getting default scoring method when not specified."""
    config = {'stroke_types': {}}
    method = get_scoring_method(config, 'unknown_metric')

    assert method['type'] == 'tolerance'
    assert 'tolerance' in method


def test_get_visualization_settings(sample_config):
    """Test getting visualization settings."""
    settings = get_visualization_settings(sample_config)

    assert settings['show_colors'] is True
    assert settings['show_arrows'] is True
    assert settings['color_thresholds']['good'] == 80


def test_get_visualization_settings_defaults():
    """Test getting default visualization settings."""
    config = {'stroke_types': {}}
    settings = get_visualization_settings(config)

    assert settings['show_colors'] is True  # Default
    assert settings['show_arrows'] is True  # Default
    assert 'color_thresholds' in settings


def test_get_single_pose_tolerance(sample_config):
    """Test getting single pose tolerance."""
    tolerance = get_single_pose_tolerance(sample_config)

    assert tolerance == 0.10


def test_get_single_pose_tolerance_default():
    """Test getting default single pose tolerance."""
    config = {'stroke_types': {}}
    tolerance = get_single_pose_tolerance(config)

    assert tolerance == 0.10  # Default


def test_get_percentile_settings(sample_config):
    """Test getting percentile settings."""
    settings = get_percentile_settings(sample_config)

    assert settings['low'] == 25.0
    assert settings['high'] == 75.0


def test_get_percentile_settings_default():
    """Test getting default percentile settings."""
    config = {'stroke_types': {}}
    settings = get_percentile_settings(config)

    assert settings['low'] == 25.0  # Default
    assert settings['high'] == 75.0  # Default
