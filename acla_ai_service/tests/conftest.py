"""
Test configuration and utilities
"""

import pytest
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

@pytest.fixture
def sample_telemetry_data():
    """Sample telemetry data for testing"""
    return {
        "Physics_speed_kmh": [120.5, 125.3, 130.1, 128.7],
        "Physics_rpm": [7500, 7800, 8200, 8100],
        "Physics_gear": [3, 3, 4, 4],
        "Physics_throttle": [0.8, 0.9, 1.0, 0.95]
    }

@pytest.fixture
def sample_analysis_request():
    """Sample analysis request for testing"""
    return {
        "dataset_id": "test_dataset_123",
        "analysis_type": "comprehensive",
        "parameters": {}
    }
