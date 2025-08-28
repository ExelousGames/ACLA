"""
Tests for telemetry service
"""

import pytest
import pandas as pd
from app.services.telemetry_service import TelemetryService


class TestTelemetryService:
    """Test cases for telemetry service"""
    
    def setup_method(self):
        """Setup test instance"""
        self.service = TelemetryService()
    
    def test_validate_telemetry_data(self, sample_telemetry_data):
        """Test telemetry data validation"""
        result = self.service.validate_telemetry_data(sample_telemetry_data)
        
        assert "validation_results" in result
        assert "data_quality" in result
        assert "recommendations" in result
    
    def test_get_available_features(self):
        """Test getting available telemetry features"""
        features = self.service.get_available_features()
        
        assert "physics" in features
        assert "graphics" in features
        assert "static" in features
        assert isinstance(features["physics"], list)
    
    def test_analyze_telemetry_session(self, sample_telemetry_data):
        """Test telemetry session analysis"""
        result = self.service.analyze_telemetry_session(
            sample_telemetry_data, 
            analysis_type="performance"
        )
        
        # Should not have error
        assert "error" not in result or result.get("error") is None
