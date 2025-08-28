"""
Telemetry data processing and analysis service
"""

from typing import Dict, Any, List
import pandas as pd
from app.models.telemetry_models import TelemetryFeatures, FeatureProcessor
from app.analyzers import AdvancedRacingAnalyzer


class TelemetryService:
    """Service for telemetry data processing and analysis"""
    
    def __init__(self):
        self.telemetry_features = TelemetryFeatures()
    
    def validate_telemetry_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate telemetry data quality and completeness"""
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame(data)
            feature_processor = FeatureProcessor(df)
            
            return {
                "validation_results": feature_processor.validate_features(),
                "data_quality": self._assess_data_quality(df),
                "recommendations": self._generate_recommendations(df)
            }
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}
    
    def analyze_telemetry_session(self, data: Dict[str, Any], analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze telemetry data for a racing session"""
        try:
            # Convert to DataFrame for processing
            df = pd.DataFrame(data)
            analyzer = AdvancedRacingAnalyzer(df)
            
            if analysis_type == "comprehensive":
                return analyzer.get_telemetry_summary()
            elif analysis_type == "performance":
                return analyzer.feature_processor.extract_performance_metrics()
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}
                
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def get_available_features(self) -> Dict[str, List[str]]:
        """Get all available telemetry features by category"""
        return {
            "physics": self.telemetry_features.PHYSICS_FEATURES,
            "graphics": self.telemetry_features.GRAPHICS_FEATURES,
            "static": self.telemetry_features.STATIC_FEATURES
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of telemetry data"""
        return {
            "total_records": len(df),
            "missing_data_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns)
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations for data improvement"""
        recommendations = []
        
        # Check for missing data
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 5:
            recommendations.append(f"High missing data percentage ({missing_percentage:.1f}%) - consider data cleaning")
        
        # Check data size
        if len(df) < 100:
            recommendations.append("Small dataset size - consider collecting more data for better analysis")
        
        return recommendations
