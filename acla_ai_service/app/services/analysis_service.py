"""
Data analysis service for racing data processing
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from app.analyzers import AdvancedRacingAnalyzer


class AnalysisService:
    """Service for data analysis and processing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def analyze_dataset(self, df: pd.DataFrame, analysis_type: str = "basic") -> Dict[str, Any]:
        """Analyze dataset with specified analysis type"""
        try:
            if analysis_type == "basic":
                return self._basic_analysis(df)
            elif analysis_type == "advanced":
                return self._advanced_analysis(df)
            elif analysis_type == "racing":
                return self._racing_analysis(df)
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _basic_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic statistical analysis"""
        stats = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "null_counts": {col: int(count) for col, count in df.isnull().sum().to_dict().items()},
            "numeric_stats": {}
        }
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats["numeric_stats"][col] = {
                    "mean": self._safe_float(col_data.mean()),
                    "std": self._safe_float(col_data.std()),
                    "min": self._safe_float(col_data.min()),
                    "max": self._safe_float(col_data.max()),
                    "median": self._safe_float(col_data.median())
                }
        
        return stats
    
    def _advanced_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced analysis with ML techniques"""
        try:
            # Select numeric columns for analysis
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            
            if len(numeric_df.columns) < 2:
                return {"error": "Not enough numeric columns for advanced analysis"}
            
            # PCA analysis
            scaled_data = self.scaler.fit_transform(numeric_df)
            pca = PCA(n_components=min(5, len(numeric_df.columns)))
            pca_result = pca.fit_transform(scaled_data)
            
            return {
                "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
                "pca_components": pca.components_.tolist(),
                "correlation_matrix": numeric_df.corr().to_dict(),
                "feature_importance": dict(zip(numeric_df.columns, pca.explained_variance_ratio_))
            }
        except Exception as e:
            return {"error": f"Advanced analysis failed: {str(e)}"}
    
    def _racing_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform racing-specific analysis"""
        try:
            analyzer = AdvancedRacingAnalyzer(df)
            return analyzer.get_telemetry_summary()
        except Exception as e:
            return {"error": f"Racing analysis failed: {str(e)}"}
    
    def generate_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate insights from analysis results"""
        insights = []
        
        if "numeric_stats" in analysis_results:
            stats = analysis_results["numeric_stats"]
            for col, col_stats in stats.items():
                if col_stats["std"] > col_stats["mean"]:
                    insights.append(f"High variability detected in {col}")
        
        if "pca_explained_variance" in analysis_results:
            variance = analysis_results["pca_explained_variance"]
            if len(variance) > 0 and variance[0] > 0.5:
                insights.append(f"First principal component explains {variance[0]:.1%} of the variance")
        
        return insights
    
    def _safe_float(self, value):
        """Convert value to float, handling NaN and infinity"""
        try:
            float_val = float(value)
            if np.isnan(float_val) or np.isinf(float_val):
                return 0.0
            return float_val
        except (ValueError, TypeError):
            return 0.0
