import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import json
from telemetry_models import TelemetryFeatures, FeatureProcessor

class AdvancedRacingAnalyzer:
    """Advanced racing data analysis with ML capabilities for AC Competizione telemetry"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.scaler = StandardScaler()
        self.feature_processor = FeatureProcessor(dataframe)
        self.telemetry_features = TelemetryFeatures()
        
        # Validate and prepare telemetry data
        self.feature_validation = self.feature_processor.validate_features()
        self.processed_df = self.feature_processor.prepare_for_analysis()
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get comprehensive telemetry data summary"""
        return {
            "feature_validation": self.feature_validation,
            "data_quality": self._assess_data_quality(),
            "performance_overview": self.feature_processor.extract_performance_metrics(),
            "recommendations": self._generate_telemetry_recommendations()
        }
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Assess the quality of telemetry data"""
        quality_metrics = {
            "total_records": len(self.df),
            "data_completeness": {},
            "anomalies": {},
            "consistency_score": 0
        }
        
        # Check completeness for critical features
        critical_features = self.telemetry_features.get_performance_critical_features()
        available_critical = [f for f in critical_features if f in self.df.columns]
        
        for feature in available_critical:
            non_null_count = self.df[feature].notna().sum()
            completeness = (non_null_count / len(self.df)) * 100
            quality_metrics["data_completeness"][feature] = round(completeness, 2)
        
        # Detect anomalies in key metrics
        if 'Physics_speed_kmh' in self.df.columns:
            speed_data = self.df['Physics_speed_kmh']
            q1, q3 = speed_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = speed_data[(speed_data < lower_bound) | (speed_data > upper_bound)]
            quality_metrics["anomalies"]["speed_outliers"] = len(outliers)
        
        # Calculate overall consistency score
        completeness_scores = list(quality_metrics["data_completeness"].values())
        if completeness_scores:
            quality_metrics["consistency_score"] = round(np.mean(completeness_scores), 2)
        
        return quality_metrics
    
    def _generate_telemetry_recommendations(self) -> List[str]:
        """Generate recommendations based on telemetry analysis"""
        recommendations = []
        
        coverage = self.feature_validation["coverage_percentage"]
        
        if coverage < 50:
            recommendations.append("Low telemetry coverage detected. Ensure all telemetry systems are active.")
        elif coverage < 80:
            recommendations.append("Moderate telemetry coverage. Some advanced features may not be available.")
        else:
            recommendations.append("Excellent telemetry coverage. Full analysis capabilities available.")
        
        # Check for performance-critical missing features
        critical_features = self.telemetry_features.get_performance_critical_features()
        missing_critical = [f for f in critical_features if f not in self.df.columns]
        
        if missing_critical:
            recommendations.append(f"Missing critical performance features: {', '.join(missing_critical[:3])}")
        
        return recommendations
    
    def advanced_performance_analysis(self) -> Dict[str, Any]:
        """Perform advanced performance analysis using all available telemetry"""
        analysis = {}
        
        # Speed and acceleration analysis
        if 'Physics_speed_kmh' in self.df.columns:
            analysis['speed_analysis'] = self._analyze_speed_patterns()
        
        # Cornering analysis
        if all(f in self.df.columns for f in ['Physics_g_force_x', 'Physics_g_force_y']):
            analysis['cornering_analysis'] = self._analyze_cornering_performance()
        
        # Braking analysis
        if 'Physics_brake' in self.df.columns:
            analysis['braking_analysis'] = self._analyze_braking_performance()
        
        # Tyre performance analysis
        tyre_temp_features = [col for col in self.df.columns if 'tyre_core_temp' in col]
        if tyre_temp_features:
            analysis['tyre_analysis'] = self._analyze_tyre_performance()
        
        # Setup optimization suggestions
        setup_features = self.telemetry_features.get_setup_features()
        available_setup = [f for f in setup_features if f in self.df.columns]
        if available_setup:
            analysis['setup_analysis'] = self._analyze_setup_performance()
        
        return analysis
    
    def _analyze_speed_patterns(self) -> Dict[str, Any]:
        """Analyze speed patterns and acceleration"""
        speed_data = self.df['Physics_speed_kmh']
        
        # Calculate acceleration if possible
        acceleration = speed_data.diff() / 0.1  # Assuming 10Hz data rate
        
        patterns = {
            "max_speed": float(speed_data.max()),
            "avg_speed": float(speed_data.mean()),
            "speed_variance": float(speed_data.var()),
            "top_speed_percentage": float((speed_data > speed_data.quantile(0.9)).mean() * 100),
        }
        
        if not acceleration.isna().all():
            patterns.update({
                "max_acceleration": float(acceleration.max()),
                "max_deceleration": float(acceleration.min()),
                "avg_acceleration": float(acceleration[acceleration > 0].mean() if acceleration[acceleration > 0].any() else 0)
            })
        
        return patterns
    
    def _analyze_cornering_performance(self) -> Dict[str, Any]:
        """Analyze cornering performance using G-force data"""
        gx = self.df['Physics_g_force_x']
        gy = self.df['Physics_g_force_y']
        
        # Calculate lateral G-force magnitude
        lateral_g = np.sqrt(gx**2 + gy**2)
        
        return {
            "max_lateral_g": float(lateral_g.max()),
            "avg_lateral_g": float(lateral_g.mean()),
            "high_g_percentage": float((lateral_g > 1.0).mean() * 100),
            "cornering_consistency": float(lateral_g.std()),
            "max_longitudinal_g": float(self.df['Physics_g_force_y'].max()) if 'Physics_g_force_y' in self.df.columns else 0
        }
    
    def _analyze_braking_performance(self) -> Dict[str, Any]:
        """Analyze braking performance and patterns"""
        brake_data = self.df['Physics_brake']
        
        # Identify braking zones
        braking_threshold = brake_data.quantile(0.1)  # Above 10th percentile considered braking
        braking_zones = brake_data > braking_threshold
        
        analysis = {
            "max_brake_pressure": float(brake_data.max()),
            "avg_brake_pressure": float(brake_data.mean()),
            "braking_frequency": float(braking_zones.sum()),
            "braking_consistency": float(brake_data[braking_zones].std() if braking_zones.any() else 0)
        }
        
        # Add brake temperature analysis if available
        brake_temp_features = [col for col in self.df.columns if 'brake_temp' in col]
        if brake_temp_features:
            temps = []
            for feature in brake_temp_features:
                temps.extend(self.df[feature].values)
            
            analysis.update({
                "max_brake_temp": float(max(temps)),
                "avg_brake_temp": float(np.mean(temps)),
                "temp_variance": float(np.var(temps))
            })
        
        return analysis
    
    def _analyze_tyre_performance(self) -> Dict[str, Any]:
        """Analyze tyre performance across all four wheels"""
        tyre_features = {
            'front_left': 'Physics_tyre_core_temp_front_left',
            'front_right': 'Physics_tyre_core_temp_front_right', 
            'rear_left': 'Physics_tyre_core_temp_rear_left',
            'rear_right': 'Physics_tyre_core_temp_rear_right'
        }
        
        analysis = {
            "tyre_temperatures": {},
            "temperature_balance": {},
            "optimal_temp_percentage": {}
        }
        
        optimal_temp_range = (80, 110)  # Typical optimal range for racing tyres
        
        for position, feature in tyre_features.items():
            if feature in self.df.columns:
                temp_data = self.df[feature]
                analysis["tyre_temperatures"][position] = {
                    "max": float(temp_data.max()),
                    "avg": float(temp_data.mean()),
                    "min": float(temp_data.min())
                }
                
                # Calculate time in optimal temperature range
                in_optimal = ((temp_data >= optimal_temp_range[0]) & 
                             (temp_data <= optimal_temp_range[1]))
                analysis["optimal_temp_percentage"][position] = float(in_optimal.mean() * 100)
        
        # Calculate temperature balance between wheels
        if len(analysis["tyre_temperatures"]) >= 2:
            temps = [data["avg"] for data in analysis["tyre_temperatures"].values()]
            analysis["temperature_balance"]["variance"] = float(np.var(temps))
            analysis["temperature_balance"]["max_diff"] = float(max(temps) - min(temps))
        
        return analysis
    
    def _analyze_setup_performance(self) -> Dict[str, Any]:
        """Analyze car setup effectiveness"""
        setup_analysis = {}
        
        # Brake bias analysis
        if 'Physics_brake_bias' in self.df.columns:
            brake_bias = self.df['Physics_brake_bias']
            setup_analysis['brake_bias'] = {
                "average": float(brake_bias.mean()),
                "stability": float(brake_bias.std()),
                "range": float(brake_bias.max() - brake_bias.min())
            }
        
        # TC/ABS effectiveness
        if 'Physics_tc' in self.df.columns and 'Graphics_tc_level' in self.df.columns:
            tc_usage = self.df['Physics_tc']
            tc_level = self.df['Graphics_tc_level']
            setup_analysis['traction_control'] = {
                "avg_intervention": float(tc_usage.mean()),
                "level_setting": float(tc_level.mode().iloc[0] if not tc_level.empty else 0),
                "effectiveness": float((tc_usage > 0).mean() * 100)
            }
        
        if 'Physics_abs' in self.df.columns and 'Graphics_abs_level' in self.df.columns:
            abs_usage = self.df['Physics_abs']
            abs_level = self.df['Graphics_abs_level']
            setup_analysis['abs'] = {
                "avg_intervention": float(abs_usage.mean()),
                "level_setting": float(abs_level.mode().iloc[0] if not abs_level.empty else 0),
                "effectiveness": float((abs_usage > 0).mean() * 100)
            }
        
        return setup_analysis
        """Detect patterns in racing data using clustering"""
        if self.df.empty:
            return {"error": "No data available for pattern detection"}
        
        # Prepare features for clustering
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"error": "Not enough numeric features for pattern detection"}
        
        # Normalize the data
        features = self.scaler.fit_transform(self.df[numeric_cols])
        
        # Perform clustering
        n_clusters = min(5, len(self.df) // 2)  # Dynamic cluster count
        if n_clusters < 2:
            return {"error": "Not enough data points for clustering"}
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_data = self.df[clusters == i]
            cluster_analysis[f"pattern_{i}"] = {
                "size": len(cluster_data),
                "characteristics": self._analyze_cluster(cluster_data, numeric_cols)
            }
        
        return {
            "patterns_detected": n_clusters,
            "cluster_analysis": cluster_analysis,
            "insights": self._generate_pattern_insights(cluster_analysis)
        }
    
    def _analyze_cluster(self, cluster_data: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, float]:
        """Analyze characteristics of a cluster"""
        characteristics = {}
        for col in numeric_cols:
            characteristics[f"avg_{col}"] = float(cluster_data[col].mean())
            characteristics[f"std_{col}"] = float(cluster_data[col].std())
        return characteristics
    
    def _generate_pattern_insights(self, cluster_analysis: Dict) -> List[str]:
        """Generate human-readable insights from cluster analysis"""
        insights = []
        
        for pattern_name, pattern_data in cluster_analysis.items():
            size = pattern_data["size"]
            characteristics = pattern_data["characteristics"]
            
            # Find dominant characteristics
            avg_values = {k: v for k, v in characteristics.items() if k.startswith("avg_")}
            
            if avg_values:
                max_metric = max(avg_values.keys(), key=lambda k: avg_values[k])
                metric_name = max_metric.replace("avg_", "")
                insights.append(f"Pattern {pattern_name}: {size} data points with high {metric_name}")
        
        return insights
    
    def racing_performance_score(self) -> Dict[str, Any]:
        """Calculate a comprehensive racing performance score using all available telemetry"""
        score_components = {}
        overall_score = 0
        max_score = 0
        
        # Speed consistency score (30% weight)
        if 'Physics_speed_kmh' in self.df.columns:
            speed_data = self.df['Physics_speed_kmh']
            speed_std = speed_data.std()
            speed_mean = speed_data.mean()
            if speed_mean > 0:
                consistency_score = max(0, 100 - (speed_std / speed_mean * 100))
                score_components['speed_consistency'] = round(consistency_score, 2)
                overall_score += consistency_score * 0.3
                max_score += 30
        
        # Cornering performance score (25% weight)
        if all(f in self.df.columns for f in ['Physics_g_force_x', 'Physics_g_force_y']):
            gx = self.df['Physics_g_force_x']
            gy = self.df['Physics_g_force_y']
            lateral_g = np.sqrt(gx**2 + gy**2)
            
            # Score based on consistent high G-forces
            high_g_percentage = (lateral_g > 0.8).mean() * 100
            g_consistency = max(0, 100 - (lateral_g.std() * 20))
            cornering_score = (high_g_percentage * 0.6 + g_consistency * 0.4)
            
            score_components['cornering_performance'] = round(cornering_score, 2)
            overall_score += cornering_score * 0.25
            max_score += 25
        
        # Braking efficiency score (20% weight)
        if 'Physics_brake' in self.df.columns:
            brake_data = self.df['Physics_brake']
            # Score based on consistent braking patterns
            braking_zones = brake_data > 0.1
            if braking_zones.any():
                brake_consistency = max(0, 100 - (brake_data[braking_zones].std() * 100))
                score_components['braking_efficiency'] = round(brake_consistency, 2)
                overall_score += brake_consistency * 0.2
                max_score += 20
        
        # Tyre management score (15% weight)
        tyre_temp_features = [f for f in self.df.columns if 'tyre_core_temp' in f]
        if tyre_temp_features:
            optimal_range = (80, 110)
            total_optimal_time = 0
            for feature in tyre_temp_features:
                temp_data = self.df[feature]
                in_optimal = ((temp_data >= optimal_range[0]) & 
                             (temp_data <= optimal_range[1]))
                total_optimal_time += in_optimal.mean()
            
            tyre_score = (total_optimal_time / len(tyre_temp_features)) * 100
            score_components['tyre_management'] = round(tyre_score, 2)
            overall_score += tyre_score * 0.15
            max_score += 15
        
        # Setup optimization score (10% weight)
        setup_score = 0
        if 'Physics_tc' in self.df.columns:
            tc_data = self.df['Physics_tc']
            # Moderate TC usage is optimal
            tc_usage = (tc_data > 0).mean()
            if 0.05 <= tc_usage <= 0.2:  # 5-20% intervention is good
                setup_score += 50
            elif tc_usage < 0.05:
                setup_score += 30  # Too little, might be losing time
            else:
                setup_score += 20  # Too much, setup needs work
        
        if setup_score > 0:
            score_components['setup_optimization'] = setup_score
            overall_score += setup_score * 0.1
            max_score += 10
        
        # Lap time improvement (if available)
        if 'Graphics_last_time' in self.df.columns:
            lap_times = self.df[self.df['Graphics_last_time'] > 0]['Graphics_last_time']
            if len(lap_times) > 1:
                improvements = 0
                for i in range(1, len(lap_times)):
                    if lap_times.iloc[i] < lap_times.iloc[i-1]:
                        improvements += 1
                
                improvement_rate = (improvements / (len(lap_times) - 1)) * 100
                score_components['improvement_rate'] = round(improvement_rate, 2)
        
        # Calculate final score
        final_score = (overall_score / max_score * 100) if max_score > 0 else 0
        
        return {
            "overall_score": round(final_score, 2),
            "components": score_components,
            "grade": self._get_performance_grade(final_score),
            "recommendations": self._get_enhanced_performance_recommendations(score_components),
            "telemetry_coverage": self.feature_validation["coverage_percentage"],
            "analysis_confidence": "high" if max_score >= 70 else "medium" if max_score >= 40 else "low"
        }
    
    def _get_enhanced_performance_recommendations(self, components: Dict[str, float]) -> List[str]:
        """Generate enhanced performance improvement recommendations"""
        recommendations = []
        
        # Speed consistency recommendations
        if 'speed_consistency' in components:
            if components['speed_consistency'] < 70:
                recommendations.append("Focus on maintaining consistent speed - work on racing line precision and throttle control")
            elif components['speed_consistency'] < 85:
                recommendations.append("Good speed consistency - fine-tune braking points and corner exit timing")
        
        # Cornering performance recommendations
        if 'cornering_performance' in components:
            if components['cornering_performance'] < 60:
                recommendations.append("Cornering needs improvement - practice trail braking and gradual throttle application")
            elif components['cornering_performance'] < 80:
                recommendations.append("Solid cornering - work on maximizing lateral grip through apex positioning")
        
        # Braking efficiency recommendations
        if 'braking_efficiency' in components:
            if components['braking_efficiency'] < 70:
                recommendations.append("Inconsistent braking detected - practice threshold braking and brake point consistency")
            elif components['braking_efficiency'] < 85:
                recommendations.append("Good braking - optimize brake balance and trail braking technique")
        
        # Tyre management recommendations
        if 'tyre_management' in components:
            if components['tyre_management'] < 60:
                recommendations.append("Poor tyre temperature management - adjust driving style to keep tyres in optimal window")
            elif components['tyre_management'] < 80:
                recommendations.append("Decent tyre management - fine-tune setup to optimize tyre temperatures")
        
        # Setup recommendations
        if 'setup_optimization' in components:
            if components['setup_optimization'] < 40:
                recommendations.append("Car setup needs optimization - review brake bias, differential, and aerodynamic settings")
        
        # Improvement rate recommendations
        if 'improvement_rate' in components:
            if components['improvement_rate'] < 30:
                recommendations.append("Focus on consistent improvement - analyze data after each session to identify weak points")
            elif components['improvement_rate'] > 70:
                recommendations.append("Excellent progression rate - maintain this learning pace")
        
        if not recommendations:
            recommendations.append("Outstanding performance across all metrics! Focus on consistency and race craft")
        
        return recommendations
    
    def _get_performance_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
    
    def _get_performance_recommendations(self, components: Dict[str, float]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if 'speed_consistency' in components:
            if components['speed_consistency'] < 70:
                recommendations.append("Focus on maintaining consistent speed throughout the race")
        
        if 'improvement_rate' in components:
            if components['improvement_rate'] < 50:
                recommendations.append("Work on lap time improvement - analyze racing line and braking points")
        
        if not recommendations:
            recommendations.append("Great performance! Keep practicing to maintain your level")
        
        return recommendations
    
    def sector_analysis(self) -> Dict[str, Any]:
        """Analyze performance by track sectors if position data is available"""
        if 'position_x' not in self.df.columns or 'position_y' not in self.df.columns:
            return {"error": "Position data not available for sector analysis"}
        
        # Simple sector division based on position
        x_range = self.df['position_x'].max() - self.df['position_x'].min()
        sector_size = x_range / 3  # Divide track into 3 sectors
        
        sectors = []
        for _, row in self.df.iterrows():
            if row['position_x'] <= self.df['position_x'].min() + sector_size:
                sectors.append('Sector 1')
            elif row['position_x'] <= self.df['position_x'].min() + 2 * sector_size:
                sectors.append('Sector 2')
            else:
                sectors.append('Sector 3')
        
        self.df['sector'] = sectors
        
        # Analyze performance by sector
        sector_stats = {}
        for sector in ['Sector 1', 'Sector 2', 'Sector 3']:
            sector_data = self.df[self.df['sector'] == sector]
            if not sector_data.empty and 'speed' in sector_data.columns:
                sector_stats[sector] = {
                    "avg_speed": float(sector_data['speed'].mean()),
                    "max_speed": float(sector_data['speed'].max()),
                    "data_points": len(sector_data)
                }
        
        return {
            "sector_analysis": sector_stats,
            "best_sector": max(sector_stats.keys(), key=lambda k: sector_stats[k]["avg_speed"]) if sector_stats else None,
            "insights": self._generate_sector_insights(sector_stats)
        }
    
    def _generate_sector_insights(self, sector_stats: Dict) -> List[str]:
        """Generate insights from sector analysis"""
        insights = []
        
        if not sector_stats:
            return ["No sector data available"]
        
        speeds = {sector: stats["avg_speed"] for sector, stats in sector_stats.items()}
        fastest_sector = max(speeds.keys(), key=lambda k: speeds[k])
        slowest_sector = min(speeds.keys(), key=lambda k: speeds[k])
        
        insights.append(f"Fastest sector: {fastest_sector} (avg speed: {speeds[fastest_sector]:.1f})")
        insights.append(f"Slowest sector: {slowest_sector} (avg speed: {speeds[slowest_sector]:.1f})")
        
        speed_diff = speeds[fastest_sector] - speeds[slowest_sector]
        if speed_diff > 20:
            insights.append(f"Large speed difference ({speed_diff:.1f}) between sectors - check racing line consistency")
        
        return insights
    
    def predict_optimal_lap_time(self) -> Dict[str, Any]:
        """Predict optimal lap time based on best sector performances"""
        if 'lap_time' not in self.df.columns:
            return {"error": "Lap time data not available"}
        
        current_best = self.df['lap_time'].min()
        current_avg = self.df['lap_time'].mean()
        
        # Simple prediction based on consistency
        speed_consistency = self.df['speed'].std() / self.df['speed'].mean() if 'speed' in self.df.columns else 0.1
        
        # Lower consistency (higher std) means more room for improvement
        improvement_potential = speed_consistency * 2  # Max 2 seconds improvement
        predicted_optimal = current_best - improvement_potential
        
        return {
            "current_best_lap": round(current_best, 3),
            "current_average_lap": round(current_avg, 3),
            "predicted_optimal": round(predicted_optimal, 3),
            "improvement_potential": round(improvement_potential, 3),
            "confidence": "medium" if speed_consistency < 0.1 else "low"
        }
