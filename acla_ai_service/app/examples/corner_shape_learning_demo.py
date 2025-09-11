#!/usr/bin/env python3
"""
Example usage script for Corner Shape Unsupervised Learning Service

This script demonstrates how to use the corner shape learning service to:
1. Learn corner shapes from racing data
2. Predict corner types for new telemetry
3. Analyze corner characteristics
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the parent directory to Python path so we can import the service
sys.path.append(str(Path(__file__).parent.parent))

from app.services.corner_shape_unsupervised_service import corner_shape_service
from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService


async def demo_corner_shape_learning():
    """Demonstrate corner shape learning capabilities"""
    
    print("🏁 Corner Shape Unsupervised Learning Demo")
    print("=" * 50)
    
    # Initialize the main ML service
    ml_service = Full_dataset_TelemetryMLService()
    
    # Example track name
    track_name = "monza"
    
    print(f"\n📍 Track: {track_name}")
    print("🔄 Starting corner shape learning...")
    
    # Step 1: Learn corner shapes for the track
    try:
        clustering_params = {
            'n_clusters': 6,  # Try to find 6 different corner types
            'eps': 0.4,       # DBSCAN parameter
            'min_samples': 3  # DBSCAN minimum samples per cluster
        }
        
        learning_results = await ml_service.learn_corner_shapes(
            trackName=track_name, 
            clustering_params=clustering_params
        )
        
        if 'error' in learning_results:
            print(f"❌ Learning failed: {learning_results['error']}")
            return
        
        print("✅ Corner shape learning completed!")
        print(f"📊 Results summary:")
        print(f"   - Total sessions analyzed: {learning_results.get('total_sessions_analyzed', 0)}")
        print(f"   - Total corners found: {learning_results.get('total_corners_found', 0)}")
        
        # Get clustering results
        clustering_info = learning_results.get('clustering_results', {})
        best_algorithm = clustering_info.get('best_algorithm', 'unknown')
        best_clustering = clustering_info.get('best_clustering', {})
        
        print(f"   - Best algorithm: {best_algorithm}")
        print(f"   - Number of clusters: {best_clustering.get('n_clusters', 0)}")
        print(f"   - Silhouette score: {best_clustering.get('silhouette_score', 0):.3f}")
        
        # Show cluster analysis
        cluster_analysis = learning_results.get('cluster_analysis', {})
        cluster_chars = cluster_analysis.get('cluster_characteristics', {})
        
        print(f"\n🎯 Discovered Corner Types:")
        for cluster_name, cluster_info in cluster_chars.items():
            cluster_id = cluster_info.get('cluster_id', 'unknown')
            size = cluster_info.get('size', 0)
            percentage = cluster_info.get('percentage', 0)
            description = cluster_info.get('description', 'No description')
            
            print(f"   {cluster_name}: {size} corners ({percentage:.1f}%) - {description}")
        
    except Exception as e:
        print(f"❌ Error during corner shape learning: {str(e)}")
        return
    
    # Step 2: Demonstrate corner shape prediction
    print(f"\n🔮 Testing corner shape prediction...")
    
    # Create example telemetry data for prediction
    example_telemetry = {
        'Physics_speed_kmh': 120.5,
        'Physics_steer_angle': 1500,  # In degrees * 10
        'Physics_brake': 0.8,
        'Physics_gas': 0.2,
        'Physics_g_force_x': -1.2,
        'Physics_g_force_y': -0.8,
        'Graphics_normalized_car_position': 0.25
    }
    
    try:
        prediction_result = await ml_service.predict_corner_shape_type(
            trackName=track_name,
            current_telemetry=example_telemetry
        )
        
        if 'error' in prediction_result:
            print(f"❌ Prediction failed: {prediction_result['error']}")
        else:
            cluster_id = prediction_result.get('predicted_cluster', 'unknown')
            confidence = prediction_result.get('confidence', 0)
            description = prediction_result.get('cluster_description', 'Unknown corner type')
            
            print(f"✅ Prediction successful!")
            print(f"   - Predicted cluster: {cluster_id}")
            print(f"   - Confidence: {confidence:.2f}")
            print(f"   - Corner type: {description}")
            
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
    
    # Step 3: Get analysis summary
    print(f"\n📈 Getting analysis summary...")
    
    try:
        summary = ml_service.get_corner_shape_analysis_summary(track_name)
        
        if 'error' in summary:
            print(f"❌ Summary failed: {summary['error']}")
        else:
            print(f"✅ Analysis summary retrieved!")
            print(f"   - Model name: {summary.get('model_name', 'Unknown')}")
            print(f"   - Learning timestamp: {summary.get('learning_timestamp', 'Unknown')}")
            print(f"   - Features used: {len(summary.get('feature_names', []))}")
            
    except Exception as e:
        print(f"❌ Error getting summary: {str(e)}")
    
    print(f"\n🏁 Demo completed!")
    print("=" * 50)


async def demo_corner_features():
    """Demonstrate corner feature extraction"""
    
    print("\n🔧 Corner Feature Extraction Demo")
    print("=" * 40)
    
    # Import required classes
    from app.services.corner_shape_unsupervised_service import CornerShapeFeatureExtractor
    import pandas as pd
    import numpy as np
    
    # Create feature extractor
    extractor = CornerShapeFeatureExtractor()
    
    # Create example corner telemetry data
    corner_data = {
        'Physics_speed_kmh': [150, 140, 120, 100, 90, 95, 110, 130],
        'Physics_steer_angle': [0, 500, 1000, 1500, 1500, 1200, 800, 200],
        'Physics_brake': [0, 0.3, 0.8, 1.0, 0.5, 0, 0, 0],
        'Physics_gas': [0.8, 0.5, 0, 0, 0.2, 0.6, 0.8, 0.9],
        'Physics_g_force_x': [0, -0.5, -1.2, -1.8, -1.5, -1.0, -0.6, -0.2],
        'Physics_g_force_y': [0.2, -0.3, -0.8, -0.5, 0.1, 0.5, 0.3, 0.2],
        'Graphics_normalized_car_position': [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27]
    }
    
    corner_df = pd.DataFrame(corner_data)
    
    # Create example corner info
    corner_info = {
        'corner_start_position': 0.20,
        'corner_end_position': 0.27,
        'total_duration_points': 8,
        'phases': {
            'entry': {'normalized_car_position': 0.20, 'duration_points': 2},
            'turn_in': {'normalized_car_position': 0.22, 'duration_points': 2},
            'apex': {'normalized_car_position': 0.24, 'duration_points': 1},
            'acceleration': {'normalized_car_position': 0.25, 'duration_points': 1},
            'exit': {'normalized_car_position': 0.27, 'duration_points': 2}
        }
    }
    
    # Extract features
    try:
        features = extractor.extract_corner_shape_features(corner_df, corner_info)
        
        print("✅ Features extracted successfully!")
        print("🎯 Corner Shape Features:")
        
        for feature_name, value in features.items():
            print(f"   {feature_name}: {value:.3f}" if isinstance(value, float) else f"   {feature_name}: {value}")
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {str(e)}")
    
    print("=" * 40)


def print_usage_example():
    """Print usage example for the corner shape service"""
    
    print("\n📚 Usage Example Code")
    print("=" * 30)
    
    example_code = '''
# Basic Usage Example
from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService

# Initialize service
ml_service = Full_dataset_TelemetryMLService()

# Learn corner shapes for a track
results = await ml_service.learn_corner_shapes(
    trackName="monza",
    clustering_params={'n_clusters': 6}
)

# Predict corner type for current telemetry
telemetry = {
    'Physics_speed_kmh': 120,
    'Physics_steer_angle': 1500,
    'Physics_brake': 0.5,
    # ... more telemetry data
}

prediction = await ml_service.predict_corner_shape_type(
    trackName="monza",
    current_telemetry=telemetry
)

print(f"Corner type: {prediction['cluster_description']}")
print(f"Confidence: {prediction['confidence']:.2f}")
'''
    
    print(example_code)
    print("=" * 30)


async def main():
    """Main demo function"""
    try:
        await demo_corner_features()
        print_usage_example()
        
        # Only run the full demo if backend connection is available
        print("\n⚠️  Full demo requires backend connection to retrieve racing data.")
        print("🔄 Attempting corner shape learning demo...")
        
        await demo_corner_shape_learning()
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("📝 This is expected if no backend connection or racing data is available.")
        print("🔧 The service is ready to use when properly configured!")


if __name__ == "__main__":
    print("🚀 Starting Corner Shape Unsupervised Learning Demo...")
    asyncio.run(main())
