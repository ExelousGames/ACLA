"""
Example usage of the multi-algorithm machine learning system for racing telemetry
This script demonstrates how to use different algorithms for different prediction tasks
"""

import asyncio
import json
import pandas as pd
from typing import Dict, Any, List
from app.services.telemetry_service import TelemetryService
from app.models.ml_algorithms import AlgorithmConfiguration

async def demonstrate_multi_algorithm_system():
    """
    Demonstrate the multi-algorithm system with different prediction tasks
    """
    
    # Initialize services
    telemetry_service = TelemetryService()
    algorithm_config = AlgorithmConfiguration()
    
    print("=== Multi-Algorithm Racing Telemetry AI System ===\n")
    
    # Show available tasks and algorithms
    print("Available Prediction Tasks:")
    tasks = algorithm_config.get_supported_tasks()
    for task in tasks:
        description = algorithm_config.get_task_description(task)
        primary_algo = algorithm_config.model_configs[task]["primary"]
        alternatives = algorithm_config.model_configs[task]["alternatives"]
        print(f"  • {task}")
        print(f"    Description: {description}")
        print(f"    Primary Algorithm: {primary_algo}")
        print(f"    Alternatives: {', '.join(alternatives)}")
        print()
    
    # Generate sample telemetry data for demonstration
    sample_telemetry = generate_sample_telemetry_data()
    
    # Demonstrate different prediction tasks with optimal algorithms
    prediction_tasks = [
        {
            "name": "Lap Time Prediction",
            "model_type": "lap_time_prediction",
            "target_variable": "lap_time",
            "description": "Predicting lap times using gradient boosting"
        },
        {
            "name": "Fuel Consumption Analysis", 
            "model_type": "fuel_consumption",
            "target_variable": "fuel_consumption",
            "description": "Linear regression for fuel consumption patterns"
        },
        {
            "name": "Setup Recommendation",
            "model_type": "setup_recommendation", 
            "target_variable": "performance_category",
            "description": "Classification for optimal car setup"
        },
        {
            "name": "Brake Performance Analysis",
            "model_type": "brake_performance",
            "target_variable": "brake_efficiency",
            "description": "SVR for brake performance prediction"
        }
    ]
    
    print("=== Training Models with Different Algorithms ===\n")
    
    results = {}
    
    for task in prediction_tasks:
        print(f"Training: {task['name']}")
        print(f"Description: {task['description']}")
        print(f"Model Type: {task['model_type']}")
        
        try:
            # Train with default algorithm
            result = await telemetry_service.train_ai_model(
                telemetry_data=sample_telemetry,
                target_variable=task["target_variable"],
                model_type=task["model_type"],
                user_id="demo_user",
                session_metadata={"demo": True, "task": task["name"]}
            )
            
            if result.get("success", False):
                print(f"✓ Success with {result['algorithm_used']} algorithm")
                print(f"  - Algorithm Type: {result['algorithm_type']}")
                print(f"  - Features Used: {result['feature_count']}")
                print(f"  - Training Samples: {result['training_samples']}")
                print(f"  - Supports Incremental Learning: {result['supports_incremental']}")
                
                # Show metrics
                metrics = result['training_metrics']
                if result['algorithm_type'] == 'regression':
                    print(f"  - Test R² Score: {metrics.get('test_r2', 'N/A'):.3f}")
                    print(f"  - Test RMSE: {(metrics.get('test_mse', 0) ** 0.5):.3f}")
                else:
                    print(f"  - Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.3f}")
                
                # Show feature importance if available
                if result.get('feature_importance'):
                    top_features = list(result['feature_importance'].items())[:3]
                    print(f"  - Top Features: {[f[0] for f in top_features]}")
                
                print(f"  - Recommendations: {len(result['recommendations'])} suggestions")
                
                results[task["model_type"]] = result
                
            else:
                print(f"✗ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"✗ Error: {str(e)}")
        
        print()
    
    # Demonstrate algorithm comparison
    print("=== Algorithm Comparison for Lap Time Prediction ===\n")
    
    lap_time_algorithms = algorithm_config.get_algorithm_alternatives("lap_time_prediction")
    comparison_results = {}
    
    for algorithm in lap_time_algorithms[:3]:  # Test top 3 algorithms
        print(f"Testing {algorithm}...")
        
        try:
            result = await telemetry_service.train_ai_model(
                telemetry_data=sample_telemetry,
                target_variable="lap_time",
                model_type="lap_time_prediction",
                preferred_algorithm=algorithm,
                user_id="comparison_test"
            )
            
            if result.get("success", False):
                metrics = result['training_metrics']
                r2_score = metrics.get('test_r2', 0)
                rmse = (metrics.get('test_mse', 0) ** 0.5)
                
                comparison_results[algorithm] = {
                    "r2_score": r2_score,
                    "rmse": rmse,
                    "training_time": "Fast" if algorithm in ["linear_regression", "ridge"] else "Medium"
                }
                
                print(f"  ✓ R² Score: {r2_score:.3f}, RMSE: {rmse:.3f}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown')}")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    # Show comparison summary
    if comparison_results:
        print("\nComparison Summary:")
        best_algorithm = max(comparison_results.keys(), 
                           key=lambda x: comparison_results[x]["r2_score"])
        
        for algo, metrics in comparison_results.items():
            status = "🏆 BEST" if algo == best_algorithm else ""
            print(f"  {algo}: R²={metrics['r2_score']:.3f}, "
                  f"RMSE={metrics['rmse']:.3f}, "
                  f"Speed={metrics['training_time']} {status}")
    
    print("\n=== Incremental Learning Demonstration ===\n")
    
    # Demonstrate incremental learning
    if "lap_time_prediction" in results:
        base_model = results["lap_time_prediction"]["model_data"]
        
        print("Training incremental update with new data...")
        new_telemetry = generate_sample_telemetry_data(size=50)
        
        try:
            incremental_result = await telemetry_service.train_ai_model(
                telemetry_data=new_telemetry,
                target_variable="lap_time",
                model_type="lap_time_prediction",
                existing_model_data=base_model,
                user_id="incremental_test"
            )
            
            if incremental_result.get("success", False):
                print(f"✓ Incremental training successful")
                print(f"  - Model Version: {incremental_result['model_version']}")
                print(f"  - Algorithm: {incremental_result['algorithm_used']}")
                
                # Compare performance
                old_r2 = results["lap_time_prediction"]["training_metrics"].get("test_r2", 0)
                new_r2 = incremental_result["training_metrics"].get("test_r2", 0)
                improvement = new_r2 - old_r2
                
                print(f"  - Performance Change: {improvement:+.3f} R² score")
            else:
                print(f"✗ Incremental training failed: {incremental_result.get('error')}")
                
        except Exception as e:
            print(f"✗ Incremental training error: {str(e)}")
    
    print("\n=== Summary ===")
    print(f"Successfully demonstrated {len(results)} different prediction tasks")
    print(f"Each task used its optimal algorithm automatically")
    print("The system supports:")
    print("  • Multiple algorithm types (regression, classification)")
    print("  • Task-specific feature selection")
    print("  • Incremental learning where supported")
    print("  • Performance monitoring and recommendations")
    print("  • Easy algorithm comparison and selection")


def generate_sample_telemetry_data(size: int = 100) -> List[Dict[str, Any]]:
    """
    Generate sample telemetry data for demonstration
    """
    import random
    import math
    
    data = []
    
    for i in range(size):
        # Generate realistic racing telemetry values
        speed = random.uniform(80, 300)  # km/h
        throttle = random.uniform(0, 100)  # percentage
        brake = random.uniform(0, 100) if speed > 200 else random.uniform(0, 30)
        steering_angle = random.uniform(-30, 30)  # degrees
        
        # Derived values
        lap_time = 90 + random.uniform(-10, 10) + (300 - speed) * 0.1  # Simplified lap time calculation
        fuel_consumption = throttle * 0.02 + random.uniform(0, 0.5)
        brake_efficiency = max(0, 100 - brake * 0.8 + random.uniform(-10, 10))
        
        # Performance category (for classification)
        if lap_time < 85:
            performance_category = 2  # Fast
        elif lap_time < 95:
            performance_category = 1  # Medium  
        else:
            performance_category = 0  # Slow
        
        record = {
            # Physics data
            "Physics_speed_kmh": speed,
            "Physics_gas": throttle,
            "Physics_brake": brake,
            "Physics_steer_angle": steering_angle,
            "Physics_g_force_x": random.uniform(-2, 2),
            "Physics_g_force_y": random.uniform(-2, 2),
            "Physics_g_force_z": random.uniform(-1, 1),
            "Physics_wheel_slip_front_left": random.uniform(0, 10),
            "Physics_wheel_slip_front_right": random.uniform(0, 10),
            "Physics_brake_temp_front_left": random.uniform(200, 800),
            "Physics_brake_temp_front_right": random.uniform(200, 800),
            "Physics_tyre_core_temp_front_left": random.uniform(70, 120),
            "Physics_tyre_core_temp_front_right": random.uniform(70, 120),
            
            # Graphics data
            "Graphics_fuel_per_lap": fuel_consumption,
            "Graphics_last_time": lap_time,
            "Graphics_last_sector_time": lap_time / 3,
            "Graphics_position": random.randint(1, 20),
            
            # Target variables
            "lap_time": lap_time,
            "fuel_consumption": fuel_consumption,
            "brake_efficiency": brake_efficiency,
            "performance_category": performance_category,
            
            # Timestamp
            "timestamp": i
        }
        
        data.append(record)
    
    return data


if __name__ == "__main__":
    asyncio.run(demonstrate_multi_algorithm_system())
