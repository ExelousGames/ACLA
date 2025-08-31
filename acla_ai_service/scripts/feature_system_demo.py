"""
Demonstration of the centralized feature recommendation system
"""

from app.models.telemetry_models import TelemetryFeatures

def demonstrate_feature_system():
    """Show how the centralized feature system works"""
    
    print("=== Centralized Feature Recommendation System ===\n")
    
    # Show all available model types and their features
    model_types = [
        "lap_time_prediction",
        "sector_time_prediction", 
        "fuel_consumption",
        "brake_performance",
        "tire_strategy",
        "setup_recommendation",
        "damage_prediction",
        "overtaking_opportunity",
        "racing_line_optimization",
        "weather_adaptation",
        "consistency_analysis"
    ]
    
    print("Model Types and Their Recommended Features:\n")
    
    for model_type in model_types:
        features = TelemetryFeatures.get_features_for_model_type(model_type)
        print(f"📊 {model_type.replace('_', ' ').title()}:")
        print(f"   Features: {len(features)} recommended")
        print(f"   Top 5: {features[:5]}")
        print()
    
    # Show feature categories
    print("=== Feature Categories ===\n")
    
    categories = TelemetryFeatures.get_feature_categories()
    
    for category, features in categories.items():
        print(f"🔧 {category.replace('_', ' ').title()}:")
        print(f"   Count: {len(features)} features")
        print(f"   Examples: {features[:3]}")
        print()
    
    # Demonstrate feature filtering
    print("=== Feature Filtering Example ===\n")
    
    # Simulate available columns in a dataset
    available_columns = [
        "Physics_speed_kmh",
        "Physics_gas", 
        "Physics_brake",
        "Physics_gear",
        "Physics_rpm",
        "Physics_g_force_x",
        "Physics_g_force_y",
        "Graphics_last_time",
        "Graphics_position",
        "Graphics_fuel_per_lap",
        "Physics_brake_temp_front_left",
        "Physics_tyre_core_temp_front_left",
        "some_other_column"
    ]
    
    print(f"Available columns in dataset: {len(available_columns)}")
    print(f"Columns: {available_columns[:5]}...")
    print()
    
    # Test feature filtering for different model types
    test_models = ["lap_time_prediction", "fuel_consumption", "brake_performance"]
    
    for model_type in test_models:
        recommended = TelemetryFeatures.get_features_for_model_type(model_type)
        filtered = TelemetryFeatures.filter_available_features(recommended, available_columns)
        
        print(f"🎯 {model_type}:")
        print(f"   Recommended: {len(recommended)} features")
        print(f"   Available: {len(filtered)} features")
        print(f"   Filtered features: {filtered}")
        print()
    
    # Test fallback features
    print("=== Fallback Feature System ===\n")
    
    fallback = TelemetryFeatures.get_fallback_features(available_columns, "Graphics_last_time")
    print(f"Fallback features (excluding target 'Graphics_last_time'):")
    print(f"Count: {len(fallback)}")
    print(f"Features: {fallback}")
    print()
    
    # Show total feature coverage
    print("=== System Overview ===\n")
    
    all_features = TelemetryFeatures.get_all_features()
    performance_features = TelemetryFeatures.get_performance_critical_features()
    
    print(f"📈 Total AC Competizione Features: {len(all_features)}")
    print(f"🏁 Performance Critical Features: {len(performance_features)}")
    print(f"🎛️  Feature Categories: {len(categories)}")
    print(f"🤖 Supported Model Types: {len(model_types)}")
    
    print("\n✅ All features are now centrally managed in TelemetryFeatures class!")
    print("✅ Task-specific feature selection is automated!")
    print("✅ Fallback system handles missing features gracefully!")


if __name__ == "__main__":
    demonstrate_feature_system()
