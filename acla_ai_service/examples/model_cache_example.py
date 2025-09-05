"""
Example script demonstrating the Model Cache Service usage
"""

import asyncio
import json
from datetime import datetime
from app.services.scikit_ml_service import TelemetryMLService

async def demonstrate_model_cache():
    """Demonstrate model caching functionality"""
    
    print("=== Model Cache Service Demonstration ===\n")
    
    # Initialize the ML service
    ml_service = TelemetryMLService()
    
    # 1. Show initial cache stats
    print("1. Initial cache statistics:")
    cache_stats = ml_service.get_cache_stats()
    print(json.dumps(cache_stats, indent=2))
    print()
    
    # 2. Simulate getting expert guidance (which will cache the model)
    track_name = "Monza"
    car_name = "BMW_M4_GT3"
    
    # Create sample telemetry data
    sample_telemetry = {
        "Physics_speed_kmh": 150.0,
        "Physics_steer": 0.1,
        "Physics_gas": 0.8,
        "Physics_brake": 0.0,
        "Graphics_normalized_car_position": 0.5,
        "Physics_g_force_lateral": 1.2,
        "Physics_g_force_longitudinal": 0.5,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"2. First call to get expert guidance for {track_name}/{car_name}:")
    print("   This should result in a cache miss and fetch from backend...")
    
    try:
        guidance_1 = await ml_service.get_imitation_learning_expert_guidance(
            current_telemetry=sample_telemetry,
            trackName=track_name,
            carName=car_name,
            guidance_type="both"
        )
        
        if guidance_1.get("success", False):
            print("   ✓ Expert guidance retrieved successfully")
        else:
            print(f"   ✗ Failed to get guidance: {guidance_1.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"   ✗ Exception occurred: {str(e)}")
    
    print()
    
    # 3. Show cache stats after first call
    print("3. Cache statistics after first call:")
    cache_stats = ml_service.get_cache_stats()
    print(f"   Cache size: {cache_stats['cache_size']}")
    print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"   Memory usage: {cache_stats['memory_usage_mb']:.2f} MB")
    print(f"   Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
    print()
    
    # 4. Second call - should hit cache
    print("4. Second call to get expert guidance:")
    print("   This should result in a cache hit...")
    
    try:
        guidance_2 = await ml_service.get_imitation_learning_expert_guidance(
            current_telemetry=sample_telemetry,
            trackName=track_name,
            carName=car_name,
            guidance_type="actions"
        )
        
        if guidance_2.get("success", False):
            print("   ✓ Expert guidance retrieved successfully (from cache)")
        else:
            print(f"   ✗ Failed to get guidance: {guidance_2.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"   ✗ Exception occurred: {str(e)}")
    
    print()
    
    # 5. Show updated cache stats
    print("5. Cache statistics after second call:")
    cache_stats = ml_service.get_cache_stats()
    print(f"   Cache size: {cache_stats['cache_size']}")
    print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"   Memory usage: {cache_stats['memory_usage_mb']:.2f} MB")
    print(f"   Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
    print()
    
    # 6. Show model cache info
    print("6. Specific model cache information:")
    model_info = ml_service.get_model_cache_info(
        model_type="imitation_learning",
        track_name=track_name,
        car_name=car_name,
        model_subtype="complete_model_data"
    )
    
    if model_info:
        print(f"   Model cached: Yes")
        print(f"   Access count: {model_info['access_count']}")
        print(f"   Size: {model_info['size_mb']:.2f} MB")
        print(f"   TTL remaining: {model_info['ttl_remaining_seconds']:.1f} seconds")
        print(f"   Last accessed: {model_info['last_accessed']}")
    else:
        print("   Model cached: No")
    
    print()
    
    # 7. Test cache invalidation
    print("7. Testing cache invalidation:")
    invalidated = ml_service.invalidate_model_cache(
        model_type="imitation_learning",
        track_name=track_name,
        car_name=car_name,
        model_subtype="complete_model_data"
    )
    print(f"   Model invalidated: {invalidated}")
    print()
    
    # 8. Final cache stats
    print("8. Final cache statistics after invalidation:")
    cache_stats = ml_service.get_cache_stats()
    print(f"   Cache size: {cache_stats['cache_size']}")
    print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"   Memory usage: {cache_stats['memory_usage_mb']:.2f} MB")
    print()
    
    # 9. Test preloading
    print("9. Testing model preloading:")
    preload_results = ml_service.preload_models_for_session(
        track_name="Spa",
        car_name="Mercedes_AMG_GT3",
        model_types=["imitation_learning"]
    )
    print(f"   Preloaded models: {preload_results['preloaded_models']}")
    print(f"   Failed models: {preload_results['failed_models']}")
    print(f"   Total time: {preload_results['total_preload_time']:.2f}s")
    print()
    
    print("=== Cache Service Demonstration Complete ===")


def demonstrate_cache_management():
    """Demonstrate cache management without async operations"""
    
    print("=== Cache Management Demonstration ===\n")
    
    ml_service = TelemetryMLService()
    
    # Show cache configuration
    cache_stats = ml_service.get_cache_stats()
    print("Cache Configuration:")
    print(f"  Max cache size: {cache_stats['max_cache_size']} models")
    print(f"  Max memory: {cache_stats['max_memory_mb']} MB")
    print(f"  Current size: {cache_stats['cache_size']} models")
    print(f"  Current memory usage: {cache_stats['memory_usage_mb']:.2f} MB")
    print()
    
    # Demonstrate cache operations
    print("Available Cache Management Methods:")
    print("  - invalidate_model_cache(model_type, track_name, car_name, model_subtype)")
    print("  - invalidate_track_cache(track_name)")  
    print("  - invalidate_car_cache(car_name)")
    print("  - get_cache_stats()")
    print("  - get_model_cache_info(model_type, track_name, car_name, model_subtype)")
    print("  - clear_all_cache()")
    print("  - preload_models_for_session(track_name, car_name, model_types)")
    print()
    
    print("Cache Performance Benefits:")
    print("  ✓ Faster prediction responses (no backend calls)")
    print("  ✓ Reduced network latency")
    print("  ✓ Better user experience during real-time guidance")
    print("  ✓ Lower backend server load")
    print("  ✓ Automatic memory management with LRU eviction")
    print("  ✓ TTL-based expiration for fresh data")
    print()


if __name__ == "__main__":
    print("Model Cache Service Examples")
    print("============================")
    print()
    
    # Run cache management demo (synchronous)
    demonstrate_cache_management()
    
    # Run async cache demo if desired
    print("\nTo run the full async demonstration, use:")
    print("  asyncio.run(demonstrate_model_cache())")
    
    # Uncomment the line below to run the async demo
    # asyncio.run(demonstrate_model_cache())
