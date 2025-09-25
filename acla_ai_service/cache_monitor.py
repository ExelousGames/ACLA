#!/usr/bin/env python3
"""
Cache Monitor for Large Models

This script helps monitor and manage cache usage for large ML models.
Run this to check cache status and optimize for large models.
"""

import asyncio
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService
from app.services.model_cache_service import model_cache_service
from app.services.backend_service import backend_service

def print_separator(title: str):
    """Print a formatted separator"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

async def main():
    print_separator("CACHE MONITOR - Large Model Analysis")
    
    # Initialize the ML service
    ml_service = Full_dataset_TelemetryMLService()
    
    # Print current cache debug info
    print_separator("Current Cache Status")
    ml_service.print_cache_debug_info()
    
    # Analyze large model usage
    print_separator("Large Model Analysis")
    analysis = ml_service.analyze_large_model_cache_usage()
    
    print(f"Memory Usage: {analysis['memory_usage_percent']:.1f}% ({analysis['total_memory_mb']:.1f}MB / {analysis['max_memory_mb']:.1f}MB)")
    print(f"Large Models (>500MB): {len(analysis['large_models'])}")
    print(f"Small Models (<500MB): {len(analysis['small_models'])}")
    
    if analysis['large_models']:
        print(f"\nLarge Models:")
        for model in analysis['large_models']:
            ttl_remaining = model.get('ttl_remaining_seconds', 0)
            ttl_hours = ttl_remaining / 3600 if ttl_remaining else 0
            print(f"  - {model['key']}: {model['size_mb']:.1f}MB, accessed {model['access_count']} times, TTL: {ttl_hours:.1f}h")
    
    if analysis['recommendations']:
        print(f"\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  • {rec}")
    
    # Offer optimization
    if analysis['memory_usage_percent'] > 70:
        print_separator("Optimization Available")
        print("Cache memory usage is high. Run optimization? (y/n): ", end="")
        response = input().strip().lower()
        
        if response == 'y':
            print("\nRunning optimization...")
            opt_result = ml_service.optimize_cache_for_large_models()
            
            print(f"Optimization completed: {opt_result['success']}")
            for action in opt_result['actions_taken']:
                print(f"  ✓ {action}")
            
            # Show updated stats
            print_separator("Updated Cache Status")
            ml_service.print_cache_debug_info()
    
    # Offer cache testing
    print_separator("Cache Performance Testing")
    print("Test cache performance with a model? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        print("Enter model type (imitation_learning, transformer_expert_action, etc.): ", end="")
        model_type = input().strip()
        
        print("Enter track name (optional, press Enter to skip): ", end="")
        track_name = input().strip() or None
        
        print("Enter car name (optional, press Enter to skip): ", end="")  
        car_name = input().strip() or None
        
        print(f"\nTesting cache performance for {model_type}...")
        
        # Note: This would need a deserializer function for real testing
        # For now, just show the test framework
        print("(Note: This would require implementing a deserializer function for the specific model type)")
        print("Example usage:")
        print(f"""
# In your code:
results = await ml_service.test_caching_performance(
    model_type="{model_type}",
    track_name="{track_name}",
    car_name="{car_name}",
    deserializer_func=your_deserializer_function,
    num_tests=3
)
        """)
    
    print_separator("Cache Management Tips")
    print("""
For managing 900MB models effectively:

1. Raw Data Caching (Enabled by default):
   - Models are stored as raw data and deserialized on-demand
   - Reduces memory footprint and improves cache hit rates
   - Configure with: ml_service.configure_caching_strategy(model_type, cache_raw_data=True)

2. Memory Configuration:
   - Current max memory: {:.1f}GB 
   - For multiple 900MB models, consider increasing to 8-12GB
   - Edit cache_config.py to adjust max_memory_mb

3. TTL Management:
   - Large models have longer TTLs (2-4 hours) to avoid re-downloading
   - Check TTL remaining with: ml_service.get_cache_debug_info()

4. Force Refresh:
   - Use validate_and_preload_model() with force_refresh=True if needed
   - This bypasses cache and re-downloads from backend

5. Monitor regularly:
   - Run this script periodically to check cache health
   - Use ml_service.print_cache_debug_info() in your code
    """.format(analysis['max_memory_mb'] / 1024))

if __name__ == "__main__":
    asyncio.run(main())