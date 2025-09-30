"""
Large Dataset Processing Example for ACLA

This example demonstrates how the updated pipeline handles very large datasets
without any fallback mechanisms, using only efficient streaming processing.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the service
sys.path.append(str(Path(__file__).parent.parent))

from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService


async def process_large_dataset_example():
    """
    Example of processing a very large telemetry dataset
    """
    print("="*80)
    print("ACLA Large Dataset Processing - No Fallback Mode")
    print("="*80)
    
    # Initialize the ML service
    ml_service = Full_dataset_TelemetryMLService()
    
    # Print cache configuration
    print("\n1. Hybrid Cache Configuration:")
    ml_service.print_data_cache_info()
    
    # Example track with presumably large dataset
    track_name = "Silverstone"
    
    try:
        print(f"\n2. Processing Large Dataset for {track_name}")
        print("   Configuration:")
        print("   - Always assumes very large dataset")
        print("   - No fallback to standard processing")
        print("   - Uses streaming with HDF5/Dask backend")
        print("   - Conservative memory limits (50k records per chunk)")
        print("   - Aggressive filtering (top 0.5% as expert data)")
        
        # Start the pipeline - will always use efficient processing
        result = await ml_service.StartImitateExpertPipeline(track_name)
        
        if result.get("success"):
            print(f"\n✅ Large Dataset Pipeline Completed Successfully!")
            print(f"   Track: {result.get('track_name')}")
            
            transformer_result = result.get('transformer_training', {})
            print(f"   Transformer Status: {transformer_result.get('status', 'Unknown')}")
            
            if transformer_result.get('training_summary'):
                summary = transformer_result['training_summary']
                print(f"   Training Segments: {summary.get('total_segments', 'Unknown')}")
                print(f"   Model Performance: {summary.get('final_performance', 'Unknown')}")
                
        else:
            print(f"\n❌ Pipeline Failed: {result.get('error', 'Unknown error')}")
            print("   This indicates an issue with:")
            print("   - Hybrid cache data retrieval")
            print("   - Streaming data processing")
            print("   - HDF5/Dask backend processing")
            
    except Exception as e:
        print(f"\n❌ Pipeline Exception: {str(e)}")
        print("   The pipeline failed at the data processing level")
        print("   No fallback was attempted as configured")
    
    # Show final cache state
    print(f"\n3. Final Cache State:")
    ml_service.print_data_cache_info()
    
    print("\n" + "="*80)
    print("Large Dataset Processing Characteristics:")
    print("✓ Memory-efficient streaming processing")
    print("✓ HDF5 compressed storage with LZ4")
    print("✓ Dask distributed processing when available")
    print("✓ Conservative memory limits (50k records/chunk)")
    print("✓ Aggressive data filtering (0.5% expert selection)")
    print("✓ No fallback - fails fast if streaming doesn't work")
    print("✓ Optimized for multi-GB datasets")
    print("="*80)


async def demonstrate_cache_management():
    """
    Demonstrate cache management for large datasets
    """
    print("\n" + "="*60)
    print("CACHE MANAGEMENT FOR LARGE DATASETS")
    print("="*60)
    
    ml_service = Full_dataset_TelemetryMLService()
    
    # Show cache info
    cache_info = ml_service.get_data_cache_info()
    print(f"\nCache Status:")
    print(f"- Memory datasets: {cache_info['memory_cache']['entries']}")
    print(f"- Disk datasets: {len(cache_info['disk_cache']['entries'])}")
    print(f"- Dask enabled: {cache_info['dask_enabled']}")
    
    if cache_info['disk_cache']['entries']:
        print(f"\nCached Datasets:")
        for entry in cache_info['disk_cache']['entries'][:3]:  # Show first 3
            print(f"  - {entry['track_name']}: {entry['record_count']} records "
                  f"({entry['size_mb']:.1f}MB, {entry['storage_type']})")
    
    # Example: Clear specific track cache
    track_to_clear = "Monza"  # Example
    print(f"\nClearing cache for {track_to_clear}...")
    ml_service.clear_data_cache(track_to_clear)
    
    print("Cache management completed.")


if __name__ == "__main__":
    """
    Run the large dataset processing example
    """
    try:
        print("Starting Large Dataset Processing Example...")
        
        # Run the main example
        asyncio.run(process_large_dataset_example())
        
        # Demonstrate cache management
        asyncio.run(demonstrate_cache_management())
        
        print("\nExample completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nCleaning up...")