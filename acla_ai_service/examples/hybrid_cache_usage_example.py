"""
Usage example for the new Hybrid Data Cache system in ACLA

This script demonstrates how to use the improved large dataset processing
with the hybrid cache system for memory-efficient training.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the service
sys.path.append(str(Path(__file__).parent.parent))

from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService


async def main():
    """
    Demonstrate hybrid data cache usage for large telemetry datasets
    """
    print("="*60)
    print("ACLA Hybrid Data Cache Usage Example")
    print("="*60)
    
    # Initialize the ML service with hybrid caching
    ml_service = Full_dataset_TelemetryMLService()
    
    # Print initial cache state
    print("\n1. Initial Cache State:")
    ml_service.print_data_cache_info()
    
    # Example track name
    track_name = "Monza"
    
    try:
        print(f"\n2. Starting Imitation Expert Pipeline for {track_name}")
        print("   This will:")
        print("   - Check hybrid cache first")
        print("   - Download and cache data if not present")
        print("   - Use efficient processing for large datasets")
        print("   - Train transformer model")
        
        # Start the pipeline (this will use the new hybrid cache)
        result = await ml_service.StartImitateExpertPipeline(track_name)
        
        if result.get("success"):
            print(f"✅ Pipeline completed successfully!")
            print(f"   Track: {result.get('track_name')}")
            print(f"   Transformer training: {result.get('transformer_training', {}).get('status', 'Unknown')}")
        else:
            print(f"❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Pipeline error: {str(e)}")
    
    # Print final cache state
    print(f"\n3. Final Cache State after processing {track_name}:")
    ml_service.print_data_cache_info()
    
    # Example: Clear cache for specific track
    print(f"\n4. Cache Management Example:")
    print(f"   Current cache info:")
    cache_info = ml_service.get_data_cache_info()
    print(f"   - Memory datasets: {cache_info['memory_cache']['entries']}")
    print(f"   - Disk datasets: {len(cache_info['disk_cache']['entries'])}")
    
    # Uncomment to clear cache for the track
    # ml_service.clear_data_cache(track_name)
    # print(f"   Cleared cache for {track_name}")
    
    print("\n" + "="*60)
    print("Example completed. Key benefits of hybrid cache:")
    print("✓ Memory-efficient processing of large datasets")
    print("✓ Persistent disk storage with compression")
    print("✓ Automatic fallback for memory limits")
    print("✓ Dask integration for distributed processing")
    print("✓ Smart caching prevents re-downloading")
    print("="*60)


if __name__ == "__main__":
    """
    Usage examples:
    
    # Basic usage
    python usage_example.py
    
    # To run with specific track
    # Modify track_name variable above
    """
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()