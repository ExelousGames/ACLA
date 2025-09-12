#!/usr/bin/env python3
"""
Test script for Corner Identification Unsupervised Service

This script tests the corner identification service with sample data
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the Python path to import modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import the Corner Geometry Service
from app.services.corner_identification_unsupervised_service import CornerGeometryUnsupervisedService


async def test_corner_geometry():
    """
    Test function for corner geometry extraction service
    """
    print("=" * 60)
    print("ACLA Corner Geometry Service Test")
    print("=" * 60)
    
    try:
        # Initialize the service
        print("[INFO] Initializing Corner Geometry Service...")
        models_directory = os.path.join(parent_dir, "models", "corner_geometry")
        service = CornerGeometryUnsupervisedService(models_directory=models_directory)
        
        print("[INFO] Starting corner geometry extraction test...")
        print("[INFO] This will:")
        print("       - Retrieve all racing sessions from the backend")
        print("       - Extract telemetry data from each session")
        print("       - Detect corners using curvature and steering analysis")
        print("       - Extract geometry features for sequence learning:")
        print("         * Curvature progression sequences")
        print("         * Position and tangent angle sequences")
        print("         * Corner angles, radius, arc length")
        print("         * Phase timing ratios")
        print("         * Geometric complexity measures")
        print("       - Cluster similar corner geometries using unsupervised learning")
        print("       - Format features for sequence-to-sequence learning")
        print("       - Save results to the backend")
        print()
        
        # Test with a specific track
        track_name = 'brands_hatch'  # You can change this to any track
        car_name = None  # Analyze all cars, or specify a car name
        
        results = await service.extract_corner_geometry_features(track_name, car_name)
        
        print("\n" + "=" * 60)
        print("CORNER GEOMETRY EXTRACTION COMPLETED")
        print("=" * 60)
        
        # Display results summary
        if results and "error" not in results:
            print(f"[SUCCESS] Geometry extraction completed successfully!")
            print(f"[INFO] Track: {results.get('track_name', 'Unknown')}")
            print(f"[INFO] Total corners detected: {results.get('total_corners', 0)}")
            
            # Display sequence metadata
            if 'sequence_metadata' in results:
                seq_metadata = results['sequence_metadata']
                print(f"\n[SEQUENCE LEARNING METADATA]")
                print(f"  - Max sequence length: {seq_metadata.get('max_sequence_length', 0)}")
                print(f"  - Avg sequence length: {seq_metadata.get('avg_sequence_length', 0):.1f}")
                print(f"  - Geometry feature count: {seq_metadata.get('geometry_feature_count', 0)}")
                print(f"  - Feature types: {', '.join(seq_metadata.get('sequence_feature_types', []))}")
            
            # Display geometry overview
            if 'geometry_summary' in results and 'geometry_overview' in results['geometry_summary']:
                geometry_overview = results['geometry_summary']['geometry_overview']
                print(f"\n[GEOMETRY OVERVIEW]")
                print(f"  - Average curvature: {geometry_overview.get('average_curvature', 0):.6f}")
                print(f"  - Average radius: {geometry_overview.get('average_radius', 0):.2f} meters")
                print(f"  - Average complexity: {geometry_overview.get('average_complexity', 0):.4f}")
            
            # Display direction distribution
            if 'geometry_summary' in results and 'direction_distribution' in results['geometry_summary']:
                directions = results['geometry_summary']['direction_distribution']
                print(f"\n[DIRECTION DISTRIBUTION]")
                print(f"  - Left turns: {directions.get('left_turns', 0)}")
                print(f"  - Right turns: {directions.get('right_turns', 0)}")
                print(f"  - Straight sections: {directions.get('straight_sections', 0)}")
            
            # Display clustering results
            if 'geometry_clusters' in results:
                clusters = results['geometry_clusters']
                print(f"\n[GEOMETRY CLUSTERING]")
                print(f"  - Total clusters identified: {len(clusters)}")
                for cluster_name, corner_ids in clusters.items():
                    print(f"  - {cluster_name}: {len(corner_ids)} corners")
        
        else:
            print(f"✗ Geometry extraction failed: {results.get('error', 'Unknown error')}")
    
        # Test retrieval of existing features
        print(f"\n[TESTING FEATURE RETRIEVAL]")
        retrieved_results = await service.get_corner_geometry_features(track_name, car_name)
        
        if retrieved_results and "error" not in retrieved_results:
            print(f"[SUCCESS] Retrieved features with {retrieved_results.get('total_corners', 0)} corners")
        else:
            print(f"[INFO] Feature retrieval test completed")
            
    except Exception as e:
        print(f"\n[ERROR] Corner geometry test failed: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        return 1
    
    print("\n" + "=" * 60)
    print("Corner geometry test completed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    # Run the async test function
    exit_code = asyncio.run(test_corner_geometry())
    sys.exit(exit_code)
