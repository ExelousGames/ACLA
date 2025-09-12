#!/usr/bin/env python3
"""
Standalone script to run corner identification analysis

This script initializes the Corner Identification Service and runs analysis
to identify and characterize corners from telemetry data.
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


async def main():
    """
    Main function to run corner identification analysis
    """
    print("=" * 60)
    print("ACLA Corner Geometry Feature Extraction")
    print("=" * 60)
    
    # Configuration
    TRACK_NAME = 'brands_hatch'  # Change this to analyze different tracks
    CAR_NAME = None  # Set to specific car name or None for all cars
    
    try:
        # Initialize the service
        print("[INFO] Initializing Corner Geometry Service...")
        models_directory = os.path.join(parent_dir, "models", "corner_geometry")
        service = CornerGeometryUnsupervisedService(models_directory=models_directory)
        
        print(f"[INFO] Starting geometry extraction for track: {TRACK_NAME}")
        print("[INFO] Analysis will perform:")
        print("       - Telemetry data retrieval and preprocessing")
        print("       - Corner detection using curvature analysis")
        print("       - Corner phase segmentation (entry, apex, exit)")
        print("       - Geometry feature extraction for sequence learning:")
        print("         * Curvature progression sequences")
        print("         * Position and tangent angle sequences") 
        print("         * Corner angles, radius, arc length")
        print("         * Phase timing ratios")
        print("         * Geometric complexity measures")
        print("       - Unsupervised clustering of similar corner geometries")
        print("       - Sequence-optimized feature formatting")
        print("       - Results storage in backend database")
        print()
        
        # Run the corner geometry extraction
        results = await service.extract_corner_geometry_features(TRACK_NAME, CAR_NAME)

        print("\n" + "=" * 60)
        print("CORNER GEOMETRY EXTRACTION COMPLETED")
        print("=" * 60)
        
        # Display comprehensive results
        if results and "error" not in results:
            print(f"[SUCCESS] Geometry extraction completed successfully!")
            
            # Basic statistics
            print(f"\n[BASIC STATISTICS]")
            print(f"  - Track: {results.get('track_name', 'Unknown')}")
            print(f"  - Car: {results.get('car_name', 'All cars') or 'All cars'}")
            print(f"  - Total corners detected: {results.get('total_corners', 0)}")
            
            data_stats = results.get('data_statistics', {})
            print(f"  - Telemetry records processed: {data_stats.get('total_records', 'N/A')}")
            print(f"  - Sessions analyzed: {data_stats.get('sessions_analyzed', 'N/A')}")
            print(f"  - Average corner duration: {data_stats.get('average_corner_duration', 0):.1f} data points")
            
            # Sequence learning metadata
            seq_metadata = results.get('sequence_metadata', {})
            print(f"\n[SEQUENCE LEARNING METADATA]")
            print(f"  - Max sequence length: {seq_metadata.get('max_sequence_length', 0)} data points")
            print(f"  - Average sequence length: {seq_metadata.get('avg_sequence_length', 0):.1f} data points")
            print(f"  - Geometry feature count: {seq_metadata.get('geometry_feature_count', 0)}")
            print(f"  - Sequence feature types: {', '.join(seq_metadata.get('sequence_feature_types', []))}")
            
            # Geometry summary
            geometry_summary = results.get('geometry_summary', {})
            
            # Geometry overview
            if 'geometry_overview' in geometry_summary:
                overview = geometry_summary['geometry_overview']
                print(f"\n[GEOMETRY OVERVIEW]")
                print(f"  - Average curvature: {overview.get('average_curvature', 0):.6f}")
                print(f"  - Average radius: {overview.get('average_radius', 0):.2f} meters")
                print(f"  - Average corner angle: {overview.get('average_corner_angle', 0):.4f} radians")
                print(f"  - Average complexity: {overview.get('average_complexity', 0):.4f}")
            
            # Direction distribution
            if 'direction_distribution' in geometry_summary:
                directions = geometry_summary['direction_distribution']
                print(f"\n[DIRECTION DISTRIBUTION]")
                print(f"  - Left turns: {directions.get('left_turns', 0)}")
                print(f"  - Right turns: {directions.get('right_turns', 0)}")
                print(f"  - Straight sections: {directions.get('straight_sections', 0)}")
                print(f"  - Turn balance: {directions.get('turn_balance', 0):.3f}")
            
            # Curvature analysis
            if 'curvature_analysis' in geometry_summary:
                curvature_analysis = geometry_summary['curvature_analysis']
                print(f"\n[CURVATURE ANALYSIS]")
                for curvature_type, count in curvature_analysis.items():
                    print(f"  - {curvature_type}: {count}")
            
            # Phase timing analysis
            if 'phase_timing_analysis' in geometry_summary:
                phase_analysis = geometry_summary['phase_timing_analysis']
                print(f"\n[PHASE TIMING ANALYSIS]")
                print(f"  - Average entry ratio: {phase_analysis.get('average_entry_ratio', 0):.3f}")
                print(f"  - Average apex ratio: {phase_analysis.get('average_apex_ratio', 0):.3f}") 
                print(f"  - Average exit ratio: {phase_analysis.get('average_exit_ratio', 0):.3f}")
                print(f"  - Entry ratio std: {phase_analysis.get('entry_ratio_std', 0):.3f}")
                print(f"  - Apex ratio std: {phase_analysis.get('apex_ratio_std', 0):.3f}")
                print(f"  - Exit ratio std: {phase_analysis.get('exit_ratio_std', 0):.3f}")
            
            # Sequence characteristics
            if 'sequence_characteristics' in geometry_summary:
                seq_chars = geometry_summary['sequence_characteristics']
                print(f"\n[SEQUENCE CHARACTERISTICS]")
                print(f"  - Min sequence length: {seq_chars.get('min_sequence_length', 0)} data points")
                print(f"  - Max sequence length: {seq_chars.get('max_sequence_length', 0)} data points")
                print(f"  - Avg sequence length: {seq_chars.get('avg_sequence_length', 0):.1f} data points")
                print(f"  - Sequence length std: {seq_chars.get('sequence_length_std', 0):.1f}")
            
            # Clustering results
            if 'clustering_results' in geometry_summary:
                cluster_results = geometry_summary['clustering_results']
                print(f"\n[GEOMETRY CLUSTERING RESULTS]")
                print(f"  - Total clusters identified: {cluster_results.get('total_clusters', 0)}")
                
                if 'cluster_sizes' in cluster_results:
                    cluster_sizes = cluster_results['cluster_sizes']
                    print(f"  - Cluster sizes:")
                    for cluster_name, size in cluster_sizes.items():
                        print(f"    * {cluster_name}: {size} corners")
                
                if 'dominant_cluster' in cluster_results:
                    print(f"  - Dominant cluster: {cluster_results['dominant_cluster']}")
            
            # Sequence learning insights
            if 'sequence_learning_insights' in geometry_summary:
                insights = geometry_summary['sequence_learning_insights']
                print(f"\n[SEQUENCE LEARNING INSIGHTS]")
                
                if 'geometry_complexity_distribution' in insights:
                    complexity_dist = insights['geometry_complexity_distribution']
                    print(f"  - Geometry complexity distribution:")
                    print(f"    * Simple corners: {complexity_dist.get('simple', 0)}")
                    print(f"    * Moderate corners: {complexity_dist.get('moderate', 0)}")
                    print(f"    * Complex corners: {complexity_dist.get('complex', 0)}")
                
                if 'recommended_sequence_padding' in insights:
                    print(f"  - Recommended sequence padding: {insights['recommended_sequence_padding']} data points")
                
                if 'feature_normalization_ranges' in insights:
                    norm_ranges = insights['feature_normalization_ranges']
                    print(f"  - Feature normalization ranges:")
                    for feature, range_vals in norm_ranges.items():
                        print(f"    * {feature}: [{range_vals[0]:.4f}, {range_vals[1]:.4f}]")
            
            # Show sample corner geometry data
            corner_geometries = results.get('corner_geometries', [])
            if corner_geometries:
                print(f"\n[SAMPLE CORNER GEOMETRY DATA]")
                sample_corner = corner_geometries[0]  # First corner as example
                
                geometry_scalars = sample_corner.get('geometry_scalars', {})
                print(f"Sample Corner: {sample_corner.get('corner_id', 'unknown')}")
                print(f"  - Corner angle: {geometry_scalars.get('corner_angle', 0):.4f} radians")
                print(f"  - Corner radius: {geometry_scalars.get('corner_radius', 0):.2f} meters")
                print(f"  - Max curvature: {geometry_scalars.get('max_curvature', 0):.6f}")
                print(f"  - Direction: {'Left' if geometry_scalars.get('direction', 0) < 0 else 'Right' if geometry_scalars.get('direction', 0) > 0 else 'Straight'}")
                print(f"  - Complexity score: {geometry_scalars.get('corner_complexity', 0):.4f}")
                
                geometry_sequences = sample_corner.get('geometry_sequences', {})
                curvature_progression = geometry_sequences.get('curvature_progression', [])
                print(f"  - Curvature sequence length: {len(curvature_progression)} data points")
                
                position_sequence = geometry_sequences.get('position_sequence', [])
                print(f"  - Position sequence length: {len(position_sequence)} data points")
        
        elif results and "error" in results:
            print(f"[ERROR] Geometry extraction failed: {results['error']}")
            return 1
        else:
            print("[WARNING] Geometry extraction completed but no results returned")
            
    except Exception as e:
        print(f"\n[ERROR] Corner geometry extraction failed: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        return 1
    
    print("\n" + "=" * 60)
    print("Corner geometry extraction completed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
