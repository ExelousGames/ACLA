"""
Example usage of Corner Geometry Unsupervised Service

This example demonstrates how to use the corner geometry service
to extract geometric features optimized for sequence-to-sequence learning.
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.corner_identification_unsupervised_service import CornerGeometryUnsupervisedService


async def corner_geometry_example():
    """
    Example of how to use the corner geometry service for sequence learning
    """
    print("Corner Geometry Service Example")
    print("=" * 50)
    
    # Initialize the service
    service = CornerGeometryUnsupervisedService(models_directory="models/corner_geometry")
    
    # Example 1: Extract corner geometry features for a specific track
    print("\n[Example 1] Extracting corner geometry for Brands Hatch:")
    try:
        results = await service.extract_corner_geometry_features('brands_hatch')
        
        if results and "error" not in results:
            print(f"✓ Successfully extracted geometry for {results['total_corners']} corners")
            
            # Show sequence metadata
            if 'sequence_metadata' in results:
                metadata = results['sequence_metadata']
                print(f"Sequence metadata:")
                print(f"  - Max sequence length: {metadata.get('max_sequence_length', 0)} data points")
                print(f"  - Avg sequence length: {metadata.get('avg_sequence_length', 0):.1f} data points")
                print(f"  - Geometry features: {metadata.get('geometry_feature_count', 0)}")
            
            # Show geometry summary
            if 'geometry_summary' in results and 'direction_distribution' in results['geometry_summary']:
                directions = results['geometry_summary']['direction_distribution']
                print(f"Direction distribution: {directions}")
        
        else:
            print(f"✗ Geometry extraction failed: {results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"✗ Error in extraction: {str(e)}")
    
    # Example 2: Get existing geometry features
    print("\n[Example 2] Getting existing geometry features:")
    try:
        # Get existing features (or extract new ones if not found)
        geometry_data = await service.get_corner_geometry_features('brands_hatch')
        
        if geometry_data and "error" not in geometry_data:
            corner_geometries = geometry_data.get('corner_geometries', [])
            print(f"Retrieved geometry data for {len(corner_geometries)} corners")
            
            if corner_geometries:
                # Show first corner as example
                sample_corner = corner_geometries[0]
                print(f"Sample corner geometry data structure:")
                print(f"  - Corner ID: {sample_corner.get('corner_id', 'unknown')}")
                print(f"  - Geometry scalars: {list(sample_corner.get('geometry_scalars', {}).keys())}")
                print(f"  - Geometry sequences: {list(sample_corner.get('geometry_sequences', {}).keys())}")
                print(f"  - Phase ratios: {list(sample_corner.get('phase_ratios', {}).keys())}")
        
        else:
            print("✗ Could not get geometry features")
    
    except Exception as e:
        print(f"✗ Error getting features: {str(e)}")
    
    # Example 3: Understanding geometry feature structure
    print("\n[Example 3] Geometry feature structure for sequence learning:")
    print("Each detected corner includes:")
    print("  GEOMETRY SCALARS (13 features):")
    print("    - corner_angle, corner_radius, arc_length, chord_length")
    print("    - max_curvature, avg_curvature, curvature_rate_of_change")
    print("    - direction (-1=left, +1=right, 0=straight)")
    print("    - banking_angle, corner_complexity")
    print("    - entry_angle, exit_angle, apex_sharpness")
    print()
    print("  GEOMETRY SEQUENCES (variable length):")
    print("    - curvature_progression: [float, ...] - curvature values through corner")
    print("    - position_sequence: [(x,y), ...] - car position coordinates")  
    print("    - tangent_angles: [float, ...] - tangent angles at each position")
    print("    - normal_vectors: [(x,y), ...] - normal vectors at each position")
    print()
    print("  PHASE RATIOS (3 features):")
    print("    - entry_duration_ratio, apex_duration_ratio, exit_duration_ratio")
    
    print("\n[Example 4] Sequence-to-sequence learning use cases:")
    print("These geometry features can be used for:")
    print("  - Predicting optimal racing lines through corners")
    print("  - Learning corner entry/exit speed profiles") 
    print("  - Generating smooth trajectory sequences")
    print("  - Corner type classification and similarity analysis")
    print("  - Driver behavior modeling and coaching")
    
    print("\nExample completed!")


# Example of how to access geometry data programmatically
def analyze_geometry_data_example(results):
    """
    Example of how to programmatically access corner geometry data for ML
    """
    if not results or "error" in results:
        return
    
    print("\n[Data Access Example for ML]")
    
    # Access corner geometry features
    corner_geometries = results.get('corner_geometries', [])
    
    for i, corner_geometry in enumerate(corner_geometries[:2]):  # Show first 2 corners
        print(f"\nCorner {i+1}: {corner_geometry.get('corner_id', 'unknown')}")
        
        # Scalar geometry features (for feature vectors)
        scalars = corner_geometry.get('geometry_scalars', {})
        print(f"  Scalar features:")
        print(f"    Corner angle: {scalars.get('corner_angle', 0):.4f} radians")
        print(f"    Corner radius: {scalars.get('corner_radius', 0):.2f} meters")
        print(f"    Max curvature: {scalars.get('max_curvature', 0):.6f}")
        print(f"    Direction: {'Left' if scalars.get('direction', 0) < 0 else 'Right' if scalars.get('direction', 0) > 0 else 'Straight'}")
        print(f"    Complexity: {scalars.get('corner_complexity', 0):.4f}")
        
        # Sequence features (for RNNs, LSTMs, Transformers)
        sequences = corner_geometry.get('geometry_sequences', {})
        curvature_seq = sequences.get('curvature_progression', [])
        position_seq = sequences.get('position_sequence', [])
        
        print(f"  Sequence features:")
        print(f"    Curvature sequence length: {len(curvature_seq)} data points")
        print(f"    Position sequence length: {len(position_seq)} data points")
        
        if curvature_seq:
            print(f"    Curvature range: [{min(curvature_seq):.6f}, {max(curvature_seq):.6f}]")
        
        # Phase timing ratios (for temporal understanding)
        phase_ratios = corner_geometry.get('phase_ratios', {})
        print(f"  Phase ratios:")
        print(f"    Entry: {phase_ratios.get('entry_duration_ratio', 0):.3f}")
        print(f"    Apex: {phase_ratios.get('apex_duration_ratio', 0):.3f}")
        print(f"    Exit: {phase_ratios.get('exit_duration_ratio', 0):.3f}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(corner_geometry_example())
