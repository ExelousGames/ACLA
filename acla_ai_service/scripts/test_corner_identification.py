#!/usr/bin/env python3
"""
Test script for Corner Identification Unsupervised Service

This script demonstrates how to use the corner identification service to:
1. Learn corner patterns from telemetry data
2. Extract corner characteristics as features
3. Enhance telemetry data with corner features
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from services.corner_identification_unsupervised_service import corner_identification_service


async def test_corner_identification():
    """Test the corner identification service"""
    
    print("=== Corner Identification Unsupervised Service Test ===\n")
    
    # Test parameters
    track_name = "monza"
    car_name = "porsche_991ii_gt3_r"
    
    print(f"Testing corner identification for:")
    print(f"Track: {track_name}")
    print(f"Car: {car_name}")
    print("-" * 50)
    
    # Test 1: Learn corner patterns for a track
    print("\n1. Learning corner patterns...")
    try:
        corner_results = await corner_identification_service.learn_track_corner_patterns(
            trackName=track_name,
            carName=car_name
        )
        
        if corner_results.get("success"):
            print(f"✅ Successfully learned corner patterns!")
            print(f"   - Total corners identified: {corner_results.get('total_corners_identified', 0)}")
            print(f"   - Corner clusters: {len(corner_results.get('corner_clusters', {}).get('clusters', []))}")
            
            # Display corner patterns summary
            corner_patterns = corner_results.get("corner_patterns", [])
            if corner_patterns:
                print(f"\nCorner Details:")
                for i, pattern in enumerate(corner_patterns[:5]):  # Show first 5 corners
                    char = pattern["characteristics"]
                    print(f"   Corner {i+1}:")
                    print(f"     - Type: {char.get('corner_type', 'unknown')}")
                    print(f"     - Direction: {char.get('corner_direction', 'unknown')}")
                    print(f"     - Duration: {char.get('total_corner_duration', 0):.1f} data points")
                    print(f"     - Max Steering: {char.get('apex_max_steering', 0):.3f}")
                    print(f"     - Speed Efficiency: {char.get('speed_efficiency', 0):.3f}")
                
                if len(corner_patterns) > 5:
                    print(f"   ... and {len(corner_patterns) - 5} more corners")
            
            # Display cluster summary
            clusters = corner_results.get("corner_clusters", {}).get("clusters", [])
            if clusters:
                print(f"\nCorner Type Clusters:")
                for cluster in clusters:
                    print(f"   - {cluster.get('type', 'unknown')}: {cluster.get('corner_count', 0)} corners")
        else:
            print(f"❌ Failed to learn corner patterns: {corner_results.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Error during corner pattern learning: {str(e)}")
    
    # Test 2: Get corner identification summary
    print("\n2. Getting corner identification summary...")
    try:
        summary = corner_identification_service.get_corner_identification_summary(track_name, car_name)
        
        if summary.get("success"):
            print(f"✅ Corner identification summary retrieved!")
            print(f"   - Total corners: {summary.get('total_corners', 0)}")
            print(f"   - Learning timestamp: {summary.get('learning_timestamp', 'N/A')}")
            
            corner_types = summary.get("corner_types", {})
            if corner_types:
                print(f"   - Corner types distribution:")
                for corner_type, count in corner_types.items():
                    print(f"     * {corner_type}: {count}")
        else:
            print(f"ℹ️  No existing corner profile found: {summary.get('error', 'Unknown')}")
    
    except Exception as e:
        print(f"❌ Error getting corner summary: {str(e)}")
    
    # Test 3: Test feature extraction with sample data
    print("\n3. Testing corner feature extraction...")
    try:
        # Create sample telemetry data for testing
        sample_telemetry = [
            {
                "Physics_steer_angle": 0.0,
                "Physics_speed_kmh": 200.0,
                "Physics_brake": 0.0,
                "Physics_gas": 1.0,
                "Physics_g_force_x": 0.0,
                "Physics_g_force_z": 0.0,
                "Graphics_normalized_car_position": 0.1
            },
            {
                "Physics_steer_angle": 0.2,
                "Physics_speed_kmh": 180.0,
                "Physics_brake": 0.3,
                "Physics_gas": 0.7,
                "Physics_g_force_x": -1.5,
                "Physics_g_force_z": 0.8,
                "Graphics_normalized_car_position": 0.2
            },
            {
                "Physics_steer_angle": 0.8,
                "Physics_speed_kmh": 120.0,
                "Physics_brake": 0.8,
                "Physics_gas": 0.0,
                "Physics_g_force_x": -2.5,
                "Physics_g_force_z": 1.2,
                "Graphics_normalized_car_position": 0.3
            },
            {
                "Physics_steer_angle": 1.2,
                "Physics_speed_kmh": 90.0,
                "Physics_brake": 0.2,
                "Physics_gas": 0.0,
                "Physics_g_force_x": -3.0,
                "Physics_g_force_z": 0.5,
                "Graphics_normalized_car_position": 0.4
            },
            {
                "Physics_steer_angle": 0.9,
                "Physics_speed_kmh": 110.0,
                "Physics_brake": 0.0,
                "Physics_gas": 0.5,
                "Physics_g_force_x": -2.2,
                "Physics_g_force_z": -0.5,
                "Graphics_normalized_car_position": 0.5
            }
        ]
        
        enhanced_telemetry = await corner_identification_service.extract_corner_features_for_telemetry(
            sample_telemetry, track_name, car_name
        )
        
        print(f"✅ Enhanced {len(enhanced_telemetry)} telemetry records with corner features")
        
        # Show some of the new corner features
        if enhanced_telemetry:
            first_record = enhanced_telemetry[0]
            corner_features = [key for key in first_record.keys() if key.startswith('corner_')]
            
            print(f"   - Added {len(corner_features)} corner features per record")
            print(f"   - Sample corner features: {corner_features[:10]}")  # Show first 10
            
            # Show corner feature values for a record that might be in a corner
            if len(enhanced_telemetry) > 2:
                corner_record = enhanced_telemetry[2]  # Third record (likely in corner)
                print(f"   - Example corner features for record 3:")
                print(f"     * Is in corner: {corner_record.get('is_in_corner', 0)}")
                print(f"     * Corner ID: {corner_record.get('corner_id', -1)}")
                print(f"     * Corner phase: {corner_record.get('corner_phase', 0)}")
                print(f"     * Corner type: {corner_record.get('corner_type_numeric', 0)}")
                print(f"     * Apex max steering: {corner_record.get('corner_apex_max_steering', 0):.3f}")
    
    except Exception as e:
        print(f"❌ Error during feature extraction test: {str(e)}")
    
    # Test 4: Cache management
    print("\n4. Testing cache management...")
    try:
        print("   - Clearing corner identification cache...")
        corner_identification_service.clear_corner_cache(track_name, car_name)
        print("   ✅ Cache cleared successfully")
    
    except Exception as e:
        print(f"❌ Error during cache management: {str(e)}")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    print("Starting Corner Identification Unsupervised Service Test...")
    
    try:
        asyncio.run(test_corner_identification())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
