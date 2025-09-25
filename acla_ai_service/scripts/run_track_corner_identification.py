#!/usr/bin/env python3
"""
Standalone script to run analyze_track_cornering function only

This script initializes the TelemetryMLService and runs the analyze_track_cornering function
to analyze track corners and cornering phases from telemetry data across all racing sessions.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the Python path to import modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import the TelemetryMLService
from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService


async def main():
    """
    Main function to run track corner identification
    """
    print("=" * 60)
    print("ACLA Track Corner Identification Script")
    print("=" * 60)
    
    try:
        # Initialize the TelemetryMLService
        print("[INFO] Initializing TelemetryMLService...")
        models_directory = os.path.join(parent_dir, "models")
        ml_service = Full_dataset_TelemetryMLService(models_directory=models_directory)
        
        print("[INFO] Starting track corner identification...")
        print("[INFO] This will:")
        print("       - Retrieve all racing sessions from the backend")
        print("       - Extract telemetry data from each session")
        print("       - Analyze steering angles and car positions")
        print("       - Identify corner sections and cornering phases")
        print("       - Generate corner analysis summary")
        print("       - Save results to the backend")
        print()
        
        # Run the analyze_track_cornering function
        results = await ml_service.analyze_track_cornering('brands_hatch')

        print("\n" + "=" * 60)
        print("CORNER IDENTIFICATION COMPLETED")
        print("=" * 60)
        
        # Display results summary
        if results and "error" not in results:
            print(f"[SUCCESS] Track corner identification completed successfully!")
            print(f"[INFO] Results summary:",results)
            # Check for corner analysis results
            if 'corners' in results:
                corners = results['corners']
                print(f"\n[CORNER ANALYSIS]")
                print(f"  - Total corners identified: {len(corners)}")
                
                for corner_id, corner_data in corners.items():
                    print(f"  - Corner {corner_id}:")
                    if 'phases' in corner_data:
                        phases = corner_data['phases']
                        print(f"    * Phases: {', '.join(phases.keys())}")
                    if 'max_steering_angle' in corner_data:
                        print(f"    * Max steering angle: {corner_data['max_steering_angle']:.2f}°")
                    if 'duration' in corner_data:
                        print(f"    * Duration: {corner_data['duration']} data points")
            
            # Check for cornering phase statistics
            if 'phase_statistics' in results:
                stats = results['phase_statistics']
                print(f"\n[PHASE STATISTICS]")
                for phase, phase_stats in stats.items():
                    if isinstance(phase_stats, dict):
                        print(f"  - {phase}:")
                        if 'count' in phase_stats:
                            print(f"    * Occurrences: {phase_stats['count']}")
                        if 'avg_duration' in phase_stats:
                            print(f"    * Average duration: {phase_stats['avg_duration']:.1f} data points")
                        if 'avg_intensity' in phase_stats:
                            print(f"    * Average intensity: {phase_stats['avg_intensity']:.2f}")
            
            # Check for data statistics
            if 'data_statistics' in results:
                stats = results['data_statistics']
                print(f"\n[DATA STATISTICS]")
                print(f"  - Total telemetry records: {stats.get('total_records', 'N/A')}")
                print(f"  - Sessions analyzed: {stats.get('sessions_count', 'N/A')}")
                print(f"  - Lap sections analyzed: {stats.get('lap_sections', 'N/A')}")
                print(f"  - Features used: {stats.get('feature_count', 'N/A')}")
            
            # Check for track-specific insights
            if 'track_insights' in results:
                insights = results['track_insights']
                print(f"\n[TRACK INSIGHTS]")
                if 'most_challenging_corner' in insights:
                    print(f"  - Most challenging corner: {insights['most_challenging_corner']}")
                if 'average_corners_per_lap' in insights:
                    print(f"  - Average corners per lap: {insights['average_corners_per_lap']:.1f}")
                if 'corner_types' in insights:
                    corner_types = insights['corner_types']
                    print(f"  - Corner types distribution:")
                    for corner_type, count in corner_types.items():
                        print(f"    * {corner_type}: {count}")
            
        elif results and "error" in results:
            print(f"[ERROR] Corner identification failed: {results['error']}")
            return 1
        else:
            print("[WARNING] Corner identification completed but no results returned")
            
    except Exception as e:
        print(f"\n[ERROR] Corner identification failed: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        return 1
    
    print("\n" + "=" * 60)
    print("Script execution completed.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    # Run the async main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
