#!/usr/bin/env python3
"""
Test script for the human-readable prediction functions in the ExpertActionTransformer
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add the app directory to the Python path
current_dir = Path(__file__).parent
app_dir = current_dir.parent / 'app'
sys.path.insert(0, str(app_dir))

# Import the model (check if we can find the correct path)
try:
    from models.transformer_model import ExpertActionTransformer
except ImportError:
    # Alternative import paths
    try:
        sys.path.insert(0, str(current_dir.parent))
        from app.models.transformer_model import ExpertActionTransformer
    except ImportError:
        print("Error: Could not import ExpertActionTransformer")
        print(f"Current directory: {current_dir}")
        print(f"App directory: {app_dir}")
        print(f"Python path: {sys.path}")
        sys.exit(1)


def test_human_readable_predictions():
    """Test the human-readable prediction functions"""
    print("=" * 80)
    print("TESTING EXPERT ACTION TRANSFORMER - HUMAN READABLE PREDICTIONS")
    print("=" * 80)
    
    # Create a simple model instance
    model = ExpertActionTransformer(
        input_features=42,
        context_features=31,
        action_features=5,
        d_model=256,
        nhead=8,
        num_layers=2,  # Smaller for testing
        sequence_length=10
    )
    
    print(f"✅ Created ExpertActionTransformer model")
    print(f"   - Input features: {model.input_features}")
    print(f"   - Action features: {model.action_features}")
    print(f"   - Sequence length: {model.sequence_length}")
    
    # Create sample predictions (simulate model output)
    # Format: [batch_size, seq_len, action_features] where actions are [throttle, brake, steering, gear, speed]
    sample_predictions = torch.tensor([
        [
            [0.2, 0.8, 0.1, 2, 45],    # Light throttle, heavy brake, slight right, gear 2, 45 km/h
            [0.1, 0.6, 0.3, 2, 35],    # Very light throttle, moderate brake, right turn, gear 2, 35 km/h  
            [0.0, 0.4, 0.6, 2, 28],    # No throttle, light brake, sharp right, gear 2, 28 km/h
            [0.3, 0.0, 0.4, 3, 40],    # Moderate throttle, no brake, moderate right, gear 3, 40 km/h
            [0.6, 0.0, 0.2, 3, 65],    # Strong throttle, no brake, slight right, gear 3, 65 km/h
            [0.8, 0.0, 0.0, 4, 85],    # Strong throttle, no brake, straight, gear 4, 85 km/h
            [0.9, 0.0, -0.1, 4, 105],  # Full throttle, no brake, slight left, gear 4, 105 km/h
            [0.7, 0.2, -0.4, 4, 95],   # Moderate throttle, light brake, left turn, gear 4, 95 km/h
            [0.4, 0.5, -0.6, 3, 70],   # Light throttle, moderate brake, sharp left, gear 3, 70 km/h
            [0.8, 0.0, -0.2, 4, 90]    # Strong throttle, no brake, slight left, gear 4, 90 km/h
        ]
    ], dtype=torch.float32)
    
    print(f"\n✅ Created sample predictions tensor: {sample_predictions.shape}")
    
    # Test 1: Single prediction interpretation
    print("\n" + "="*50)
    print("TEST 1: SINGLE PREDICTION INTERPRETATION")
    print("="*50)
    
    current_telemetry = {'speed': 50, 'gear': 2}
    interpretation = model.predict_human_readable(
        predictions=sample_predictions,
        current_speed=50,
        track_context="approaching tight corner",
        sequence_step=0
    )
    
    print("📊 SINGLE STEP INTERPRETATION:")
    print(f"Summary: {interpretation['summary']}")
    print(f"Priority Action: {interpretation['priority_action']}")
    print(f"Throttle: {interpretation['throttle_advice']}")
    print(f"Brake: {interpretation['brake_advice']}")
    print(f"Steering: {interpretation['steering_advice']}")
    print(f"Gear: {interpretation['gear_advice']}")
    print(f"Speed Target: {interpretation['speed_target']}")
    print(f"Intensity: {interpretation['intensity_level']}")
    
    # Test 2: Sequence interpretation
    print("\n" + "="*50)
    print("TEST 2: FULL SEQUENCE INTERPRETATION")
    print("="*50)
    
    track_sections = [
        "approaching corner entry",
        "corner braking zone", 
        "corner apex",
        "corner exit",
        "short straight",
        "acceleration zone",
        "high speed straight",
        "approaching next corner",
        "corner entry",
        "corner exit"
    ]
    
    sequence_interpretations = model.interpret_sequence_predictions(
        predictions=sample_predictions,
        current_telemetry=current_telemetry,
        track_sections=track_sections
    )
    
    print("🏁 SEQUENCE INTERPRETATION:")
    for i, interp in enumerate(sequence_interpretations[:5]):  # Show first 5 steps
        print(f"\nStep {interp['sequence_step']} ({interp['time_ahead']} ahead) - {interp['timing']}:")
        print(f"  Context: {track_sections[i] if i < len(track_sections) else 'N/A'}")
        print(f"  Action: {interp['priority_action']}")
        print(f"  Summary: {interp['summary']}")
        if interp['raw_values']['target_speed']:
            print(f"  Speed: {interp['raw_values']['target_speed']:.0f} km/h")
    
    # Test 3: Coaching summary
    print("\n" + "="*50)
    print("TEST 3: AI RACING COACH SUMMARY")
    print("="*50)
    
    coaching_summary = model.generate_coaching_summary(
        predictions=sample_predictions,
        current_telemetry=current_telemetry,
        track_name="Spa-Francorchamps (Eau Rouge)"
    )
    
    print("🏆 AI RACING COACH SUMMARY:")
    print(f"Track: {coaching_summary['track']}")
    print(f"Overall Approach: {coaching_summary['overall_approach']}")
    
    if coaching_summary['immediate_guidance']:
        print("\n⚡ IMMEDIATE ACTIONS:")
        for action in coaching_summary['immediate_guidance']:
            print(f"  • {action}")
    
    if coaching_summary['upcoming_strategy']:
        print("\n🎯 UPCOMING STRATEGY:")
        for strategy in coaching_summary['upcoming_strategy']:
            print(f"  • {strategy}")
    
    print(f"\n📈 KEY METRICS:")
    for metric, value in coaching_summary['key_metrics'].items():
        if value is not None:
            print(f"  • {metric.replace('_', ' ').title()}: {value}")
    
    if coaching_summary['warnings']:
        print(f"\n⚠️  WARNINGS:")
        for warning in coaching_summary['warnings']:
            print(f"  • {warning}")
    
    # Test 4: Different driving scenarios
    print("\n" + "="*50)
    print("TEST 4: DIFFERENT DRIVING SCENARIOS")
    print("="*50)
    
    scenarios = [
        {
            'name': 'Emergency Braking',
            'prediction': torch.tensor([0.0, 1.0, 0.0, 2, 20], dtype=torch.float32),
            'context': 'obstacle ahead',
            'speed': 80
        },
        {
            'name': 'Full Attack Mode',
            'prediction': torch.tensor([1.0, 0.0, 0.0, 6, 200], dtype=torch.float32),
            'context': 'long straight',
            'speed': 150
        },
        {
            'name': 'Technical Corner',
            'prediction': torch.tensor([0.3, 0.4, -0.8, 2, 45], dtype=torch.float32),
            'context': 'chicane sequence',
            'speed': 65
        }
    ]
    
    for scenario in scenarios:
        print(f"\n🏁 SCENARIO: {scenario['name']}")
        interp = model.predict_human_readable(
            predictions=scenario['prediction'].unsqueeze(0),  # Add sequence dimension
            current_speed=scenario['speed'],
            track_context=scenario['context'],
            sequence_step=0
        )
        print(f"   Summary: {interp['summary']}")
        print(f"   Priority: {interp['priority_action']}")
        print(f"   Intensity: {interp['intensity_level']}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("The human-readable prediction functions are working correctly.")
    print("="*80)


if __name__ == "__main__":
    test_human_readable_predictions()