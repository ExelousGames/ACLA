#!/usr/bin/env python3
import asyncio
from pathlib import Path
import sys

# Add the parent directory to the Python path to import modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService

async def main():
    try:
        ml_service = Full_dataset_TelemetryMLService()
        results = await ml_service.transformerLearning('brands_hatch')
        
        # Print formatted results
        print("\n" + "="*80)
        print("                    TRANSFORMER LEARNING RESULTS")
        print("="*80)
        
        print(f"✅ Success: {results.get('success', False)}")
        print(f"🏁 Track: {results.get('track_name', 'Unknown')}")
        print(f"📊 Training Pairs Generated: {results.get('training_pairs_generated', 0)}")
        print(f"📈 Data Points Enriched: {results.get('contextual_data_enriched', 0)}")
        print(f"🎯 Expert Imitation Trained: {results.get('expert_imitation_trained', False)}")
        
        # Transformer training details
        transformer_results = results.get('transformer_training', {})
        if transformer_results.get('success'):
            training_info = transformer_results.get('training_results', {})
            dataset_info = transformer_results.get('dataset_info', {})
            model_info = transformer_results.get('model_info', {})
            
            print(f"\n🤖 TRANSFORMER MODEL TRAINING:")
            print(f"   ✅ Training Success: {transformer_results.get('success')}")
            print(f"   📁 Model Saved: {transformer_results.get('model_path', 'N/A')}")
            
            # Handle parameters with proper formatting
            parameters = model_info.get('parameters', 'N/A')
            if isinstance(parameters, (int, float)):
                print(f"   🔢 Model Parameters: {parameters:,}")
            else:
                print(f"   � Model Parameters: {parameters}")
                
            # Handle model size with proper formatting
            model_size = model_info.get('model_size_mb', 'N/A')
            if isinstance(model_size, (int, float)):
                print(f"   💾 Model Size: {model_size:.2f} MB")
            else:
                print(f"   � Model Size: {model_size}")
                
            print(f"   �📚 Training Sequences: {dataset_info.get('train_sequences', 'N/A')}")
            print(f"   🔍 Validation Sequences: {dataset_info.get('val_sequences', 'N/A')}")
            print(f"   🎯 Input Features: {dataset_info.get('input_features', 'N/A')}")
            
            # Handle validation loss with proper formatting
            val_loss = training_info.get('best_val_loss', 'N/A')
            if isinstance(val_loss, (int, float)):
                print(f"   📉 Best Validation Loss: {val_loss:.8f}")
            else:
                print(f"   📉 Best Validation Loss: {val_loss}")
                
            print(f"   🔄 Total Epochs: {training_info.get('total_epochs', 'N/A')}")
        else:
            print(f"\n❌ TRANSFORMER TRAINING FAILED:")
            print(f"   Error: {transformer_results.get('error', 'Unknown error')}")
        
        # Comparison results
        comparison = results.get('comparison_results', {})
        if comparison:
            print(f"\n📊 PERFORMANCE ANALYSIS:")
            print(f"   🎯 Overall Score: {comparison.get('overall_score', 0):.4f}")
            print(f"   📍 Performance Sections: {comparison.get('performance_sections_count', 0)}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
