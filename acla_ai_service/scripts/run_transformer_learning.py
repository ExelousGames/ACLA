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
        print("🚀 Starting Expert Imitation Transformer Learning Pipeline...")
        print("=" * 80)
        
        ml_service = Full_dataset_TelemetryMLService()
        results = await ml_service.StartImitateExpertPipeline('brands_hatch')
        
        # Display comprehensive results
        display_results(results)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

def display_results(results):
    """
    Display comprehensive results from the transformer learning pipeline
    
    Args:
        results: Dictionary containing all pipeline results
    """
    print("\n" + "=" * 80)
    print("🎯 TRANSFORMER LEARNING PIPELINE RESULTS")
    print("=" * 80)
    
    if not results.get('success', False):
        print(f"❌ Pipeline failed with error: {results.get('error', 'Unknown error')}")
        return
    
    print("✅ Pipeline completed successfully!")
    print(f"🏁 Track: {results.get('track_name', 'Unknown')}")
    
    # Data Processing Results
    print("\n📊 DATA PROCESSING SUMMARY:")
    print("-" * 40)
    print(f"  • Contextual data enriched: {results.get('contextual_data_enriched', 0):,} records")
    print(f"  • Training pairs generated: {results.get('training_pairs_generated', 0):,} pairs")
    print(f"  • Expert imitation training: {'✅ Completed' if results.get('expert_imitation_trained') else '❌ Failed'}")
    
    # Comparison Results
    comparison_results = results.get('comparison_results', {})
    if comparison_results:
        print("\n🔍 PERFORMANCE COMPARISON:")
        print("-" * 40)
        print(f"  • Total data points analyzed: {comparison_results.get('total_data_points', 0):,}")
        print(f"  • Overall performance score: {comparison_results.get('overall_score', 0.0):.4f}")
        print(f"  • Performance sections identified: {comparison_results.get('performance_sections_count', 0):,}")
    
    # Transformer Training Results
    transformer_results = results.get('transformer_training', {})
    if transformer_results and transformer_results.get('success'):
        print("\n🤖 TRANSFORMER MODEL TRAINING:")
        print("-" * 40)
        
        # Dataset Information
        dataset_info = transformer_results.get('dataset_info', {})
        print(f"  • Total sequences: {dataset_info.get('total_sequences', 0):,}")
        print(f"  • Training sequences: {dataset_info.get('train_sequences', 0):,}")
        print(f"  • Validation sequences: {dataset_info.get('val_sequences', 0):,}")
        print(f"  • Input features: {dataset_info.get('input_features', 0):,}")
        
        # Model Information
        model_info = transformer_results.get('model_info', {})
        if model_info:
            print(f"  • Model parameters: {model_info.get('parameters', 0):,}")
            print(f"  • Model size: {model_info.get('model_size_mb', 0.0):.2f} MB")
        
        # Training Results
        training_results = transformer_results.get('training_results', {})
        if training_results:
            print(f"  • Training completed: {'✅ Yes' if training_results.get('training_completed') else '❌ No'}")
            print(f"  • Total epochs trained: {training_results.get('total_epochs', 0)}")
            
            best_val_loss = training_results.get('best_val_loss')
            if best_val_loss is not None:
                print(f"  • Best validation loss: {best_val_loss:.6f}")
            
            # Training History Summary
            training_history = training_results.get('training_history', [])
            if training_history:
                print(f"  • Training history available: {len(training_history)} epochs")
                
                # Show first and last epoch metrics
                first_epoch = training_history[0] if training_history else {}
                last_epoch = training_history[-1] if training_history else {}
                
                if first_epoch and last_epoch:
                    first_train_loss = first_epoch.get('train_metrics', {}).get('total_loss', 0)
                    last_train_loss = last_epoch.get('train_metrics', {}).get('total_loss', 0)
                    
                    print(f"  • Initial training loss: {first_train_loss:.6f}")
                    print(f"  • Final training loss: {last_train_loss:.6f}")
                    
                    if first_train_loss > 0:
                        improvement = ((first_train_loss - last_train_loss) / first_train_loss) * 100
                        print(f"  • Training loss improvement: {improvement:.2f}%")
                    
                    # Validation metrics if available
                    first_val = first_epoch.get('val_metrics')
                    last_val = last_epoch.get('val_metrics')
                    if first_val and last_val:
                        first_val_loss = first_val.get('total_loss', 0)
                        last_val_loss = last_val.get('total_loss', 0)
                        print(f"  • Initial validation loss: {first_val_loss:.6f}")
                        print(f"  • Final validation loss: {last_val_loss:.6f}")
                        
                        if first_val_loss > 0:
                            val_improvement = ((first_val_loss - last_val_loss) / first_val_loss) * 100
                            print(f"  • Validation loss improvement: {val_improvement:.2f}%")
    else:
        transformer_error = transformer_results.get('error', 'Unknown error') if transformer_results else 'No transformer results'
        print(f"\n❌ TRANSFORMER TRAINING FAILED: {transformer_error}")
    
    print("\n" + "=" * 80)
    print("🏆 PIPELINE EXECUTION COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
