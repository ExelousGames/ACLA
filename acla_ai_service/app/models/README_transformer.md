# Expert Action Transformer Model

This document describes the Expert Action Transformer model for predicting optimal racing actions from telemetry data in Assetto Corsa Competizione.

## Overview

The Expert Action Transformer is a neural network model that learns to predict expert-level racing actions (steering, throttle, brake) from current telemetry data. It uses the transformer architecture with attention mechanisms to understand complex patterns in racing data and suggest the sequence of actions needed to reach expert performance.

## Model Architecture

### Core Components

1. **ExpertActionTransformer**: Main transformer model
   - Input: Telemetry features (speed, g-forces, temperatures, etc.)
   - Output: Action sequences (steering angle, throttle, brake pressure)
   - Architecture: Encoder-decoder transformer with positional encoding

2. **TelemetryActionDataset**: Data preprocessing and sequence creation
   - Handles telemetry-to-action pair formatting
   - Creates training sequences with configurable lengths
   - Applies data scaling and normalization

3. **ExpertActionTrainer**: Training pipeline
   - Handles model training with validation
   - Implements early stopping and learning rate scheduling
   - Supports model checkpointing and recovery

### Key Features

- **Sequence-to-Sequence Learning**: Predicts future action sequences from current telemetry
- **Attention Mechanism**: Focuses on relevant telemetry patterns for each action prediction
- **Performance Scoring**: Additional output head for performance quality prediction
- **Autoregressive Generation**: Can generate action sequences of arbitrary length

## Usage in transformerLearning Method

The transformer model is integrated into the `transformerLearning` method in `full_dataset_ml_service.py`:

```python
async def transformerLearning(self, trackName: str):
    # 1. Load racing sessions data
    sessions = await backend_service.get_all_racing_sessions(trackName)
    
    # 2. Filter and process telemetry data
    # - Get top 1% fastest laps as expert demonstrations
    # - Use remaining laps for contextual learning
    
    # 3. Train expert imitation model
    imitation_learning = ImitateExpertLearningService()
    imitation_learning.train_ai_model(top_laps_telemetry_data)
    
    # 4. Enrich contextual data with corner identification and tire grip features
    enriched_contextual_data = await self.enriched_contextual_data(rest_laps)
    
    # 5. Compare telemetry with expert actions to find improvement sections
    training_and_expert_action = imitation_learning.compare_telemetry_with_expert(
        enriched_contextual_data, 5, 5
    )
    
    # 6. Train transformer model
    transformer_results = await self._train_expert_action_transformer(
        training_and_expert_action=training_and_expert_action,
        trackName=trackName
    )
```

## Data Flow

### Training Data Preparation

1. **Expert Demonstrations**: Top 1% fastest laps provide expert action patterns
2. **Contextual Enhancement**: Remaining laps are enriched with:
   - Corner identification features
   - Tire grip analysis
   - Performance scoring sections

3. **Training Pairs**: Each pair contains:
   - `telemetry_section`: Rising/peak/falling performance telemetry data
   - `expert_actions`: Corresponding optimal actions from expert demonstrations

### Model Training Process

1. **Data Extraction**: Convert training pairs to telemetry-action sequences
2. **Feature Processing**: 
   - Extract numeric telemetry features
   - Apply standardization scaling
   - Create input-output sequences

3. **Model Architecture Setup**:
   - Input features: Determined from telemetry data
   - Action features: 3 (steering, throttle, brake)
   - Model dimensions: Configurable (default: 256)

4. **Training Loop**:
   - Split data into train/validation sets
   - Use DataLoaders for batch processing
   - Apply early stopping and learning rate scheduling
   - Save best model checkpoints

### Prediction Process

The trained model can predict expert action sequences:

```python
async def predict_expert_actions(self, 
                               current_telemetry: List[Dict[str, Any]],
                               trackName: str,
                               sequence_length: int = 20,
                               temperature: float = 1.0) -> Dict[str, Any]:
```

**Parameters**:
- `current_telemetry`: Real-time telemetry data
- `trackName`: Track for model selection
- `sequence_length`: Number of future actions to predict
- `temperature`: Sampling temperature (lower = more conservative)

**Returns**:
- Predicted action sequence with confidence scores
- Performance predictions for each timestep

## Model Configuration

### Default Parameters

```python
model_params = {
    "d_model": 256,           # Model dimension
    "nhead": 8,               # Attention heads
    "num_encoder_layers": 6,  # Encoder layers
    "num_decoder_layers": 6,  # Decoder layers
    "dim_feedforward": 1024,  # Feedforward dimension
    "dropout": 0.1,           # Dropout rate
    "sequence_length": 50,    # Input sequence length
    "prediction_horizon": 20  # Output sequence length
}
```

### Training Parameters

```python
training_params = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_epochs": 50,
    "patience": 10,           # Early stopping patience
    "batch_size": 16          # Adaptive based on data size
}
```

## Output Format

### Training Results

```json
{
    "success": true,
    "model_path": "models/transformer_track_name.pth",
    "training_results": {
        "training_completed": true,
        "best_val_loss": 0.123,
        "total_epochs": 35
    },
    "dataset_info": {
        "total_sequences": 1250,
        "train_sequences": 1000,
        "val_sequences": 250,
        "input_features": 16
    },
    "model_info": {
        "parameters": 2456789,
        "model_size_mb": 9.87
    }
}
```

### Prediction Results

```json
{
    "success": true,
    "track_name": "Spa-Francorchamps",
    "predicted_actions": [
        {
            "timestep": 1,
            "steering_angle": 0.123,
            "throttle": 0.856,
            "brake": 0.0,
            "performance_score": 0.934
        }
    ],
    "prediction_info": {
        "sequence_length": 20,
        "temperature": 1.0,
        "input_features": 16,
        "input_timesteps": 50
    }
}
```

## Integration with Backend

The transformer model integrates with the backend storage system:

1. **Model Saving**: Trained models are saved to both local filesystem and backend database
2. **Model Loading**: Predictions load model weights and parameters from backend
3. **Metadata Storage**: Training metrics and model configuration stored as metadata

## Requirements

- PyTorch >= 1.9.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0

## Error Handling

The implementation includes robust error handling for:

- Missing PyTorch installation
- Insufficient training data
- Model loading failures
- Backend communication errors
- Data format incompatibilities

## Performance Considerations

- **Model Size**: Default configuration creates ~2.5M parameter model
- **Training Time**: Depends on data size, typically 10-50 epochs
- **Memory Usage**: Batch size automatically adjusted based on available data
- **Inference Speed**: Fast autoregressive generation for real-time use

## Future Enhancements

Potential improvements for the transformer model:

1. **Multi-Modal Inputs**: Incorporate track layout and weather data
2. **Hierarchical Attention**: Different attention scales for different time horizons
3. **Reinforcement Learning**: Fine-tune with RL for track-specific optimization
4. **Uncertainty Estimation**: Predict confidence intervals for actions
5. **Online Learning**: Adapt model parameters during racing sessions
