# AI Model System Migration Guide

This document explains how to use the new AI Model system that allows saving and incrementally training personalized AI models for each user and track combination.

## Overview

The new AI Model system provides:
1. **Persistent Model Storage**: Models are saved in MongoDB and can be retrieved later
2. **Incremental Training**: Existing models can be updated with new racing session data
3. **Version Control**: Each model update creates a new version while preserving history
4. **Track-Specific Models**: Each user can have different models for different tracks
5. **Multiple Model Types**: Support for lap time prediction, sector analysis, and setup optimization

## Key Components

### Backend Module Structure
```
acla_backend/src/modules/ai-model/
├── ai-model.controller.ts     # REST API endpoints
├── ai-model.service.ts        # Business logic
├── ai-model.module.ts         # Module configuration
└── README.md                  # Documentation

acla_backend/src/
├── schemas/ai-model.schema.ts # MongoDB schema
└── dto/ai-model.dto.ts        # Data transfer objects
```

### AI Service Endpoints (Python)
```
/model/train                   # Train new model from scratch
/model/incremental-training    # Update existing model
/model/predict                 # Make predictions
/model/validate               # Validate model performance
/model/metrics/{id}           # Get model metrics
```

## Usage Scenarios

### 1. First Time User - Create Initial Model

When a user uploads their first racing sessions for a track:

```typescript
// 1. User uploads racing sessions (existing workflow)
// 2. System automatically creates initial model
const initialModel = await aiModelService.createModel({
  userId: user.id,
  trackName: 'spa-francorchamps',
  modelName: 'My Spa Model v1',
  modelVersion: '1.0.0',
  modelMetadata: {
    modelType: 'lap_time_prediction',
    trainingSessionsCount: 3,
    features: ['speed', 'throttle', 'brake', 'gear'],
    // ... other metadata
  },
  trainingSessionIds: ['session1', 'session2', 'session3'],
  isActive: true
});
```

### 2. Returning User - Incremental Training

When a user uploads new sessions for a track they already have a model for:

```typescript
// 1. Check for existing active model
const activeModel = await aiModelService.findActiveModel(
  user.id, 
  'spa-francorchamps', 
  'lap_time_prediction'
);

// 2. If model exists, perform incremental training
if (activeModel) {
  const updatedModel = await aiModelService.incrementalTraining({
    modelId: activeModel.id,
    newSessionIds: ['new_session_1', 'new_session_2'],
    validateModel: true
  });
  
  // Model version automatically increments (e.g., 1.0.0 -> 1.0.1)
  console.log('Updated to version:', updatedModel.modelVersion);
}
```

### 3. Making Predictions

Use a trained model to predict performance:

```typescript
const prediction = await aiModelService.makePrediction({
  modelId: activeModel.id,
  inputData: {
    speed: 180,
    throttle: 0.85,
    brake: 0.0,
    gear: 6,
    track_position: 0.5
  },
  predictionOptions: {
    include_confidence: true,
    include_feature_importance: true
  }
});

console.log('Predicted lap time:', prediction.predictions.lap_time);
console.log('Confidence:', prediction.confidence);
```

## Integration Points

### 1. Racing Session Upload (Automatic)

The system automatically integrates with the existing racing session upload process:

```typescript
// In racing-session.controller.ts (already implemented)
@Post('upload/complete')
async completeUpload(...) {
  // ... existing upload logic ...
  
  // Check for existing model and update automatically
  const activeModel = await this.aiModelService.findActiveModel(
    userEmail, mapName, 'lap_time_prediction'
  );
  
  if (activeModel) {
    // Automatically perform incremental training
    await this.aiModelService.incrementalTraining({
      modelId: activeModel.id,
      newSessionIds: [uploadId]
    });
  }
}
```

### 2. Frontend Integration

Frontend can interact with the AI models through new endpoints:

```typescript
// Get user's models for a track
GET /ai-model/user/spa-francorchamps?activeOnly=true

// Train a new model manually
POST /ai-model/train-new
{
  "trackName": "spa-francorchamps",
  "modelType": "lap_time_prediction",
  "sessionIds": ["session1", "session2"]
}

// Make predictions
POST /ai-model/predict
{
  "modelId": "model_id",
  "inputData": { /* telemetry data */ }
}
```

## Database Impact

### New Collection: `aimodels`

The system adds a new MongoDB collection to store AI models:

```javascript
// Example document
{
  "_id": ObjectId("..."),
  "userId": "user123",
  "trackName": "spa-francorchamps", 
  "modelName": "My Spa Model",
  "modelVersion": "1.2.3",
  "modelData": { /* serialized model */ },
  "modelMetadata": {
    "modelType": "lap_time_prediction",
    "trainingSessionsCount": 15,
    "accuracy": 0.85,
    "mse": 0.12,
    "features": ["speed", "throttle", "brake"]
  },
  "trainingSessionIds": ["session1", "session2", "..."],
  "isActive": true,
  "createdAt": ISODate("..."),
  "updatedAt": ISODate("...")
}
```

### Indexes

Optimized queries with compound indexes:
- `{userId: 1, trackName: 1, modelType: 1, isActive: 1}`
- `{userId: 1, trackName: 1, modelVersion: -1}`

## Performance Considerations

### Model Storage
- Models are stored as serialized objects in MongoDB
- Large models (>16MB) may require GridFS (future enhancement)
- Model compression can be added for optimization

### Incremental Training
- Only new session data is sent to AI service
- Existing model weights are reused and updated
- Training time scales with new data size, not total model size

### Caching
- Active models can be cached in memory for faster predictions
- Model metadata is lightweight for quick queries

## Security & Privacy

### Access Control
- Users can only access their own models
- JWT authentication required for all endpoints
- Models are tied to specific user IDs

### Data Privacy
- Model data contains no personal information
- Only racing telemetry features are stored
- Models can be deleted at any time

## Monitoring & Metrics

### Model Performance Tracking
```typescript
// Get model performance history
GET /ai-model/performance-history/spa-francorchamps/lap_time_prediction

// Response includes accuracy trends over versions
{
  "versions": [
    {"version": "1.0.0", "accuracy": 0.75, "createdAt": "..."},
    {"version": "1.0.1", "accuracy": 0.82, "createdAt": "..."},
    {"version": "1.0.2", "accuracy": 0.85, "createdAt": "..."}
  ]
}
```

### Usage Analytics
- Track model creation/update frequency
- Monitor prediction request patterns
- Measure model performance improvements

## Migration Steps

### 1. Backend Setup
1. ✅ Add AI Model module to backend
2. ✅ Update app.module.ts to include AiModelModule
3. ✅ Integrate with racing session upload process
4. ✅ Add new endpoints to AI service (Python)

### 2. Database Setup
1. MongoDB will automatically create the new collection
2. Indexes will be created on first model save
3. No migration scripts needed for existing data

### 3. Frontend Integration (Future)
1. Add model management UI
2. Display model performance metrics
3. Manual model training interface
4. Prediction visualization

### 4. Testing
1. Test model creation with sample data
2. Verify incremental training workflow
3. Test prediction accuracy
4. Performance testing with large datasets

## Future Enhancements

### Advanced Features
- **Model Sharing**: Allow users to share models with others
- **Auto-retraining**: Automatically retrain models based on performance degradation
- **Model Comparison**: A/B testing between different model versions
- **Export/Import**: Save/load models to/from files

### Optimization
- **Model Compression**: Reduce storage requirements
- **Distributed Training**: Scale training across multiple instances
- **Real-time Predictions**: Stream predictions during live racing
- **Edge Deployment**: Deploy models to edge devices for low-latency predictions

This system provides a solid foundation for personalized AI that improves over time as users contribute more racing data.
