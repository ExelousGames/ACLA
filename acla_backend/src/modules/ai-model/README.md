# AI Model Module

This module provides functionality for managing trained AI models for racing performance analysis. It allows users to create, store, and incrementally train personalized AI models for specific tracks.

## Features

### Model Management
- **Create new AI models** from racing session data
- **Store trained models** in MongoDB with comprehensive metadata
- **Version control** for model iterations
- **Incremental training** to improve existing models
- **Model validation** and performance tracking

### Supported Model Types
- `lap_time_prediction`: Predicts optimal lap times based on telemetry data
- `sector_analysis`: Analyzes sector-wise performance and improvements
- `setup_optimization`: Recommends optimal car setup configurations

## API Endpoints

### Model CRUD Operations

#### Create Model
```
POST /ai-model
```
Creates a new AI model for a user on a specific track.

**Body:**
```json
{
  "trackName": "spa-francorchamps",
  "modelName": "My Spa Model",
  "modelVersion": "1.0.0",
  "modelMetadata": {
    "modelType": "lap_time_prediction",
    "trainingSessionsCount": 5,
    "features": ["speed", "throttle", "brake", "gear"]
  },
  "trainingSessionIds": ["session1", "session2", "session3"],
  "isActive": true
}
```

#### Get User Models
```
GET /ai-model/user/{trackName}?modelType=lap_time_prediction&activeOnly=true
```
Retrieves all models for a user on a specific track.

#### Get Active Model
```
GET /ai-model/active/{trackName}/{modelType}
```
Gets the currently active model for a user/track/type combination.

#### Update Model
```
PUT /ai-model/{id}
```
Updates an existing model's metadata or activates/deactivates it.

#### Delete Model
```
DELETE /ai-model/{id}
```
Removes a model from the database.

### Model Training

#### Train New Model
```
POST /ai-model/train-new
```
Trains a new model from scratch using specified racing sessions.

**Body:**
```json
{
  "trackName": "spa-francorchamps",
  "modelType": "lap_time_prediction", 
  "modelName": "Enhanced Spa Model",
  "sessionIds": ["session1", "session2", "session3", "session4"],
  "trainingParameters": {
    "algorithm": "random_forest",
    "max_depth": 10,
    "n_estimators": 100
  }
}
```

#### Incremental Training
```
POST /ai-model/incremental-training
```
Adds new racing session data to an existing model.

**Body:**
```json
{
  "modelId": "model_id_here",
  "newSessionIds": ["new_session1", "new_session2"],
  "trainingParameters": {
    "learning_rate": 0.01
  },
  "validateModel": true
}
```

### Model Usage

#### Make Predictions
```
POST /ai-model/predict
```
Uses a trained model to make predictions on new data.

**Body:**
```json
{
  "modelId": "model_id_here",
  "inputData": {
    "speed": 180,
    "throttle": 0.8,
    "brake": 0.0,
    "gear": 6,
    "track_position": 0.5
  },
  "predictionOptions": {
    "include_confidence": true,
    "include_feature_importance": true
  }
}
```

#### Performance History
```
GET /ai-model/performance-history/{trackName}/{modelType}
```
Gets the performance evolution of models over time.

## Database Schema

The AI models are stored in MongoDB with the following structure:

```typescript
{
  userId: ObjectId,             // Reference to UserInfo document
  trackName: string,           // Track the model is trained for
  modelName: string,           // Human-readable model name
  modelVersion: string,        // Version number (e.g., "1.2.3")
  modelData: object,           // Serialized model weights/parameters
  modelMetadata: {
    trainingSessionsCount: number,
    lastTrainingDate: Date,
    performanceMetrics: object,
    modelType: string,
    accuracy?: number,
    mse?: number,
    features: string[],
    hyperparameters?: object
  },
  trainingSessionIds: string[], // Sessions used for training
  isActive: boolean,           // Whether this is the active model
  description?: string,        // Optional description
  validationResults?: object,  // Cross-validation results
  featureImportance?: object,  // Feature importance scores
  modelSize?: number,          // Model size in bytes
  trainingDuration?: number,   // Training time in ms
  createdAt: Date,            // Auto-generated timestamp
  updatedAt: Date             // Auto-generated timestamp
}
```

## Integration with AI Service

The module communicates with the Python AI service for:
- Model training and incremental training
- Making predictions
- Model validation
- Performance metrics calculation

## Usage Example

1. **Train initial model:**
   ```typescript
   // Train a new lap time prediction model for Spa
   const newModel = await aiModelService.trainNewModel({
     trackName: 'spa-francorchamps',
     modelType: 'lap_time_prediction',
     modelName: 'My First Spa Model',
     sessionIds: ['session1', 'session2', 'session3']
   });
   ```

2. **Add more training data:**
   ```typescript
   // After collecting more racing sessions
   const updatedModel = await aiModelService.incrementalTraining({
     modelId: newModel.id,
     newSessionIds: ['session4', 'session5']
   });
   ```

3. **Make predictions:**
   ```typescript
   // Predict lap time for new telemetry data
   const prediction = await aiModelService.makePrediction({
     modelId: updatedModel.id,
     inputData: telemetryData
   });
   ```

## Security

- All endpoints are protected with JWT authentication
- Users can only access their own models
- Model data is encrypted when stored
- Rate limiting is applied to prevent abuse

## Future Enhancements

- Model sharing between users (with permission)
- Automated model retraining based on new data
- Model comparison and A/B testing
- Export/import functionality for models
- Real-time model performance monitoring
