# GridFS Multi-Collection Integration

This implementation provides GridFS support with multi-collection capabilities for the ACLA backend.

## Features

- **Multi-Collection Support**: Store files in different GridFS collections/buckets
- **Docker-Ready**: Properly configured for Docker environments with connection health checks
- **JSON Helpers**: Built-in methods for storing and retrieving JSON data
- **Model Data Storage**: AI model data is stored in GridFS instead of MongoDB documents
- **Type Safety**: Centralized bucket constants with TypeScript support

## Collections/Buckets

The system uses predefined buckets defined in `GRIDFS_BUCKETS`:

```typescript
export const GRIDFS_BUCKETS = {
    AI_MODELS: 'ai_models',           // AI model data and weights
    TELEMETRY_DATA: 'telemetry_data', // Raw telemetry data from racing sessions
    TRAINING_DATASETS: 'training_datasets', // Training datasets for machine learning
    MODEL_BACKUPS: 'model_backups'    // Backup copies of AI models
} as const;
```

## Usage Examples

### Basic File Operations

```typescript
import { GridFSService, GRIDFS_BUCKETS } from '../gridfs/gridfs.service';

// Upload a file to the default AI_MODELS bucket
const fileId = await gridfsService.uploadJSON(modelData, 'model.json', metadata);

// Upload to a specific bucket
const fileId = await gridfsService.uploadJSON(data, 'filename.json', metadata, GRIDFS_BUCKETS.TELEMETRY_DATA);

// Download from default bucket
const data = await gridfsService.downloadJSON(fileId);

// Download from specific bucket
const data = await gridfsService.downloadJSON(fileId, GRIDFS_BUCKETS.TELEMETRY_DATA);

// Delete file from specific bucket
await gridfsService.deleteFile(fileId, GRIDFS_BUCKETS.AI_MODELS);
```

### Bucket Validation

```typescript
// Check if bucket name is valid
const isValid = GridFSService.isValidBucketName('ai_models'); // true

// Get bucket type
const bucketType = GridFSService.getBucketType('ai_models'); // 'AI_MODELS'

// Get all available bucket names
const allBuckets = GridFSService.getAllBucketNames(); // ['ai_models', 'telemetry_data', ...]
```

### AI Model Service Examples

```typescript
// Create model with GridFS storage
const model = await aiModelService.create({
  trackName: 'Silverstone',
  carName: 'F1_2023',
  modelType: 'lap_time_prediction',
  modelData: { weights: [...], config: {...} },
  // ... other fields
});

// Get model with data loaded from GridFS
const modelWithData = await aiModelService.findOneWithModelData(modelId);

// Save training dataset
const datasetFileId = await aiModelService.saveTrainingDataset(
  trainingData,
  'silverstone_f1_training.json',
  { sessions: 100, laps: 2500 }
);

// Create backup
const backupFileId = await aiModelService.createModelBackup(modelId);

// Restore from backup
const restoredModel = await aiModelService.restoreModelFromBackup(backupFileId);
```

### Health Check and Statistics

```typescript
// Check GridFS health
const isHealthy = await gridfsService.isHealthy('ai_models');

// Get statistics for a specific bucket
const stats = await gridfsService.getStats('ai_models');

// Get statistics for all buckets
const allStats = await gridfsService.getAllBucketsStats();
```

## API Endpoints

### Standard AI Model Operations
- `GET /ai-model` - List all models
- `POST /ai-model` - Create new model
- `GET /ai-model/:id` - Get model metadata only
- `GET /ai-model/:id/with-data` - Get model with data from GridFS
- `PUT /ai-model/:id` - Update model
- `DELETE /ai-model/:id` - Delete model and associated GridFS files

### GridFS Management
- `GET /ai-model/gridfs/health` - Check GridFS health
- `GET /ai-model/gridfs/stats` - Get GridFS statistics
- `POST /ai-model/:id/backup` - Create model backup
- `POST /ai-model/restore/:backupFileId` - Restore model from backup

### Active Model Operations
- `GET /ai-model/active/:track/:car/:type` - Get active model metadata
- `GET /ai-model/active/:track/:car/:type/with-data` - Get active model with data
- `POST /ai-model/:id/activate` - Activate specific model

## Docker-Specific Features

### Connection Handling
- **Retry Logic**: Automatically retries initialization if MongoDB isn't ready yet
- **Graceful Shutdown**: Properly cleans up resources when container stops
- **Health Checks**: Comprehensive health checks for container orchestration
- **Connection Monitoring**: Real-time connection state monitoring

### Enhanced Error Handling
- **Timeout Protection**: Upload/download operations have timeout protection
- **Better Error Messages**: More descriptive error messages with bucket context
- **Connection Recovery**: Automatic reconnection handling for Docker restarts

### Debugging and Monitoring
- **Service Info**: Detailed connection and service status information
- **Reinitialization**: Manual GridFS reinitialization for troubleshooting
- **Enhanced Logging**: Better logging for Docker container debugging

## Docker API Endpoints

### Enhanced GridFS Management
- `GET /ai-model/gridfs/health` - Comprehensive health check
- `GET /ai-model/gridfs/stats` - GridFS statistics for all buckets
- `GET /ai-model/gridfs/info` - Detailed service and connection information
- `POST /ai-model/gridfs/reinitialize` - Manual GridFS reinitialization

## Benefits

1. **Scalability**: Large model files don't bloat MongoDB documents
2. **Performance**: GridFS handles large file streaming efficiently  
3. **Flexibility**: Different file types can be stored in separate collections
4. **Backup/Recovery**: Easy model versioning and backup capabilities
5. **Docker-Ready**: 
   - Handles connection timing issues in containerized environments
   - Automatic retry logic for service startup
   - Graceful shutdown handling
   - Comprehensive health checks for container orchestration
   - Manual recovery options for troubleshooting

## Configuration

The GridFS service automatically initializes with the MongoDB connection from NestJS. Enhanced Docker support includes:

### Environment Variables
- Standard MongoDB environment variables are used
- Connection timeout and retry logic is built-in
- No additional configuration required for Docker environments

### Docker Health Checks
The service provides multiple health check endpoints that can be used in Docker Compose or Kubernetes:

```yaml
# docker-compose.yml
services:
  backend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/ai-model/gridfs/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## File Naming Convention

Files are automatically named with the pattern:
```
{type}_{trackName}_{carName}_{modelType}_{timestamp}.json
```

Example: `model_Silverstone_F1_2023_lap_time_prediction_1641234567890.json`
