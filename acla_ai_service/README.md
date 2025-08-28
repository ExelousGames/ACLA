# ACLA AI Service

This AI service provides intelligent analysis capabilities for the ACLA racing application.

## Features

- **Dataset Analysis**: Automatically analyze racing session data
- **Natural Language Queries**: Ask questions about your racing data in plain English
- **Performance Insights**: Get insights about lap times, speeds, and racing patterns
- **Backend Integration**: Seamlessly integrates with ACLA backend to call functions
- **Real-time Processing**: Process racing data as it's uploaded

## API Endpoints

### Health Check
- `GET /health` - Check service health

### Dataset Management
- `POST /datasets/upload` - Upload a dataset for analysis
- `GET /datasets` - List all available datasets

### AI Queries
- `POST /query` - Ask natural language questions about datasets
- `POST /analyze` - Perform specific analysis on datasets

### Racing Session Analysis
- `POST /racing-session/analyze` - Analyze racing session data

### Backend Integration
- `POST /backend/call` - Call backend functions from AI service

## Environment Variables

- `BACKEND_URL`: URL of the ACLA backend service
- `OPENAI_API_KEY`: OpenAI API key for advanced AI features (optional)
- `AI_SERVICE_HOST`: Host to bind the service (default: 0.0.0.0)
- `AI_SERVICE_PORT`: Port to run the service (default: 8000)

## Usage Examples

### Asking Questions About Racing Data

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was my average lap time?",
    "dataset_id": "session_123"
  }'
```

### Analyzing Performance

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "session_123",
    "analysis_type": "performance"
  }'
```

## Integration with Backend

The AI service automatically receives racing session data when users upload sessions through the backend. It provides:

1. **Automatic Analysis**: Every uploaded session is automatically analyzed
2. **Question Answering**: Users can ask questions about their racing data
3. **Performance Insights**: Get detailed insights about racing performance
4. **Function Calling**: AI can call backend functions to retrieve additional data

## Data Processing

The service can process various types of racing data including:
- Lap times
- Speed data
- Acceleration/deceleration patterns
- Track position data
- Racing line analysis
