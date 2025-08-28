# ACLA AI Service Setup and Testing

## Quick Start

1. **Start the services in development mode:**
   ```bash
   cd f:\Git-Repo\ACLA
   docker-compose -f docker-compose.dev.yaml up --build
   ```

2. **Test the AI service:**
   ```bash
   cd acla_ai_service
   python test_ai_service.py
   ```

## Services Overview

After running `docker-compose up`, you'll have:

- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:7001  
- **AI Service**: http://localhost:8000
- **MongoDB**: http://localhost:27017
- **Mongo Express**: http://localhost:8081

## AI Service Features

### 1. Dataset Analysis
The AI service can analyze racing datasets and provide:
- Basic statistics
- Performance metrics
- Racing patterns detection
- Sector analysis
- Performance scoring

### 2. Natural Language Queries
Users can ask questions about their racing data:
- "What was my average lap time?"
- "How consistent was my speed?"
- "Which sector was my fastest?"
- "What's my performance score?"

### 3. Advanced Analytics
- **Pattern Detection**: Uses machine learning to detect patterns in racing behavior
- **Performance Scoring**: Comprehensive scoring system with recommendations
- **Sector Analysis**: Track divided into sectors for detailed analysis
- **Optimal Lap Prediction**: Predicts potential lap time improvements

### 4. Backend Integration
The AI service integrates seamlessly with your backend:
- Automatic analysis when racing sessions are uploaded
- Function calling capabilities to retrieve additional data
- User authentication and authorization support

## API Examples

### Query Racing Data
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was my best lap time and how does it compare to my average?",
    "dataset_id": "your_session_id"
  }'
```

### Get Performance Score
```bash
curl -X POST "http://localhost:8000/racing-session/performance-score" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id"
  }'
```

### Detect Racing Patterns
```bash
curl -X POST "http://localhost:8000/racing-session/patterns" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id"
  }'
```

## Frontend Integration

To integrate with your frontend, you can call the backend endpoints:

```javascript
// Ask a question about racing data
const response = await fetch('/api/ai/racing-session/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${userToken}`
  },
  body: JSON.stringify({
    session_id: 'session_123',
    question: 'How can I improve my lap times?'
  })
});

// Get performance score
const scoreResponse = await fetch('/api/ai/racing-session/performance-score', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${userToken}`
  },
  body: JSON.stringify({
    session_id: 'session_123'
  })
});
```

## Customization

### Adding Custom Analysis
To add custom analysis functions:
1. Extend the `AdvancedRacingAnalyzer` class in `advanced_analyzer.py`
2. Add new endpoints in `main.py`
3. Add corresponding methods in the backend AI service client

### Adding New Data Types
The service can be extended to handle different types of racing data by:
1. Updating the data processing pipeline
2. Adding new analysis methods
3. Extending the natural language processing capabilities

## Troubleshooting

### Common Issues

1. **AI Service not starting:**
   - Check if port 8000 is available
   - Verify Python dependencies are installed
   - Check Docker logs: `docker logs acla_ai_service_c`

2. **Backend can't connect to AI service:**
   - Verify the AI_SERVICE_URL environment variable
   - Check if both services are in the same Docker network
   - Test connectivity: `docker exec acla_backend_c ping acla_ai_service_c`

3. **Analysis failing:**
   - Check data format - ensure it's properly structured
   - Verify required columns are present
   - Check AI service logs for detailed error messages

### Performance Optimization

- For large datasets, consider implementing data pagination
- Use caching for frequently requested analyses
- Consider implementing asynchronous processing for complex analyses
