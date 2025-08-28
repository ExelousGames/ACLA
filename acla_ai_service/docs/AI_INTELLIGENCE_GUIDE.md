# ACLA AI Service - Intelligent Racing Data Analysis

## 🧠 AI-Powered Features

The ACLA AI Service now includes advanced artificial intelligence capabilities that can understand natural language questions and provide intelligent insights about your racing data.

## 🚀 What's New

### Natural Language Understanding
- Ask questions in plain English about your racing data
- Get intelligent responses powered by OpenAI GPT-4
- Automatic function calling based on your questions

### AI-Powered Analysis
- Intelligent performance insights
- Personalized coaching advice
- Data explanation in simple terms
- Pattern recognition and improvement suggestions

### Function Calling
The AI can automatically call these functions based on your questions:
- `analyze_racing_performance` - Comprehensive performance analysis
- `get_session_statistics` - Statistical analysis of sessions
- `compare_sessions` - Compare multiple racing sessions
- `get_improvement_suggestions` - AI-generated improvement tips
- `search_racing_data` - Search through your data
- `get_telemetry_insights` - Deep telemetry analysis

## 📋 Setup

### 1. OpenAI Configuration (Optional but Recommended)
Set your OpenAI API key for full AI functionality:

```bash
# In your .env file
OPENAI_API_KEY=your-openai-api-key-here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Service
```bash
python main.py
```

## 🎯 Example Usage

### Natural Language Queries
Instead of calling specific API endpoints, you can now ask questions naturally:

```bash
# Ask about performance
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is my fastest lap time?",
    "dataset_id": "your_session_id"
  }'

# Ask for improvement advice
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How can I improve my cornering consistency?",
    "dataset_id": "your_session_id"
  }'

# Ask for comparisons
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare my sector times across different sessions",
    "dataset_id": "your_session_id"
  }'
```

### AI Conversation
Have natural conversations about racing:

```bash
curl -X POST "http://localhost:8000/ai/conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "I am a beginner driver. Can you help me understand what to focus on?",
    "dataset_id": "your_session_id"
  }'
```

### Coaching Advice
Get personalized coaching based on your data:

```bash
curl -X POST "http://localhost:8000/ai/coach-advice" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id",
    "focus_area": "consistency",
    "skill_level": "intermediate"
  }'
```

## 🎮 Example Questions You Can Ask

### Performance Analysis
- "What's my fastest lap time?"
- "How consistent is my driving?"
- "Show me my performance score"
- "Analyze my racing patterns"

### Improvement Focus
- "What areas should I focus on to improve?"
- "Give me suggestions for better cornering"
- "How can I be more consistent?"
- "What's my biggest weakness?"

### Data Understanding
- "Explain this racing data to me"
- "What do these telemetry numbers mean?"
- "Which metrics are most important?"
- "How do I read this analysis?"

### Comparisons
- "Compare my performance with my previous session"
- "Which session was my best?"
- "Show me the difference between my fast and slow laps"

## 🔧 API Endpoints

### Core AI Endpoints
- `POST /query` - Natural language query processing with function calling
- `POST /query/basic` - Basic query processing (fallback)
- `POST /ai/conversation` - Natural conversation with AI
- `POST /ai/explain-data` - Get AI explanation of your data
- `POST /ai/coach-advice` - Get personalized coaching advice

### Traditional Endpoints (Still Available)
- `POST /datasets/upload` - Upload racing session data
- `POST /analyze` - Direct analysis calls
- `POST /racing-session/analyze` - Racing-specific analysis
- `POST /telemetry/upload` - Upload telemetry data

## 🧪 Testing

Run the intelligence test script:

```bash
python test_ai_intelligence.py
```

This will:
1. Upload sample racing data
2. Test natural language understanding
3. Test AI conversation capabilities
4. Test function calling
5. Test coaching advice generation

## 💡 Tips for Best Results

### Asking Questions
- Be specific about what you want to know
- Mention the racing aspect you're interested in
- Ask follow-up questions for deeper insights

### Data Quality
- Upload complete racing sessions for better analysis
- Include telemetry data for comprehensive insights
- Ensure data has proper timestamps and lap information

### AI Configuration
- Set up OpenAI API key for intelligent responses
- Without OpenAI, the service falls back to basic analysis
- GPT-4 is recommended for best function calling results

## 🔄 Fallback Mode

If OpenAI is not configured, the service automatically falls back to:
- Basic pattern matching for question analysis
- Traditional statistical analysis
- Simple response generation
- All analysis functions still work normally

## 🎯 Future Enhancements

- Multi-turn conversations with memory
- Voice-to-text integration
- Automated race strategy suggestions
- Real-time coaching during sessions
- Driver performance trends over time

## 🐛 Troubleshooting

### Common Issues

1. **"OpenAI integration is not configured"**
   - Set the `OPENAI_API_KEY` environment variable
   - Restart the service after setting the key

2. **Function calls not working**
   - Check that you have valid session data uploaded
   - Ensure the dataset_id in your query is correct

3. **Slow responses**
   - OpenAI API calls can take a few seconds
   - Large datasets may require more processing time

4. **"Session not found" errors**
   - Upload data first using `/datasets/upload`
   - Use the correct session ID in your queries

### Debug Mode
Enable detailed logging by setting:
```bash
LOG_LEVEL=DEBUG
```

## 📚 Integration Examples

### Frontend Integration
```javascript
// Natural language query
const response = await fetch('/api/ai/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: userInput,
    dataset_id: currentSessionId
  })
});

const result = await response.json();
console.log(result.answer); // AI-generated response
console.log(result.function_calls); // Any functions that were called
```

### Backend Integration
```python
import httpx

async def ask_ai_about_session(question: str, session_id: str):
    async with httpx.AsyncClient() as client:
        response = await client.post("http://ai-service:8000/query", json={
            "question": question,
            "dataset_id": session_id
        })
        return response.json()
```

This intelligent AI system transforms how drivers interact with their racing data, making analysis accessible through natural conversation while maintaining all the powerful analytical capabilities underneath.
