# ACLA AI Service - OpenAI Integration Setup Guide

## 🚀 Overview

This guide will help you set up the enhanced ACLA AI Service with OpenAI integration for intelligent racing data analysis, natural language queries, and automatic backend function calling.

## 🔧 Prerequisites

1. **OpenAI API Key** (Required for intelligent features)
   - Sign up at [OpenAI](https://platform.openai.com/)
   - Create an API key
   - Ensure you have access to GPT-4 for best results

2. **Python Environment**
   - Python 3.9+
   - Virtual environment (recommended)

3. **Backend Service**
   - ACLA Backend running on port 7001
   - AI Model endpoints available

## 📦 Installation

### 1. Install Dependencies

```bash
cd acla_ai_service
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit the `.env` file:

```bash
# REQUIRED: Add your OpenAI API key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Backend URL (adjust if needed)
BACKEND_URL=http://localhost:7001

# Service configuration
AI_SERVICE_HOST=0.0.0.0
AI_SERVICE_PORT=8000
DEBUG=true
```

### 3. Start the Service

```bash
python main.py
```

Or using uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 🧪 Testing the Integration

Run the comprehensive test suite:

```bash
python test_ai_integration.py
```

This will test:
- ✅ Service health
- 🧠 OpenAI integration
- 📤 Data upload
- 🗣️ Natural language queries
- 💬 AI conversation
- 🏁 Coaching advice
- 🔧 Backend function calling

## 🎯 Key Features

### 1. Natural Language Queries

Users can ask questions in plain English:

```javascript
// Frontend example
const response = await fetch('/api/ai/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "How can I improve my lap times?",
    dataset_id: sessionId,
    user_id: userId
  })
});
```

### 2. Automatic Function Calling

The AI automatically calls appropriate functions:

- **analyze_racing_performance** - For performance questions
- **get_telemetry_insights** - For telemetry analysis
- **compare_sessions** - For comparisons
- **train_ai_model** - For model training requests
- **call_backend_function** - For backend API calls

### 3. Backend Integration

AI can call backend endpoints:

```javascript
// Backend controller example
@Post('ai-query')
async processAIQuery(@Body() body: { query: string, sessionId?: string }, @Request() req: any) {
    return this.aiModelService.processAIQuery({
        question: body.query,
        dataset_id: body.sessionId,
        user_id: req.user.id
    });
}
```

## 📋 Available Endpoints

### Core AI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Main natural language query endpoint |
| `/api/v1/ai/conversation` | POST | AI conversation |
| `/api/v1/ai/coach-advice` | POST | Personalized coaching |
| `/api/v1/ai/explain-data` | POST | Data explanation |
| `/api/v1/ai/capabilities` | GET | AI capabilities info |

### Backend Integration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ai-model/ai-query` | POST | AI queries for model operations |
| `/ai-model/intelligent-training` | POST | Natural language training requests |
| `/ai-model/ask-about-models` | POST | Questions about user models |

## 🔍 Usage Examples

### 1. Performance Analysis

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze my performance and tell me where I can improve",
    "dataset_id": "session_123",
    "user_id": "user_456"
  }'
```

### 2. Model Training

```bash
curl -X POST "http://localhost:7001/api/ai-model/intelligent-training" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-jwt-token" \
  -d '{
    "naturalLanguageRequest": "Train a lap time prediction model for Spa using my last 5 sessions"
  }'
```

### 3. Coaching Advice

```bash
curl -X POST "http://localhost:8000/api/v1/ai/coach-advice" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Give me coaching advice",
    "dataset_id": "session_123",
    "context": {
      "skill_level": "intermediate",
      "focus_area": "consistency"
    }
  }'
```

## 🛠️ Troubleshooting

### Common Issues

1. **"OpenAI integration is not configured"**
   - Check your `OPENAI_API_KEY` in `.env`
   - Restart the service after setting the key
   - Verify API key is valid

2. **"Function calls not working"**
   - Ensure you're using GPT-4 (set in ai_service.py)
   - Check that session data exists
   - Verify backend connectivity

3. **Backend function calls failing**
   - Check `BACKEND_URL` in configuration
   - Ensure backend service is running
   - Verify authentication headers

4. **Slow responses**
   - OpenAI API calls can take 2-10 seconds
   - Large datasets require more processing time
   - Consider implementing caching

### Debug Mode

Enable detailed logging:

```bash
LOG_LEVEL=DEBUG
```

### Fallback Mode

If OpenAI is not configured, the service automatically falls back to:
- Basic keyword matching
- Traditional statistical analysis
- Simple response generation

## 🚀 Frontend Integration

### React Example

```javascript
import { useState } from 'react';

function AIQueryComponent({ sessionId, userId, authToken }) {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const askAI = async () => {
    setLoading(true);
    try {
      const result = await fetch('/api/ai/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
          question: query,
          dataset_id: sessionId,
          user_id: userId
        })
      });
      
      const data = await result.json();
      setResponse(data);
    } catch (error) {
      console.error('AI query failed:', error);
    }
    setLoading(false);
  };

  return (
    <div>
      <textarea 
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask about your racing performance..."
      />
      <button onClick={askAI} disabled={loading}>
        {loading ? 'Thinking...' : 'Ask AI'}
      </button>
      
      {response && (
        <div>
          <h3>AI Response:</h3>
          <p>{response.answer}</p>
          
          {response.function_calls?.length > 0 && (
            <div>
              <h4>Analysis Performed:</h4>
              <ul>
                {response.function_calls.map((fc, i) => (
                  <li key={i}>{fc.function}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

## 🎯 Next Steps

1. **Configure OpenAI API Key** - Essential for intelligent features
2. **Test with Real Data** - Upload actual racing sessions
3. **Train AI Models** - Use natural language to train models
4. **Integrate with Frontend** - Add AI query components
5. **Monitor Performance** - Track API usage and response times

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [ACLA Backend API Reference](../acla_backend/README.md)

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review service logs for errors
3. Test with the provided test suite
4. Verify all dependencies are installed

---

**Ready to revolutionize your racing data analysis with AI! 🏁🤖**
