"""
AI Service for natural language processing and conversation
"""

from typing import Dict, Any, Optional, List
import json
import asyncio
from openai import AsyncOpenAI
from app.core import settings
from app.services.telemetry_service import TelemetryService
from app.services.backend_service import BackendService


class AIService:
    """Service for AI-powered analysis and conversation"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.telemetry_service = TelemetryService()
        self.backend_service = BackendService()
    
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """Define available functions for OpenAI function calling,
        if OpenAI decides to call a function, it executes it"""
        return [
            {
                "name": "analyze_racing_performance",
                "description": "Analyze racing performance from telemetry data including lap times, sector times, and speed analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Racing session ID to analyze"},
                        "analysis_type": {"type": "string", "enum": ["overall", "sectors", "consistency", "comparison"], "description": "Type of performance analysis"},
                        "focus_areas": {"type": "array", "items": {"type": "string"}, "description": "Specific areas to focus on (e.g., braking, cornering, acceleration)"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "get_telemetry_insights",
                "description": "Get detailed telemetry insights including speed traces, g-forces, and car dynamics",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID for telemetry analysis"},
                        "data_types": {"type": "array", "items": {"type": "string"}, "description": "Types of telemetry data to analyze (speed, acceleration, braking, steering)"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "compare_sessions",
                "description": "Compare multiple racing sessions to identify improvements and patterns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_ids": {"type": "array", "items": {"type": "string"}, "description": "List of session IDs to compare"},
                        "comparison_metrics": {"type": "array", "items": {"type": "string"}, "description": "Metrics to compare (lap_times, sectors, consistency, etc.)"}
                    },
                    "required": ["session_ids"]
                }
            },
            {
                "name": "get_improvement_suggestions",
                "description": "Generate personalized improvement suggestions based on racing data analysis",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID to analyze for improvements"},
                        "skill_level": {"type": "string", "enum": ["beginner", "intermediate", "advanced"], "description": "Driver skill level"},
                        "focus_area": {"type": "string", "description": "Specific area to focus improvements on"}
                    },
                    "required": ["session_id"]
                }
            },
            {
                "name": "call_backend_function",
                "description": "Call backend API functions to retrieve or modify data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint": {"type": "string", "description": "Backend API endpoint to call"},
                        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"], "description": "HTTP method"},
                        "data": {"type": "object", "description": "Data to send with the request"},
                        "user_id": {"type": "string", "description": "User ID for authentication"}
                    },
                    "required": ["endpoint", "method"]
                }
            },
            {
                "name": "train_ai_model",
                "description": "Train a new AI model or perform incremental training on existing models",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_type": {"type": "string", "enum": ["lap_time_prediction", "sector_analysis", "setup_optimization"], "description": "Type of AI model to train"},
                        "session_ids": {"type": "array", "items": {"type": "string"}, "description": "Training data session IDs"},
                        "model_name": {"type": "string", "description": "Name for the new model"},
                        "track_name": {"type": "string", "description": "Track name for the model"},
                        "user_id": {"type": "string", "description": "User ID who owns the model"}
                    },
                    "required": ["model_type", "session_ids", "user_id"]
                }
            },
            {
                "name": "predict_with_model",
                "description": "Make predictions using trained AI models",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string", "description": "ID of the trained model"},
                        "input_data": {"type": "object", "description": "Input telemetry data for prediction"},
                        "prediction_type": {"type": "string", "description": "Type of prediction to make"}
                    },
                    "required": ["model_id", "input_data"]
                }
            }
        ]
    
    async def process_natural_language_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process natural language queries about racing data with function calling"""
        if not self.openai_client:
            return await self._fallback_query_processing(query, context)
        
        try:
            # Prepare context information
            context_info = ""
            if context:
                if "session_id" in context:
                    context_info += f"Current session: {context['session_id']}\n"
                if "user_id" in context:
                    context_info += f"User ID: {context['user_id']}\n"
                if "track_name" in context:
                    context_info += f"Track: {context['track_name']}\n"
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert racing data analyst and AI assistant for ACLA (Assetto Corsa Competizione Lap Analyzer). 
                    You help drivers understand their racing performance through telemetry analysis.
                    
                    Available context: {context_info}
                    
                    You can call functions to:
                    - Analyze racing performance and telemetry data
                    - Compare sessions and identify improvements  
                    - Train AI models for predictions
                    - Call backend APIs to retrieve or modify data
                    - Generate personalized coaching advice
                    
                    Always provide helpful, actionable insights in a conversational manner.
                    When users ask about their performance, call the appropriate analysis functions.
                    When they want to train models or make predictions, use the AI model functions.
                    When they need data from the backend, use the backend calling function.
                    """
                },
                {
                    "role": "user", 
                    "content": query
                }
            ]
            
            # Call OpenAI with function calling
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                functions=self.get_available_functions(),
                function_call="auto",
                temperature=0.7,
                max_tokens=1500
            )
            
            message = response.choices[0].message
            
            # Handle function calls
            function_results = []
            
            #If OpenAI decides to call a function, it executes it:
            if message.function_call:
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)
                
                # Execute the function - ends function results back to OpenAI for final response.
                result = await self._execute_function(function_name, function_args, context)
                function_results.append({
                    "function": function_name,
                    "arguments": function_args,
                    "result": result
                })
                
                # Get follow-up response from AI
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "function_call": message.function_call
                })
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(result)
                })
                
                # Get final response
                final_response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                )
                
                return {
                    "answer": final_response.choices[0].message.content,
                    "function_calls": function_results,
                    "context": context
                }
            
            return {
                "answer": message.content,
                "function_calls": function_results,
                "context": context
            }
            
        except Exception as e:
            return {
                "error": f"AI processing failed: {str(e)}",
                "fallback": await self._fallback_query_processing(query, context)
            }
    
    async def _execute_function(self, function_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the called function"""
        try:
            if function_name == "analyze_racing_performance":
                return await self.telemetry_service.analyze_racing_performance(
                    arguments.get("session_id"),
                    arguments.get("analysis_type", "overall"),
                    arguments.get("focus_areas", [])
                )
            
            elif function_name == "get_telemetry_insights":
                return await self.telemetry_service.get_telemetry_insights(
                    arguments.get("session_id"),
                    arguments.get("data_types", ["speed", "acceleration"])
                )
            
            elif function_name == "compare_sessions":
                return await self.telemetry_service.compare_sessions(
                    arguments.get("session_ids"),
                    arguments.get("comparison_metrics", ["lap_times"])
                )
            
            elif function_name == "get_improvement_suggestions":
                return await self.telemetry_service.get_improvement_suggestions(
                    arguments.get("session_id"),
                    arguments.get("skill_level", "intermediate"),
                    arguments.get("focus_area")
                )
            
            elif function_name == "call_backend_function":
                return await self.backend_service.call_backend_function(
                    arguments.get("endpoint"),
                    arguments.get("method", "GET"),
                    arguments.get("data"),
                    self._get_auth_headers(arguments.get("user_id"))
                )
            
            elif function_name == "train_ai_model":
                return await self._train_ai_model(arguments, context)
            
            elif function_name == "predict_with_model":
                return await self._predict_with_model(arguments, context)
            
            else:
                return {"error": f"Unknown function: {function_name}"}
                
        except Exception as e:
            return {"error": f"Function execution failed: {str(e)}"}
    
    async def _train_ai_model(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train an AI model via backend"""
        endpoint = "ai-model/train-new"
        data = {
            "trackName": arguments.get("track_name", context.get("track_name") if context else "unknown"),
            "modelType": arguments.get("model_type"),
            "modelName": arguments.get("model_name", f"AI Model {arguments.get('model_type')}"),
            "sessionIds": arguments.get("session_ids"),
            "trainingParameters": arguments.get("training_parameters", {})
        }
        
        return await self.backend_service.call_backend_function(
            endpoint, "POST", data, 
            self._get_auth_headers(arguments.get("user_id"))
        )
    
    async def _predict_with_model(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make prediction with AI model via backend"""
        endpoint = "ai-model/predict"
        data = {
            "modelId": arguments.get("model_id"),
            "inputData": arguments.get("input_data"),
            "predictionOptions": arguments.get("prediction_options", {})
        }
        
        return await self.backend_service.call_backend_function(
            endpoint, "POST", data,
            self._get_auth_headers(context.get("user_id") if context else None)
        )
    
    def _get_auth_headers(self, user_id: Optional[str] = None) -> Dict[str, str]:
        """Get authentication headers for backend calls"""
        headers = {"Content-Type": "application/json"}
        if user_id:
            # Add JWT token or user authentication here
            headers["X-User-ID"] = user_id
        return headers
    
    async def _fallback_query_processing(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback processing when OpenAI is not available"""
        # Basic keyword matching and analysis
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["lap time", "fastest", "time", "performance"]):
            if context and "session_id" in context:
                result = await self.telemetry_service.analyze_racing_performance(
                    context["session_id"], "overall", []
                )
                return {
                    "answer": "Here's your performance analysis (basic mode - OpenAI not configured):",
                    "data": result
                }
        
        elif any(word in query_lower for word in ["improve", "better", "advice", "coaching"]):
            return {
                "answer": "For improvement suggestions, please configure OpenAI API key for intelligent coaching advice.",
                "suggestion": "Basic advice: Focus on consistency first, then work on finding the racing line and braking points."
            }
        
        else:
            return {
                "answer": "I can help with racing data analysis. Try asking about lap times, performance, or improvements. For intelligent responses, please configure OpenAI API key.",
                "available_functions": ["analyze_racing_performance", "get_telemetry_insights", "compare_sessions"]
            }
    
    async def generate_coaching_advice(self, performance_data: Dict[str, Any], skill_level: str = "intermediate") -> Dict[str, Any]:
        """Generate AI coaching advice based on performance data"""
        if not self.openai_client:
            return {"error": "OpenAI API key not configured"}
        
        try:
            prompt = f"""
            As an expert racing coach, analyze this performance data and provide specific, actionable coaching advice for a {skill_level} level driver:
            
            Performance Data:
            {json.dumps(performance_data, indent=2)}
            
            Please provide:
            1. Key strengths and weaknesses identified
            2. Specific areas for improvement
            3. Actionable techniques and strategies
            4. Practice recommendations
            5. Next steps for development
            
            Make the advice encouraging and practical.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                "coaching_advice": response.choices[0].message.content,
                "skill_level": skill_level,
                "analysis_data": performance_data
            }
            
        except Exception as e:
            return {"error": f"Coaching advice generation failed: {str(e)}"}
    
    async def explain_data_patterns(self, data_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanations for data patterns"""
        if not self.openai_client:
            return {"error": "OpenAI API key not configured"}
        
        try:
            prompt = f"""
            Explain these racing data patterns in simple, easy-to-understand terms:
            
            Data Analysis:
            {json.dumps(data_analysis, indent=2)}
            
            Please provide:
            1. What the data shows in plain language
            2. Why these patterns occur
            3. What they mean for racing performance
            4. How to interpret the numbers
            
            Make it accessible for drivers who may not be data experts.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )
            
            return {
                "explanation": response.choices[0].message.content,
                "data_summary": data_analysis
            }
            
        except Exception as e:
            return {"error": f"Data explanation failed: {str(e)}"}
