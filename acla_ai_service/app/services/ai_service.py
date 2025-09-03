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
                "name": "predict_with_telemetry_model",
                "description": "Use trained AI model to predict racing performance based on current telemetry data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "model_id": {"type": "string", "description": "ID of the trained model to use"},
                        "current_telemetry": {"type": "object", "description": "Current telemetry data for prediction"},
                        "prediction_context": {"type": "object", "description": "Additional context like track conditions, session type"}
                    },
                    "required": ["model_id", "current_telemetry"]
                }
            },
            {
                "name": "explain_telemetry_patterns",
                "description": "Generate AI explanations of telemetry data patterns in easy-to-understand language",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "telemetry_data": {"type": "object", "description": "Telemetry analysis results to explain"},
                        "complexity_level": {"type": "string", "enum": ["simple", "detailed", "technical"], "description": "Level of explanation complexity"}
                    },
                    "required": ["telemetry_data"]
                }
            }
        ]
    
    async def process_natural_language_query(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process natural language queries about racing data and function calling"""
        if not self.openai_client:
            raise Exception("OpenAI API is not function properly.")
        
        try:
            # Prepare context information
            context_info = ""
            if context:
                if "track_name" in context:
                    context_info += f"Track: {context['track_name']}\n"
            
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert racing data analyst and AI assistant for ACLA (Assetto Corsa Competizione Lap Analyzer). 
                    You help drivers understand their racing performance through telemetry analysis and personalized AI models.
                    
                    Available context: {context_info}
                    
                    IMPORTANT: You can call specialized functions and AI models to answer user questions:
                    
                    TELEMETRY ANALYSIS FUNCTIONS:
                    - get_telemetry_insights: Get detailed telemetry insights (speed, acceleration, braking, steering)
                    - compare_sessions: Compare multiple racing sessions                   
                    - call_backend_function: Call backend APIs for data retrieval
                    
                    AI MODEL FUNCTIONS (Personalized predictions using user's own driving data):
                    - train_telemetry_ai_model: Train custom AI models on user's telemetry data
                    - predict_with_telemetry_model: Use trained AI models for personalized predictions
                    
                    INTELLIGENT COACHING FUNCTIONS:
                    - generate_ai_recommendations: Generate intelligent coaching recommendations
                    - explain_telemetry_patterns: Explain complex data patterns in simple terms
                    
                    PROCESS:
                    1. Understand what the user wants to know or achieve
                    2. Call appropriate functions to get data from telemetry systems or AI models
                    3. Use your racing expertise to interpret the data
                    4. Provide comprehensive, actionable advice based on actual data
                    
                    EXAMPLES:
                    - "Train an AI to predict my lap times" → Call train_telemetry_ai_model
                    - "What lap time will I get?" → Call predict_with_telemetry_model
                    - "Compare my sessions" → Call compare_sessions
                    
                    Always base your answers on real data from the functions, not assumptions.
                    The AI models learn from each user's unique driving style for personalized predictions.
                    """
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            # STEP 1: Send initial query to OpenAI to determine what functions to call
            tools = [{"type": "function", "function": func} for func in self.get_available_functions()]
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1500
            )
            
            message = response.choices[0].message
            function_results = []
            
            # STEP 2: If OpenAI decides to call functions, execute them to get data from local AI models
            if message.tool_calls:
                print(f"[DEBUG] OpenAI decided to call {len(message.tool_calls)} function(s)")
                
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"[DEBUG] Executing function: {function_name} with args: {function_args}")
                    
                    # Execute the function to get data from local AI models/telemetry
                    result = await self._execute_function(function_name, function_args, context)
                    function_results.append({
                        "function": function_name,
                        "arguments": function_args,
                        "result": result
                    })
                    
                    # Add the function call and result to conversation history
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": message.tool_calls
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })
                
                print(f"[DEBUG] All functions executed, sending results back to OpenAI for final response")
                
                # STEP 3: Send function results back to OpenAI for final comprehensive response
                final_response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100
                )
                
                return {
                    "answer": final_response.choices[0].message.content,
                    "function_calls": function_results,
                    "context": context,
                    "processing_steps": [
                        "1. Analyzed user query with OpenAI",
                        f"2. OpenAI determined {len(function_results)} function(s) needed",
                        "3. Executed functions to get data from local AI models",
                        "4. Sent results back to OpenAI for final response"
                    ]
                }
            
            # If no functions were called, return the direct response
            return {
                "answer": message.content,
                "function_calls": function_results,
                "context": context,
                "processing_steps": [
                    "1. Analyzed query with OpenAI",
                    "2. No additional data needed - direct response provided"
                ]
            }
            
        except Exception as e:
            print(f"[ERROR] AI processing failed: {str(e)}")
            return {
                "error": f"AI processing failed: {str(e)}",
                "fallback": await self._fallback_query_processing(prompt, context)
            }
    
    async def _execute_function(self, function_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """execute the called function to retrieve data from local AI models and telemetry systems"""
        try:
            print(f"[DEBUG] Executing {function_name} to get data from local systems...")

            # Dispatch table for function handlers
            handlers = {
                "get_telemetry_insights": lambda: self.telemetry_service.get_telemetry_insights(
                    arguments.get("session_id"),
                    arguments.get("data_types", ["speed", "acceleration"])
                ),
                "compare_sessions": lambda: self.telemetry_service.compare_sessions(
                    arguments.get("session_ids"),
                    arguments.get("comparison_metrics", ["lap_times"])
                ),
                "train_telemetry_ai_model": lambda: self._train_ai_model_via_backend(arguments, context),
                "predict_with_telemetry_model": lambda: self._predict_via_backend(arguments, context),
                "get_user_models": lambda: self.get_user_models_via_backend(
                    arguments.get("user_id"),
                    arguments.get("track_name"),
                    arguments.get("model_type")
                ),
                "get_active_model": lambda: self.get_active_model_via_backend(
                    arguments.get("user_id"),
                    arguments.get("track_name"),
                    arguments.get("model_type")
                ),
                "perform_incremental_training": lambda: self.perform_incremental_training_via_backend(
                    arguments.get("model_id"),
                    arguments.get("session_ids"),
                    arguments.get("user_id")
                ),
                "call_backend_function": lambda: self.backend_service.call_backend_function(
                    arguments.get("endpoint"),
                    arguments.get("method", "GET"),
                    arguments.get("data"),
                    self._get_auth_headers(arguments.get("user_id"))
                ),
                "train_ai_model": lambda: self._train_ai_model(arguments, context),
                "predict_with_model": lambda: self._predict_with_model(arguments, context),
                "generate_ai_recommendations": lambda: self._generate_ai_recommendations(arguments, context),
                "explain_telemetry_patterns": lambda: self._explain_telemetry_patterns(arguments, context),
            }

            handler = handlers.get(function_name)
            if handler:
                return await handler()
            else:
                print(f"[ERROR] Unknown function: {function_name}")
                return {"error": f"Unknown function: {function_name}"}

        except Exception as e:
            print(f"[ERROR] Function {function_name} execution failed: {str(e)}")
            return {"error": f"Function execution failed: {str(e)}"}
    
    async def _predict_via_backend(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make prediction through backend ai-model controller"""
        try:
            user_id = context.get("user_id") if context else None
            
            # Prepare prediction request for backend
            prediction_request = {
                "modelId": arguments.get("model_id"),
                "inputData": arguments.get("current_telemetry"),
                "predictionOptions": arguments.get("prediction_context", {})
            }
            
            # Call backend ai-model/predict endpoint
            result = await self.backend_service.call_backend_function(
                endpoint="ai-model/predict",
                method="POST",
                data=prediction_request,
                headers=self._get_auth_headers(user_id)
            )
            
            return result
            
        except Exception as e:
            return {"error": f"Backend prediction failed: {str(e)}"}
    
    async def _get_model_insights_via_backend(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get model insights through backend"""
        try:
            user_id = context.get("user_id") if context else None
            model_id = arguments.get("model_id")
            
            # Get model details from backend
            result = await self.backend_service.call_backend_function(
                endpoint=f"ai-model/{model_id}",
                method="GET",
                headers=self._get_auth_headers(user_id)
            )
            
            if not result.get("success"):
                return {"error": "Model not found or access denied"}
            
            model_data = result.get("data", {})
            
            # Extract insights from model metadata
            metadata = model_data.get("modelMetadata", {})
            insights = {
                "model_id": model_id,
                "model_type": metadata.get("modelType", "unknown"),
                "training_sessions_count": metadata.get("trainingSessionsCount", 0),
                "last_training_date": metadata.get("lastTrainingDate"),
                "performance_metrics": metadata.get("performanceMetrics", {}),
                "features": metadata.get("features", []),
                "model_version": model_data.get("modelVersion", "1.0.0"),
                "is_active": model_data.get("isActive", False),
                "track_name": model_data.get("trackName", "unknown")
            }
            
            return {
                "success": True,
                "insights": insights,
                "include_feature_importance": arguments.get("include_feature_importance", True),
                "include_performance_metrics": arguments.get("include_performance_metrics", True)
            }
            
        except Exception as e:
            return {"error": f"Backend model insights failed: {str(e)}"}
    
    async def get_user_models(self, user_id: str, track_name: str = None, model_type: str = None) -> Dict[str, Any]:
        """Get user's models through backend ai-model controller"""
        try:
            # Build endpoint based on parameters
            if track_name:
                endpoint = f"ai-model/user/{track_name}"
                params = {}
                if model_type:
                    params["modelType"] = model_type
                
                # Add query parameters if any
                if params:
                    param_string = "&".join([f"{k}={v}" for k, v in params.items()])
                    endpoint += f"?{param_string}"
            else:
                # Get all user models
                endpoint = f"ai-model/user/all"
            
            result = await self.backend_service.call_backend_function(
                endpoint=endpoint,
                method="GET",
                headers=self._get_auth_headers(user_id)
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_active_model_via_backend(self, user_id: str, track_name: str, model_type: str) -> Dict[str, Any]:
        """Get active model for user/track/type through backend"""
        try:
            result = await self.backend_service.call_backend_function(
                endpoint=f"ai-model/active/{track_name}/{model_type}",
                method="GET",
                headers=self._get_auth_headers(user_id)
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}

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
        
        if any(word in query_lower for word in ["improve", "better", "advice", "coaching"]):
            return {
                "answer": "For improvement suggestions, please configure OpenAI API key for intelligent coaching advice.",
                "suggestion": "Basic advice: Focus on consistency first, then work on finding the racing line and braking points."
            }
        
        else:
            return {
                "answer": "I can help with racing data analysis. Try asking about lap times, performance, or improvements. For intelligent responses, please configure OpenAI API key.",
                "available_functions": ["get_telemetry_insights", "compare_sessions"]
            }
    
    async def _get_telemetry_data_for_sessions(self, session_ids: List[str], user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get telemetry data for specified sessions from backend"""
        try:
            telemetry_data = []
            
            for session_id in session_ids:
                # Call backend to get telemetry data for this session
                result = await self.backend_service.call_backend_function(
                    endpoint=f"/api/sessions/{session_id}/telemetry",
                    method="GET",
                    headers=self._get_auth_headers(user_id)
                )
                
                if result.get("success") and result.get("data"):
                    telemetry_data.extend(result["data"])
            
            return telemetry_data
            
        except Exception as e:
            print(f"[ERROR] Failed to get telemetry data: {str(e)}")
            return []
    
    async def _get_model_data_from_backend(self, model_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get trained model data from backend"""
        try:
            result = await self.backend_service.call_backend_function(
                endpoint=f"/api/ai-models/{model_id}",
                method="GET",
                headers=self._get_auth_headers(user_id)
            )
            
            if result.get("success"):
                return result.get("data")
            else:
                print(f"[ERROR] Failed to get model data: {result.get('error')}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to get model data: {str(e)}")
            return None
    
    async def _save_model_to_backend(self, training_result: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Save trained model data to backend"""
        try:
            model_data = {
                "user_id": user_id,
                "model_type": training_result.get("model_type"),
                "target_variable": training_result.get("target_variable"),
                "model_data": training_result.get("model_data"),
                "feature_names": training_result.get("feature_names"),
                "training_metrics": training_result.get("training_metrics"),
                "feature_count": training_result.get("feature_count"),
                "training_samples": training_result.get("training_samples"),
                "model_version": training_result.get("model_version"),
                "session_metadata": training_result.get("session_metadata"),
                "created_at": training_result.get("created_at")
            }
            
            result = await self.backend_service.call_backend_function(
                endpoint="/api/ai-models",
                method="POST",
                data=model_data,
                headers=self._get_auth_headers(user_id)
            )
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Failed to save model to backend: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def delete_user_model(self, model_id: str, user_id: str) -> Dict[str, Any]:
        """Delete a user's AI model through backend controller"""
        try:
            result = await self.backend_service.call_backend_function(
                endpoint=f"ai-model/{model_id}",
                method="DELETE",
                headers=self._get_auth_headers(user_id)
            )
            
            return result
            
        except Exception as e:
            return {"success": False, "error": str(e)}