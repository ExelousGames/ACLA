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
                "name": "follow_expert_line",
                "description": "Guide the user to follow the optimal racing line based on telemetry data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID for telemetry analysis"},
                        "data_types": {"type": "array", "items": {"type": "string"}, "description": "Types of telemetry data to analyze (speed, acceleration, braking, steering)"}
                    },
                    "required": ["session_id"]
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
                    "content": f"""You are an expert racing data analyst and AI assistant for sim racing. 
                    You help drivers understand their racing performance through telemetry analysis and personalized AI models.
                    Also, user can ask you to perform some actions using the tools provided.

                    user is on Track: {context['track_name']} driving {context['car_name']}
                    
                    IMPORTANT: You can call specialized functions which can extract useful data to answer user questions or perform some actions:
                    
                    AI MODEL FUNCTIONS (Personalized predictions using user's own driving data):
                    - train_telemetry_ai_model: Train custom AI models on user's telemetry data
                    - predict_with_telemetry_model: Use trained AI models for personalized predictions
                    
                    PROCESS:
                    1. Understand what the user wants to know or achieve
                    2. Call appropriate functions to get data from telemetry systems and AI models, or perform actions
                    3. Use your racing expertise to interpret the data
                    4. Provide comprehensive, actionable advice based on actual data, or request the system to do something
                    
                    EXAMPLES:
                    - "i want you guide me on track" → Call follow_expert_line
                    
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
                    
                    # Execute the function to get data or preform actions
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
            }
    
    async def _execute_function(self, function_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """execute the called function to retrieve data from local AI models and telemetry systems"""
        try:
            print(f"[DEBUG] Executing {function_name} to get data from local systems...")

            # Dispatch table for function handlers
            handlers = {
                "get_telemetry_insights": lambda: self.backend_service.get_telemetry_insights(
                    arguments.get("session_id"),
                    arguments.get("data_types", ["speed", "acceleration"])
                ),
                "compare_sessions": lambda: self.backend_service.compare_sessions(
                    arguments.get("session_ids"),
                    arguments.get("comparison_metrics", ["lap_times"])
                )
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


