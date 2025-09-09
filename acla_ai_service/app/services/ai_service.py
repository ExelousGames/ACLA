"""
AI Service for natural language processing and conversation
"""

from typing import Dict, Any, Optional, List
import json
import asyncio
from openai import AsyncOpenAI
from app.services.full_dataset_ml_service import Full_dataset_TelemetryMLService
from app.core import settings
from app.services.telemetry_service import TelemetryService
from app.services.backend_service import BackendService


class AIService:
    """Service for AI-powered analysis and conversation"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.telemetry_service = TelemetryService()
        self.backend_service = BackendService()
        self.telemetryMLService = Full_dataset_TelemetryMLService()
    def get_available_functions(self) -> List[Dict[str, Any]]:
        """Define available functions for OpenAI function calling,
        if OpenAI decides to call a function, it executes it"""
        return [
            {
                "name": "check_car_limit",
                "description": "Check if the car is within the optimal limits based on telemetry data",
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
                "name": "enable_guide_user_racing",
                "description": "this will enable continuous guidance that monitors telemetry data and provides real-time recommendations. This function does not require any parameters.",
                "parameters": {
                }
            }
        ]
    
    async def process_natural_language_query(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Process natural language queries about racing data and function calling
        
        Args:
            prompt: The user's natural language query
            context: Optional context information (track, car, etc.)
            conversation_history: Optional previous conversation messages to maintain context
        """
        if not self.openai_client:
            raise Exception("OpenAI API is not function properly.")
        
        try:
            # Prepare context information and conversation history
            try:
                # Start with system message
                system_message = {
                    "role": "system",
                    "content": f"""You are an expert racing data analyst and AI assistant for sim racing. always stay in character.
                    You help drivers understand their racing performance through telemetry analysis and personalized AI models.
                    Also, user can ask you to perform some actions using the tools provided.

                    user is on Track: {context.get('track_name', 'Unknown') if context else 'Unknown'} driving {context.get('car_name', 'Unknown') if context else 'Unknown'}

                    IMPORTANT: You can call specialized functions which can extract useful data to answer user questions or perform some actions or enable some features requested by the user:
                    
                    AI MODEL FUNCTIONS (Personalized predictions using user's own driving data):
                    - train_telemetry_ai_model: Train custom AI models on user's telemetry data
                    - enable_guide_user_racing: enable continuously monitor telemetry and provide real-time driving guide on track
                    
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
                }
                
                # Build messages array - use provided conversation history or start fresh
                if conversation_history and len(conversation_history) > 0:
                    # Use provided conversation history and add the new user message
                    messages = conversation_history.copy()
                    messages.append({"role": "user", "content": prompt})
                else:
                    # Start fresh conversation
                    messages = [
                        system_message,
                        {"role": "user", "content": prompt}
                    ]
                    
            except Exception as e:
                print(f"[ERROR] Failed to prepare context messages: {str(e)}")
                raise Exception(f"Context preparation failed: {str(e)}") from e
            
            # STEP 1: Send initial query to OpenAI to determine what functions to call
            try:
                tools = [{"type": "function", "function": func} for func in self.get_available_functions()]
            except Exception as e:
                print(f"[ERROR] Failed to prepare tools: {str(e)}")
                raise Exception(f"Tools preparation failed: {str(e)}") from e

            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                temperature=0.7,
                max_tokens=500
            )
            except Exception as e:
                print(f"[ERROR] OpenAI API call failed: {str(e)}")
                raise Exception(f"OpenAI API call failed: {str(e)}") from e
            
            try:
                message = response.choices[0].message
                function_results = []
            except Exception as e:
                print(f"[ERROR] Failed to parse OpenAI response: {str(e)}")
                raise Exception(f"OpenAI response parsing failed: {str(e)}") from e
            
            # STEP 2: If OpenAI decides to call functions, execute them to get data from local AI models or perform actions
            if message.tool_calls:
                print(f"[DEBUG] OpenAI decided to call {len(message.tool_calls)} function(s)")

                # Add the assistant message with tool calls to conversation history (once, not per tool call)
                try:
                    messages.append({
                        "role": "assistant",
                        "content": message.content,
                        "tool_calls": message.tool_calls
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to append assistant message to history: {str(e)}")
                    # Continue processing even if message history fails
                    pass
                
                try:
                    for tool_call in message.tool_calls:
                        try:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                        except Exception as e:
                            print(f"[ERROR] Failed to parse function call: {str(e)}")
                            function_results.append({
                                "function": "parse_error",
                                "arguments": {},
                                "result": {"error": f"Function call parsing failed: {str(e)}"}
                            })
                            continue
                        
                        print(f"[DEBUG] Executing function: {function_name} with args: {function_args}")
                        
                        # Execute the function to get data or preform actions
                        try:
                            result = await self._execute_function(function_name, function_args, context)
                        except Exception as e:
                            print(f"[ERROR] Function execution failed for {function_name}: {str(e)}")
                            result = {"error": f"Function execution failed: {str(e)}"}
                        
                        function_results.append({
                            "function": function_name,
                            "arguments": function_args,
                            "result": result
                        })
                        
                        # Add the tool result to conversation history
                        try:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result)
                            })
                        except Exception as e:
                            print(f"[ERROR] Failed to append tool result to message history: {str(e)}")
                            # Continue processing even if message history fails
                            pass
                except Exception as e:
                    print(f"[ERROR] Function execution loop failed: {str(e)}")
                    raise Exception(f"Function execution loop failed: {str(e)}") from e
                
                print(f"[DEBUG] All functions executed, sending results back to OpenAI for final response")
                
                # STEP 3: Send function results back to OpenAI for final comprehensive response
                try:
                    final_response = await self.openai_client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=500
                    )
                except Exception as e:
                    print(f"[ERROR] Final OpenAI API call failed: {str(e)}")
                    raise Exception(f"Final OpenAI API call failed: {str(e)}") from e
                
                try:
                    final_answer = final_response.choices[0].message.content
                    
                    # Add the final assistant response to the conversation
                    messages.append({
                        "role": "assistant",
                        "content": final_answer
                    })
                    
                    return {
                        "answer": final_answer,
                        "function_calls": function_results,
                        "context": context,
                        "messages": messages,  # Return updated conversation for external management
                        "processing_steps": [
                            "1. Analyzed user query with OpenAI",
                            f"2. OpenAI determined {len(function_results)} function(s) needed",
                            "3. Executed functions to get data from local AI models",
                            "4. Sent results back to OpenAI for final response"
                        ]
                    }
                except Exception as e:
                    print(f"[ERROR] Failed to format final response: {str(e)}")
                    raise Exception(f"Final response formatting failed: {str(e)}") from e
            
            # If no functions were called, return the direct response
            try:
                direct_answer = message.content
                
                # Add the assistant response to the conversation
                messages.append({
                    "role": "assistant",
                    "content": direct_answer
                })
                
                return {
                    "answer": direct_answer,
                    "function_calls": function_results,
                    "context": context,
                    "messages": messages,  # Return updated conversation for external management
                    "processing_steps": [
                        "1. Analyzed query with OpenAI",
                        "2. No additional data needed - direct response provided"
                    ]
                }
            except Exception as e:
                print(f"[ERROR] Failed to format direct response: {str(e)}")
                raise Exception(f"Direct response formatting failed: {str(e)}") from e
            
        except Exception as e:
            print(f"[ERROR] Natural language query processing failed: {str(e)}")
            print(f"[ERROR] Call stack: {type(e).__name__}: {str(e)}")
            
            # Return error response with call stack information
            return {
                "error": f"Natural language query processing failed: {str(e)}",
                "error_type": type(e).__name__,
                "processing_steps": [
                    "1. Error occurred during natural language query processing",
                    f"2. Error type: {type(e).__name__}",
                    f"3. Error message: {str(e)}"
                ],
                "context": context
            }
    
    async def _execute_function(self, function_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """execute the called function to retrieve data from local AI models and telemetry systems"""
        try:
            print(f"[DEBUG] Executing {function_name}...")

            # Handle different function types (sync vs async)
            if function_name == "enable_guide_user_racing":
                return self.enable_guide_user_racing()
            elif function_name == "compare_sessions":
                return await self.backend_service.compare_sessions(
                    arguments.get("session_ids"),
                    arguments.get("comparison_metrics", ["lap_times"])
                )
            else:
                print(f"[ERROR] Unknown function: {function_name}")
                return {"error": f"Unknown function: {function_name}"}

        except Exception as e:
            print(f"[ERROR] Function {function_name} execution failed: {str(e)}")
            return {"error": f"Function execution failed: {str(e)}"}

    def enable_guide_user_racing(self, trackName: str, carName: str) -> Dict[str, Any]:
        
        track_corner_data = self.backend_service.getCompleteActiveModelData(trackName,'track_corner_analysis')

        
            
        results = { "_skip_openai_processing": False,
                   'function_name': 'enable_guide_user_racing'
                     ,'message': 'Racing guidance has been enabled! I will now monitor your telemetry data and provide real-time recommendations to help you improve your lap times and racing performance.'
                   }
        return results
