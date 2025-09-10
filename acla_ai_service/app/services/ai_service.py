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
                "name": "track_detail_for_guide",
                "description": "Start the guiding process. After calling this function, you still need to generate some sentences for gas, brake, and throttle inputs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "track_name": {"type": "string", "description": "Name of the track to retrieve data"}
                    },
                    "required": ["track_name"]
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
        
        PROCESS FLOW DIAGRAM:
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │                           TWO-STAGE AI PROCESSING                           │
        └─────────────────────────────────────────────────────────────────────────────┘
        
        ┌─────────────────┐    ┌─────────────────────────────────────────────────────┐
        │   User Prompt   │───▶│                STAGE 1: ANALYSIS                   │
        └─────────────────┘    │                                                     │
                               │  ┌─────────────────┐                               │
                               │  │   OpenAI GPT-4  │  Examines prompt &            │
                               │  │   First Query   │  decides what functions       │
                               │  │                 │  to call (if any)             │
                               │  └─────────┬───────┘                               │
                               │            │                                       │
                               │            ▼                                       │
                               │  ┌─────────────────┐                               │
                               │  │ Function Calls? │                               │
                               │  │   (tool_calls)  │                               │
                               │  └─────┬─────┬─────┘                               │
                               └────────┼─────┼─────────────────────────────────────┘
                                       NO    YES
                                        │     │
                               ┌────────▼─────▼─────────────────────────────────────┐
                               │                STAGE 2: EXECUTION                  │
                    ┌──────────┤                                                     │
                    │          │  ┌─────────────────┐    ┌─────────────────────────┐│
             Direct │          │  │  Execute Each   │    │    Function Results     ││
             Answer │          │  │    Function     │───▶│                         ││
                    │          │  │                 │    │  • Data for OpenAI      ││
                    │          │  └─────────┬───────┘    │  • Side Products (_*)   ││
                    │          │            │            │    - _guidance_enabled  ││
                    │          │            ▼            │    - _prediction_result ││
                    │          │  ┌─────────────────┐    │    - _track_corner_data ││
                    │          │  │   OpenAI GPT-4  │    │                         ││
                    │          │  │  Second Query   │◀───┤  Filter: Only send     ││
                    │          │  │  (Final Answer) │    │  non-underscore data    ││
                    │          │  └─────────┬───────┘    │  to OpenAI              ││
                    │          └────────────┼────────────┴─────────────────────────┘│
                    │                       │                                       │
                    │                       ▼                                       │
                    │          ┌─────────────────────────────────────────────────────┐
                    └─────────▶│                 FINAL RESULT                        │
                               │                                                     │
                               │  {                                                  │
                               │    "answer": "OpenAI's final response",            │
                               │    "side_products": {                              │
                               │      "track_detail_for_guide": {                 │
                               │        "_guidance_enabled": true,                  │
                               │        "_prediction_result": {...},               │
                               │        "_track_corner_data": {...}                │
                               │      }                                             │
                               │    },                                              │
                               │    "context": {...},                              │
                               │    "messages": [...]                              │
                               │  }                                                 │
                               └─────────────────────────────────────────────────────┘
        
        KEY CONCEPTS:
        • Stage 1: OpenAI analyzes user intent and decides what functions to call
        • Stage 2: Functions execute and produce TWO types of outputs:
          - Regular data: Sent to OpenAI for generating the final answer
          - Side products (prefixed with _): Returned to caller for external use
        • Final result contains both OpenAI's answer AND all side products
        
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
                    - track_detail_for_guide: start the guiding process. after calling this function, you still need to generate some sentences for gas, brake, and throttle inputs. comboine the cornering analysis data with the simple guidance data to generate detailed driving guide and real-time driving guide on track display at specific points on the track. system will notice how to process the response

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
                    model="gpt-4o",
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
                
                # Store side products from function executions
                side_products = {}
                
                try:
                    for tool_call in message.tool_calls:
                        try:
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                        except Exception as e:
                            # Store parse error in side_products for debugging
                            side_products["parse_error"] = {
                                "_error": f"Function call parsing failed: {str(e)}",
                                "_function": "parse_error",
                                "_arguments": {}
                            }
                            continue
                        
                        print(f"[DEBUG] Executing function: {function_name} with args: {function_args}")
                        
                        # Execute the function to get data or perform actions
                        try:
                            result = await self._execute_function(function_name, function_args, context)
                            
                            # Check if function has side products (anything that should be returned to caller)
                            if isinstance(result, dict):
                                # Extract data for OpenAI (excluding side products)
                                openai_data = {k: v for k, v in result.items() if not k.startswith('_')}
                                
                                # Extract side products (keys starting with _)
                                function_side_products = {k: v for k, v in result.items() if k.startswith('_')}
                                if function_side_products:
                                    side_products[function_name] = function_side_products
                                
                                # Use filtered data for OpenAI
                                result_for_openai = openai_data if openai_data else result
                            else:
                                result_for_openai = result
                                
                        except Exception as e:
                            print(f"[ERROR] Function execution failed for {function_name}: {str(e)}")
                            result = {"error": f"Function execution failed: {str(e)}"}
                            result_for_openai = result
                        
                        # Add the tool result to conversation history (use filtered data for OpenAI)
                        try:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result_for_openai)
                            })
                        except Exception as e:
                            # Continue processing even if message history fails
                            pass
                except Exception as e:
                    raise Exception(f"Function execution loop failed: {str(e)}") from e
                
                print(f"[DEBUG] All functions executed, sending results back to OpenAI for final response")
                # STEP 3: Send function results back to OpenAI for final comprehensive response
                try:
                    final_response = await self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1500
                    )
                except Exception as e:
                    raise Exception(f"Final OpenAI API call failed: {str(e)}") from e
                
                try:
                    final_answer = final_response.choices[0].message.content
                    
                    # Add the final assistant response to the conversation
                    messages.append({
                        "role": "assistant",
                        "content": final_answer
                    })
                    
                    # Prepare final result with OpenAI answer and all side products
                    result = {
                        "answer": final_answer,
                        "context": context,
                        "messages": messages,  # Return updated conversation for external management
                    }
                    
                    # Add side products from function executions to the final result
                    if side_products:
                        result["side_products"] = side_products
                        
                    return result
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
                
                result = {
                    "answer": direct_answer,
                    "context": context,
                    "messages": messages,  # Return updated conversation for external management
                }
                
                # No side products since no functions were called, but maintain consistent structure
                # (side_products will be empty/absent)
                
                return result
            except Exception as e:
                print(f"[ERROR] Failed to format direct response: {str(e)}")
                raise Exception(f"Direct response formatting failed: {str(e)}") from e
            
        except Exception as e:

            # Return error response with call stack information
            raise RuntimeError({
                "error": f"Natural language query processing failed: {str(e)}",
                "error_type": type(e).__name__,
                "processing_steps": [
                    "1. Error occurred during natural language query processing",
                    f"2. Error type: {type(e).__name__}",
                    f"3. Error message: {str(e)}"
                ],
                "context": context
            })
    
    async def _execute_function(self, function_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the called function to retrieve data from local AI models and telemetry systems
        
        FUNCTION OUTPUT SEPARATION:
        ┌─────────────────────────────────────────────────────────────────┐
        │                    Function Return Format                       │
        │                                                                 │
        │  {                                                              │
        │    # Regular keys → Sent to OpenAI for final answer            │
        │    "status": "success",                                         │
        │    "message": "Operation completed",                            │
        │                                                                 │
        │    # Keys starting with _ → Side products for external use     │
        │    "_guidance_enabled": true,                                   │
        │    "_prediction_result": {...},                                 │
        │    "_track_corner_data": {...},                                │
        │    "_skip_openai_processing": true                             │
        │  }                                                             │
        └─────────────────────────────────────────────────────────────────┘
        """
        try:
            # Handle different function types (sync vs async)
            if function_name == "track_detail_for_guide":
                return await self.track_detail_for_guide()
            elif function_name == "compare_sessions":
                return await self.backend_service.compare_sessions(
                    arguments.get("session_ids"),
                    arguments.get("comparison_metrics", ["lap_times"])
                )
            else:
                print(f"[ERROR] Unknown function: {function_name}")
                return {"error": f"Unknown function: {function_name}"}

        except Exception as e:
            return {"error": f"Function {function_name} execution failed: {str(e)}"}

    async def track_detail_for_guide(self, trackName: str = None) -> Dict[str, Any]:
        
        try:
            # Call the async method with await
            response = await self.backend_service.getCompleteActiveModelData(trackName, None, modelType='track_corner_analysis')

            # Extract the actual model data from the response
            track_corner_data = response.get("data")
            if track_corner_data is None:
                raise Exception("No model data found in the response")

            # Now you can access the data properly
            prediction_result = await self.telemetryMLService.predict_optimal_cornering(trackName, track_corner_data.get("modelData"))

            guidance_instructions = {
                "task": "follow the json_structure to generate racing guidance sentences for car operation techniques in JSON format",
                "json_structure": {
                    "throttle_guidance": [
                        "sentence 1 about throttle technique",
                        "sentence 2 about throttle technique", 
                        "sentence 3 about throttle technique",
                        "sentence 4 about throttle technique"
                    ],
                    "brake_guidance": [
                        "sentence 1 about brake technique",
                        "sentence 2 about brake technique",
                        "sentence 3 about brake technique", 
                        "sentence 4 about brake technique"
                    ],
                    "steering_guidance": [
                        "sentence 1 about steering technique",
                        "sentence 2 about steering technique",
                        "sentence 3 about steering technique",
                        "sentence 4 about steering technique"
                    ]
                },
                "car_operations": {
                    "throttle": {
                        "descriptions": ["gentle throttle", "progressive throttle", "steady throttle", "aggressive throttle"],
                        "techniques": ["maintain traction", "corner exit acceleration", "straight line power", "grip optimization"]
                    },
                    "brake": {
                        "descriptions": ["progressive braking", "trail braking", "threshold braking", "light braking"],
                        "techniques": ["weight transfer", "front grip maintenance", "maximum stopping", "speed adjustment"]
                    },
                    "steering": {
                        "descriptions": ["smooth inputs", "quick corrections", "gradual turn-in", "counter steering"],
                        "techniques": ["tire grip preservation", "oversteer management", "high-speed cornering", "slide control"]
                    }
                },
                "sentence_requirements": [
                    "Generate exactly 12 sentences total (4 for each operation type)",
                    "Each sentence should describe a specific racing technique",
                    "Focus on HOW to operate the car effectively",
                    "Use descriptive technique words from the provided lists",
                    "Make sentences actionable and practical",
                    "No distance measurements or track position references",
                    "Return ONLY the JSON object",
                    "No markdown formatting (```json```)",
                    "No explanatory text",
                    "No code blocks",
                    "Start directly with { and end with }",
                    "Exactly 4 sentences per guidance type"
                ],
                "example_output": {
                    "throttle_guidance": [
                        "Apply gentle throttle at corner apex to maintain traction",
                        "Use progressive throttle increase through corner exit", 
                        "Maintain steady throttle on long straights",
                        "Apply aggressive throttle in dry conditions with good grip"
                    ],
                    "brake_guidance": [
                        "Use progressive braking for smooth weight transfer",
                        "Apply trail braking technique to maintain front grip",
                        "Use threshold braking for maximum stopping power",
                        "Apply light braking for minor speed adjustments"
                    ],
                    "steering_guidance": [
                        "Make smooth steering inputs to preserve tire grip",
                        "Use quick steering corrections for oversteer management",
                        "Apply gradual turn-in for high-speed corners", 
                        "Use counter steering to control rear slides"
                    ]
                }
            }
        except Exception as e:
            raise Exception(f"Failed to enable racing guidance: {str(e)}")
        
        # Return both data for OpenAI and side products for external use
        return {
            # Data for OpenAI's second query
            'status': 'success',
            'message': guidance_instructions,
            
            # Side products (prefixed with _) for external use
            '_guidance_enabled': True,
            '_prediction_result': prediction_result.get("predictions", {}),
            '_skip_openai_processing': True,  # Special flag to skip second OpenAI query if needed
        }
