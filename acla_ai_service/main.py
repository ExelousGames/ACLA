from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import json
import requests
import os
from datetime import datetime
import asyncio
import httpx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from advanced_analyzer import AdvancedRacingAnalyzer
from telemetry_models import TelemetryFeatures, FeatureProcessor, TelemetryDataModel

app = FastAPI(title="ACLA AI Service", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# In-memory storage for datasets and analysis results
datasets_cache = {}
analysis_cache = {}

# Data Models
class DatasetInfo(BaseModel):
    id: str
    name: str
    columns: List[str]
    shape: tuple
    data_types: Dict[str, str]
    uploaded_at: datetime

class QueryRequest(BaseModel):
    question: str
    dataset_id: Optional[str] = None
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class AnalysisRequest(BaseModel):
    dataset_id: str
    analysis_type: str
    parameters: Optional[Dict[str, Any]] = None

class TelemetryAnalysisRequest(BaseModel):
    session_id: str
    analysis_type: str  # 'comprehensive', 'performance', 'setup', 'telemetry_summary'
    features: Optional[List[str]] = None  # Specific features to analyze
    
class TelemetryDataRequest(BaseModel):
    session_id: str
    telemetry_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class FunctionCallRequest(BaseModel):
    function_name: str
    parameters: Dict[str, Any]
    user_id: Optional[str] = None

# Backend Communication Helper
class BackendClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def call_backend_function(self, endpoint: str, method: str = "GET", data: Dict = None, headers: Dict = None):
        """Call a backend function with authentication"""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/{endpoint}"
                
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data, headers=headers)
                elif method.upper() == "PUT":
                    response = await client.put(url, json=data, headers=headers)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers)
                
                response.raise_for_status()
                return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backend call failed: {str(e)}")

    async def get_racing_sessions(self, user_id: str, map_name: str = None):
        """Get racing sessions from backend"""
        data = {"username": user_id, "map_name": map_name}
        return await self.call_backend_function("racing-session/sessionbasiclist", "POST", data)
    
    async def get_session_details(self, session_id: str):
        """Get detailed session information"""
        data = {"id": session_id}
        return await self.call_backend_function("racing-session/detailedSessionInfo", "POST", data)

backend_client = BackendClient(BACKEND_URL)

# AI Analysis Functions
class DatasetAnalyzer:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        
    def basic_stats(self):
        """Generate basic statistical analysis"""
        stats = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "dtypes": self.df.dtypes.to_dict(),
            "null_counts": self.df.isnull().sum().to_dict(),
            "numeric_stats": {}
        }
        
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            stats["numeric_stats"][col] = {
                "mean": float(self.df[col].mean()),
                "std": float(self.df[col].std()),
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "median": float(self.df[col].median())
            }
        
        return stats
    
    def correlation_analysis(self):
        """Generate correlation matrix for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {"error": "No numeric columns found for correlation analysis"}
        
        correlation_matrix = numeric_df.corr().to_dict()
        return {"correlation_matrix": correlation_matrix}
    
    def performance_analysis(self):
        """Analyze racing performance data with comprehensive telemetry support"""
        analysis = {}
        
        # Use advanced analyzer for enhanced analysis
        advanced_analyzer = AdvancedRacingAnalyzer(self.df)
        
        # Basic performance metrics (legacy support)
        if 'speed' in self.df.columns:
            analysis['speed_stats'] = {
                "avg_speed": float(self.df['speed'].mean()),
                "max_speed": float(self.df['speed'].max()),
                "min_speed": float(self.df['speed'].min())
            }
        elif 'Physics_speed_kmh' in self.df.columns:
            analysis['speed_stats'] = {
                "avg_speed": float(self.df['Physics_speed_kmh'].mean()),
                "max_speed": float(self.df['Physics_speed_kmh'].max()),
                "min_speed": float(self.df['Physics_speed_kmh'].min())
            }
        
        if 'lap_time' in self.df.columns:
            analysis['lap_time_stats'] = {
                "avg_lap_time": float(self.df['lap_time'].mean()),
                "best_lap_time": float(self.df['lap_time'].min()),
                "worst_lap_time": float(self.df['lap_time'].max())
            }
        elif 'Graphics_last_time' in self.df.columns:
            valid_times = self.df[self.df['Graphics_last_time'] > 0]['Graphics_last_time']
            if not valid_times.empty:
                analysis['lap_time_stats'] = {
                    "avg_lap_time": float(valid_times.mean()),
                    "best_lap_time": float(valid_times.min()),
                    "worst_lap_time": float(valid_times.max())
                }
        
        # Advanced telemetry analysis
        try:
            # Get comprehensive telemetry summary
            analysis['telemetry_summary'] = advanced_analyzer.get_telemetry_summary()
            
            # Get performance score with new telemetry features
            analysis['performance_score'] = advanced_analyzer.racing_performance_score()
            
            # Get advanced performance analysis
            analysis['advanced_analysis'] = advanced_analyzer.advanced_performance_analysis()
            
            # Get racing patterns
            analysis['patterns'] = advanced_analyzer.detect_racing_patterns()
            
            # Get sector analysis (enhanced)
            analysis['sector_analysis'] = advanced_analyzer.sector_analysis()
            
            # Get optimal predictions
            analysis['optimal_prediction'] = advanced_analyzer.predict_optimal_lap_time()
            
        except Exception as e:
            analysis['advanced_analysis_error'] = str(e)
            # Fallback to basic analysis
            try:
                analysis['performance_score'] = advanced_analyzer.racing_performance_score()
                analysis['patterns'] = advanced_analyzer.detect_racing_patterns()
            except Exception as e2:
                analysis['fallback_error'] = str(e2)
        
        return analysis

# AI Question Answering
class AIQueryProcessor:
    def __init__(self):
        self.context_memory = {}
    
    async def process_question(self, question: str, dataset_id: str = None, context: Dict = None):
        """Process natural language questions about datasets"""
        
        # Analyze the question to determine intent
        intent = self._analyze_intent(question)
        
        response = {
            "answer": "",
            "data": None,
            "visualization": None,
            "suggested_actions": []
        }
        
        if dataset_id and dataset_id in datasets_cache:
            df = datasets_cache[dataset_id]["dataframe"]
            analyzer = DatasetAnalyzer(df)
            
            if intent == "basic_stats":
                stats = analyzer.basic_stats()
                response["answer"] = self._generate_stats_answer(stats)
                response["data"] = stats
                
            elif intent == "correlation":
                corr_data = analyzer.correlation_analysis()
                response["answer"] = self._generate_correlation_answer(corr_data)
                response["data"] = corr_data
                
            elif intent == "performance":
                perf_data = analyzer.performance_analysis()
                response["answer"] = self._generate_performance_answer(perf_data)
                response["data"] = perf_data
                
            elif intent == "comparison":
                response["answer"] = "I can help you compare different aspects of your data. What specifically would you like to compare?"
                response["suggested_actions"] = ["Compare lap times", "Compare speeds", "Compare by map"]
                
            else:
                response["answer"] = self._generate_general_answer(question, df)
        
        else:
            response["answer"] = "I'd be happy to help analyze your racing data! Please upload a dataset first or specify which dataset you'd like me to analyze."
            response["suggested_actions"] = ["Upload racing session data", "View available datasets"]
        
        return response
    
    def _analyze_intent(self, question: str) -> str:
        """Analyze the intent of the user's question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["stats", "statistics", "summary", "overview"]):
            return "basic_stats"
        elif any(word in question_lower for word in ["correlation", "relationship", "related"]):
            return "correlation"
        elif any(word in question_lower for word in ["performance", "speed", "lap time", "racing"]):
            return "performance"
        elif any(word in question_lower for word in ["compare", "comparison", "vs", "versus"]):
            return "comparison"
        else:
            return "general"
    
    def _generate_stats_answer(self, stats: Dict) -> str:
        """Generate a natural language answer for statistics"""
        shape = stats["shape"]
        return f"Your dataset contains {shape[0]} rows and {shape[1]} columns. The main columns are: {', '.join(stats['columns'][:5])}{'...' if len(stats['columns']) > 5 else ''}. I've calculated basic statistics for all numeric columns."
    
    def _generate_correlation_answer(self, corr_data: Dict) -> str:
        """Generate answer for correlation analysis"""
        if "error" in corr_data:
            return corr_data["error"]
        return "I've calculated the correlation matrix for your numeric data. This shows how different variables are related to each other."
    
    def _generate_performance_answer(self, perf_data: Dict) -> str:
        """Generate answer for performance analysis"""
        if not perf_data:
            return "I couldn't find typical racing performance metrics in your data. Could you tell me more about what performance aspects you'd like to analyze?"
        
        answer = "Here's your racing performance analysis: "
        if "speed_stats" in perf_data:
            speed = perf_data["speed_stats"]
            answer += f"Average speed: {speed['avg_speed']:.2f}, Max speed: {speed['max_speed']:.2f}. "
        
        if "lap_time_stats" in perf_data:
            lap = perf_data["lap_time_stats"]
            answer += f"Best lap time: {lap['best_lap_time']:.2f}, Average lap time: {lap['avg_lap_time']:.2f}."
        
        return answer
    
    def _generate_general_answer(self, question: str, df: pd.DataFrame) -> str:
        """Generate a general answer based on available data"""
        return f"Based on your dataset with {len(df)} records, I can help analyze various aspects. What specific information are you looking for?"

query_processor = AIQueryProcessor()

# API Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ACLA AI Service"}

@app.post("/datasets/upload")
async def upload_dataset(dataset_data: Dict[str, Any]):
    """Upload and process a dataset"""
    try:
        dataset_id = dataset_data.get("id", f"dataset_{datetime.now().timestamp()}")
        
        # Convert data to DataFrame
        if "data" in dataset_data:
            df = pd.DataFrame(dataset_data["data"])
        else:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Store in cache
        datasets_cache[dataset_id] = {
            "dataframe": df,
            "metadata": {
                "id": dataset_id,
                "name": dataset_data.get("name", f"Dataset {dataset_id}"),
                "columns": list(df.columns),
                "shape": df.shape,
                "data_types": df.dtypes.to_dict(),
                "uploaded_at": datetime.now()
            }
        }
        
        # Perform initial analysis
        analyzer = DatasetAnalyzer(df)
        analysis = analyzer.basic_stats()
        analysis_cache[dataset_id] = analysis
        
        return {
            "dataset_id": dataset_id,
            "message": "Dataset uploaded successfully",
            "metadata": datasets_cache[dataset_id]["metadata"],
            "initial_analysis": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")

@app.get("/datasets")
async def list_datasets():
    """List all available datasets"""
    datasets = []
    for dataset_id, dataset_info in datasets_cache.items():
        datasets.append(dataset_info["metadata"])
    return {"datasets": datasets}

@app.post("/query")
async def process_query(request: QueryRequest):
    """Process natural language queries about datasets"""
    try:
        response = await query_processor.process_question(
            request.question, 
            request.dataset_id, 
            request.context
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/analyze")
async def analyze_dataset(request: AnalysisRequest):
    """Perform specific analysis on a dataset"""
    try:
        if request.dataset_id not in datasets_cache:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        df = datasets_cache[request.dataset_id]["dataframe"]
        analyzer = DatasetAnalyzer(df)
        
        if request.analysis_type == "basic_stats":
            result = analyzer.basic_stats()
        elif request.analysis_type == "correlation":
            result = analyzer.correlation_analysis()
        elif request.analysis_type == "performance":
            result = analyzer.performance_analysis()
        else:
            raise HTTPException(status_code=400, detail="Unsupported analysis type")
        
        return {"analysis_type": request.analysis_type, "result": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/backend/call")
async def call_backend_function(request: FunctionCallRequest):
    """Call backend functions from AI service"""
    try:
        if request.function_name == "get_racing_sessions":
            result = await backend_client.get_racing_sessions(
                request.parameters.get("user_id"),
                request.parameters.get("map_name")
            )
        elif request.function_name == "get_session_details":
            result = await backend_client.get_session_details(
                request.parameters.get("session_id")
            )
        else:
            raise HTTPException(status_code=400, detail="Unsupported function")
        
        return {"function": request.function_name, "result": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend function call failed: {str(e)}")

@app.post("/racing-session/analyze")
async def analyze_racing_session(session_data: Dict[str, Any]):
    """Analyze racing session data from backend"""
    try:
        # Convert racing session data to DataFrame
        if "session_data" in session_data:
            df = pd.DataFrame(session_data["session_data"])
            
            # Create dataset entry
            session_id = session_data.get("session_id", f"session_{datetime.now().timestamp()}")
            datasets_cache[session_id] = {
                "dataframe": df,
                "metadata": {
                    "id": session_id,
                    "name": f"Racing Session {session_data.get('session_name', session_id)}",
                    "columns": list(df.columns),
                    "shape": df.shape,
                    "data_types": df.dtypes.to_dict(),
                    "uploaded_at": datetime.now()
                }
            }
            
            # Perform racing-specific analysis
            analyzer = DatasetAnalyzer(df)
            analysis = analyzer.performance_analysis()
            
            return {
                "session_id": session_id,
                "analysis": analysis,
                "suggestions": [
                    "Ask about lap time trends",
                    "Compare speed across different track sections",
                    "Analyze acceleration patterns",
                    "Get performance score and recommendations",
                    "Detect racing patterns"
                ]
            }
        else:
            raise HTTPException(status_code=400, detail="No session data provided")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Racing session analysis failed: {str(e)}")

@app.post("/racing-session/patterns")
async def detect_racing_patterns(request: Dict[str, str]):
    """Detect patterns in racing session data"""
    try:
        session_id = request.get("session_id")
        if not session_id or session_id not in datasets_cache:
            raise HTTPException(status_code=404, detail="Racing session not found")
        
        df = datasets_cache[session_id]["dataframe"]
        advanced_analyzer = AdvancedRacingAnalyzer(df)
        patterns = advanced_analyzer.detect_racing_patterns()
        
        return {
            "session_id": session_id,
            "patterns": patterns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern detection failed: {str(e)}")

@app.post("/racing-session/performance-score")
async def get_performance_score(request: Dict[str, str]):
    """Get comprehensive performance score for racing session"""
    try:
        session_id = request.get("session_id")
        if not session_id or session_id not in datasets_cache:
            raise HTTPException(status_code=404, detail="Racing session not found")
        
        df = datasets_cache[session_id]["dataframe"]
        advanced_analyzer = AdvancedRacingAnalyzer(df)
        score = advanced_analyzer.racing_performance_score()
        
        return {
            "session_id": session_id,
            "performance_score": score
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance score calculation failed: {str(e)}")

@app.post("/racing-session/sector-analysis")
async def get_sector_analysis(request: Dict[str, str]):
    """Get sector-wise performance analysis"""
    try:
        session_id = request.get("session_id")
        if not session_id or session_id not in datasets_cache:
            raise HTTPException(status_code=404, detail="Racing session not found")
        
        df = datasets_cache[session_id]["dataframe"]
        advanced_analyzer = AdvancedRacingAnalyzer(df)
        sector_analysis = advanced_analyzer.sector_analysis()
        
        return {
            "session_id": session_id,
            "sector_analysis": sector_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sector analysis failed: {str(e)}")

@app.post("/racing-session/optimal-prediction")
async def predict_optimal_lap_time(request: Dict[str, str]):
    """Predict optimal lap time for racing session"""
    try:
        session_id = request.get("session_id")
        if not session_id or session_id not in datasets_cache:
            raise HTTPException(status_code=404, detail="Racing session not found")
        
        df = datasets_cache[session_id]["dataframe"]
        advanced_analyzer = AdvancedRacingAnalyzer(df)
        prediction = advanced_analyzer.predict_optimal_lap_time()
        
        return {
            "session_id": session_id,
            "optimal_prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimal lap time prediction failed: {str(e)}")

@app.post("/telemetry/upload")
async def upload_telemetry_data(request: TelemetryDataRequest):
    """Upload comprehensive telemetry data with all AC Competizione features"""
    try:
        # Convert telemetry data to DataFrame
        telemetry_df = pd.DataFrame(request.telemetry_data)
        
        # Validate telemetry features
        feature_processor = FeatureProcessor(telemetry_df)
        validation_result = feature_processor.validate_features()
        
        # Process the data
        processed_df = feature_processor.prepare_for_analysis()
        
        # Store in cache with enhanced metadata
        datasets_cache[request.session_id] = {
            "dataframe": processed_df,
            "raw_dataframe": telemetry_df,
            "metadata": {
                "id": request.session_id,
                "name": f"Telemetry Session {request.session_id}",
                "columns": list(telemetry_df.columns),
                "shape": telemetry_df.shape,
                "data_types": telemetry_df.dtypes.to_dict(),
                "uploaded_at": datetime.now(),
                "telemetry_type": "ac_competizione",
                "feature_validation": validation_result
            }
        }
        
        # Perform initial telemetry analysis
        analyzer = AdvancedRacingAnalyzer(processed_df)
        telemetry_summary = analyzer.get_telemetry_summary()
        
        return {
            "session_id": request.session_id,
            "message": "Telemetry data uploaded successfully",
            "feature_validation": validation_result,
            "telemetry_summary": telemetry_summary,
            "available_analyses": [
                "comprehensive_performance",
                "setup_optimization",
                "tyre_management", 
                "sector_analysis",
                "racing_patterns"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Telemetry upload failed: {str(e)}")

@app.post("/telemetry/analyze")
async def analyze_telemetry(request: TelemetryAnalysisRequest):
    """Perform comprehensive telemetry analysis"""
    try:
        if request.session_id not in datasets_cache:
            raise HTTPException(status_code=404, detail="Telemetry session not found")
        
        df = datasets_cache[request.session_id]["dataframe"]
        analyzer = AdvancedRacingAnalyzer(df)
        
        analysis_result = {}
        
        if request.analysis_type == "comprehensive":
            analysis_result = analyzer.advanced_performance_analysis()
            analysis_result["performance_score"] = analyzer.racing_performance_score()
            analysis_result["telemetry_summary"] = analyzer.get_telemetry_summary()
            
        elif request.analysis_type == "performance":
            analysis_result = analyzer.advanced_performance_analysis()
            analysis_result["performance_score"] = analyzer.racing_performance_score()
            
        elif request.analysis_type == "setup":
            performance_analysis = analyzer.advanced_performance_analysis()
            if "setup_analysis" in performance_analysis:
                analysis_result = performance_analysis["setup_analysis"]
            else:
                analysis_result = {"error": "Setup data not available for analysis"}
                
        elif request.analysis_type == "telemetry_summary":
            analysis_result = analyzer.get_telemetry_summary()
            
        else:
            raise HTTPException(status_code=400, detail="Unsupported analysis type")
        
        return {
            "session_id": request.session_id,
            "analysis_type": request.analysis_type,
            "result": analysis_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Telemetry analysis failed: {str(e)}")

@app.get("/telemetry/features")
async def get_telemetry_features():
    """Get information about available telemetry features"""
    features = TelemetryFeatures()
    
    return {
        "total_features": len(features.get_all_features()),
        "feature_categories": {
            "physics": {
                "count": len(features.PHYSICS_FEATURES),
                "features": features.PHYSICS_FEATURES
            },
            "graphics": {
                "count": len(features.GRAPHICS_FEATURES),
                "features": features.GRAPHICS_FEATURES
            },
            "static": {
                "count": len(features.STATIC_FEATURES),
                "features": features.STATIC_FEATURES
            }
        },
        "performance_critical": features.get_performance_critical_features(),
        "setup_features": features.get_setup_features(),
        "damage_features": features.get_damage_features()
    }

@app.post("/telemetry/validate")
async def validate_telemetry_data(telemetry_data: Dict[str, Any]):
    """Validate telemetry data structure and feature coverage"""
    try:
        # Convert to DataFrame for validation
        df = pd.DataFrame(telemetry_data)
        
        # Process and validate
        feature_processor = FeatureProcessor(df)
        validation_result = feature_processor.validate_features()
        performance_metrics = feature_processor.extract_performance_metrics()
        
        return {
            "validation_result": validation_result,
            "data_quality": {
                "total_records": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(df.select_dtypes(include=['number']).columns),
                "missing_values": df.isnull().sum().sum()
            },
            "performance_metrics": performance_metrics,
            "recommendations": [
                f"Feature coverage: {validation_result['coverage_percentage']:.1f}%",
                f"Physics features: {validation_result['physics_coverage']}/{len(TelemetryFeatures.PHYSICS_FEATURES)}",
                f"Graphics features: {validation_result['graphics_coverage']}/{len(TelemetryFeatures.GRAPHICS_FEATURES)}",
                f"Static features: {validation_result['static_coverage']}/{len(TelemetryFeatures.STATIC_FEATURES)}"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Telemetry validation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
