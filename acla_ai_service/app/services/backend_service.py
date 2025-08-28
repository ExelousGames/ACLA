"""
Backend integration service for communicating with ACLA backend
"""

from typing import Dict, Any, Optional
import httpx
from app.core import settings


class BackendService:
    """Service for backend integration and communication"""
    
    def __init__(self):
        self.base_url = settings.backend_url
    
    async def call_backend_function(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
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
                else:
                    return {"error": f"Unsupported HTTP method: {method}"}
                
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP error {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": f"Backend call failed: {str(e)}"}
    
    async def get_racing_sessions(self, user_id: str, map_name: Optional[str] = None) -> Dict[str, Any]:
        """Get racing sessions from backend"""
        data = {"username": user_id}
        if map_name:
            data["map_name"] = map_name
        
        return await self.call_backend_function("racing-session/sessionbasiclist", "POST", data)
    
    async def get_session_details(self, session_id: str) -> Dict[str, Any]:
        """Get detailed session information"""
        data = {"id": session_id}
        return await self.call_backend_function("racing-session/detailedSessionInfo", "POST", data)
    
    async def save_analysis_results(self, session_id: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Save analysis results to backend"""
        data = {
            "session_id": session_id,
            "analysis_results": analysis_results
        }
        return await self.call_backend_function("analysis/save", "POST", data)
