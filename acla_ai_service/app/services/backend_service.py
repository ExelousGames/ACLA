"""
Backend integration service for communicating with ACLA backend
"""

from typing import Dict, Any, Optional
import httpx
import asyncio
import logging
from app.core import settings

logger = logging.getLogger(__name__)


class BackendService:
    """Service for backend integration and communication"""
    
    def __init__(self):
        # Construct proper URL with protocol
        server_ip = settings.backend_server_ip or "acla_backend_c"
        if not server_ip.startswith(('http://', 'https://')):
            # For development, use http. For production, this should be https
            server_ip = f"http://{server_ip}"
        
        self.base_url = server_ip
        self.base_port = settings.backend_proxy_port
        self.username = settings.backend_username
        self.password = settings.backend_password
        self.jwt_token: Optional[str] = None
        self.is_connected: bool = False
        self._session_lock = asyncio.Lock()
        self._connection_established = False
    
    async def establish_connection(self, max_retries: int = 3) -> bool:
        """Establish secure connection to backend by logging in and obtaining JWT token"""
        if not self.username or not self.password:
            logger.warning("Backend credentials not configured. Connection will fail.")
            return False
        
        for attempt in range(max_retries):
            try:
                
                # Ensure only one login attempt at a time
                async with self._session_lock:
                    login_data = {
                        "email": self.username,
                        "password": self.password
                    }
                    
                    async with httpx.AsyncClient() as client:
                        url = f"{self.base_url}:{self.base_port}/userinfo/auth/login"
                        
                        response = await client.post(url, json=login_data)
                        
                        # Raise for status to catch HTTP errors
                        response.raise_for_status()
                        
                        auth_response = response.json()
                        self.jwt_token = auth_response.get("access_token")
                        
                        if self.jwt_token:
                            self.is_connected = True
                            self._connection_established = True
                            logger.info("✅ Successfully connected to backend and obtained JWT token")
                            return True
                        else:
                            logger.error("❌ Login successful but no JWT token received")
                            return False
                            
            except httpx.HTTPStatusError as e:
                logger.error(f"❌ Backend login failed with HTTP {e.response.status_code}: {e.response.text}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"🔄 Retrying connection in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    self.is_connected = False
                    self._connection_established = False
                    return False
            except Exception as e:
                logger.error(f"❌ Backend connection failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"🔄 Retrying connection in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    self.is_connected = False
                    self._connection_established = False
                    return False
        
        return False
    
    async def ensure_connection(self) -> bool:
        """Ensure we have a valid connection, reconnect if needed"""
        if not self.is_connected or not self.jwt_token:
            return await self.establish_connection()
        return True
    
    @property
    def connection_established(self) -> bool:
        """Check if connection was successfully established at least once"""
        return self._connection_established
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers with JWT token"""
        if self.jwt_token:
            return {"Authorization": f"Bearer {self.jwt_token}"}
        return {}
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status"""
        return {
            "connected": self.is_connected,
            "connection_established": self._connection_established,
            "has_token": bool(self.jwt_token),
            "backend_url": self.base_url,
            "username_configured": bool(self.username)
        }
    
    async def call_backend_function(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Call a backend function with authentication"""
        # Ensure we have a valid connection
        if not await self.ensure_connection():
            return {"error": "Failed to establish backend connection"}
        
        # Merge authentication headers with provided headers
        auth_headers = self.get_auth_headers()
        if headers:
            auth_headers.update(headers)
        headers = auth_headers
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}:{self.base_port}/{endpoint}"
                
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
            raise 
        except Exception as e:
            raise

    async def get_racing_sessions(self, user_id: str, map_name: Optional[str] = None) -> Dict[str, Any]:
        """Get racing sessions from backend"""
        data = {"username": user_id}
        if map_name:
            data["map_name"] = map_name
        
        return await self.call_backend_function("racing-session/sessionbasiclist", "POST", data)

    async def get_all_racing_sessions(self, trackName: str, carName: str, chunk_size: int = 1000) -> Dict[str, Any]:
        """Get all racing sessions from all users in the database"""
        try:
            # Initialize the download to get metadata about all sessions
            init_data = {
                "trackName": trackName,
                "carName": carName,
                "chunkSize": chunk_size
            }
            
            # inital the download the sessions
            init_response = await self.call_backend_function("racing-session/download/init", "POST", init_data)
            
            if "error" in init_response:
                return init_response
            
            download_id = init_response.get("downloadId")
            if not download_id:
                return {"error": "No download ID received from initialization"}
            
            # Get all session metadata
            session_metadata = init_response.get("sessionMetadata", [])
            total_sessions = init_response.get("totalSessions", 0)
            total_chunks = init_response.get("totalChunks", 0)
            
            logger.info(f"Initialized download for {total_sessions} sessions with {total_chunks} total chunks")
            
            # Collect all session data
            all_sessions_data = []
            
            # Download each session's chunks
            for session_meta in session_metadata:
                session_id = session_meta["sessionId"]
                chunk_count = session_meta["chunkCount"]
                
                session_chunks = []
                
                # Download all chunks for this session
                for chunk_index in range(chunk_count):
                    chunk_request = {
                        "downloadId": download_id,
                        "sessionId": session_id,
                        "chunkIndex": chunk_index
                    }
                    
                    chunk_response = await self.call_backend_function("racing-session/download/chunk", "POST", chunk_request)
                    
                    if "error" in chunk_response:
                        logger.error(f"Failed to download chunk {chunk_index} for session {session_id}: {chunk_response['error']}")
                        continue
                    
                    chunk_data = chunk_response.get("data", [])
                    session_chunks.extend(chunk_data)
                
                # Add complete session data
                all_sessions_data.append({
                    "sessionId": session_id,
                    "metadata": session_meta,
                    "data": session_chunks,
                    "total_records": len(session_chunks)
                })
            
            return {
                "success": True,
                "download_id": download_id,
                "total_sessions": total_sessions,
                "sessions": all_sessions_data,
                "summary": {
                    "total_sessions_retrieved": len(all_sessions_data),
                    "total_records": sum(len(session["data"]) for session in all_sessions_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving all racing sessions: {str(e)}")
            return {"error": f"Failed to retrieve all racing sessions: {str(e)}"}
    
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

    async def send_chunked_data(self, data: Dict[str, Any], endpoint: str, chunk_size: int = 1024 * 1024) -> Dict[str, Any]:
        """Send large data in chunks to a backend endpoint"""
        import json
        import uuid
        from math import ceil
        
        # Serialize data to JSON string to calculate size
        json_data = json.dumps(data)
        data_size = len(json_data.encode('utf-8'))
        
        # Calculate number of chunks needed
        total_chunks = max(1, ceil(data_size / chunk_size))
        session_id = str(uuid.uuid4())
        
        print(f"Sending data in {total_chunks} chunks (total size: {data_size} bytes, session: {session_id})")
        
        try:
            # Split data into chunks
            for chunk_index in range(total_chunks):
                start_pos = chunk_index * chunk_size
                end_pos = min(start_pos + chunk_size, len(json_data))
                chunk_data = json_data[start_pos:end_pos]
                
                # Try to parse as valid JSON if it's the complete data in one chunk
                if total_chunks == 1:
                    chunk_payload = data
                else:
                    # For multi-chunk, send the raw string chunk
                    chunk_payload = chunk_data
                
                # Create chunk data structure
                chunk_request = {
                    "sessionId": session_id,
                    "chunkIndex": chunk_index,
                    "totalChunks": total_chunks,
                    "data": chunk_payload,
                    "metadata": {
                        "timestamp": None,  # Will be set by backend
                        "size": len(chunk_data.encode('utf-8'))
                    }
                }
                
                logger.debug(f"Sending chunk {chunk_index + 1}/{total_chunks} (size: {len(chunk_data.encode('utf-8'))} bytes)")
                
                # Send chunk
                response = await self.call_backend_function(endpoint, "POST", chunk_request)
                
                if not response.get("success", False):
                    error_msg = response.get("message", "Unknown error")
                    raise Exception(f"Chunk {chunk_index + 1} failed: {error_msg}")
                
                # Log progress
                if response.get("isComplete", False):
                    logger.info(f"✅ All chunks sent successfully. Final response: {response.get('message', 'Complete')}")
                    return response
                else:
                    received = response.get("receivedChunks", chunk_index + 1)
                    logger.debug(f"Chunk {chunk_index + 1}/{total_chunks} sent successfully ({received}/{total_chunks} received)")
            
            return {"success": True, "message": "All chunks sent", "sessionId": session_id}
            
        except Exception as e:
            logger.error(f"❌ Failed to send chunked data: {str(e)}")
            raise

    async def save_imitation_learning_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Save imitation learning results to backend using chunked transfer"""
        print("[INFO] Saving imitation learning results to backend...")
        logger.info("Saving imitation learning results to backend...")
        
        try: 
            # Use chunked upload for large data
            response = await self.send_chunked_data(
                data=results, 
                endpoint="ai-model/imitation-learning/save",
                chunk_size=512 * 1024  # 512KB chunks
            )
            
            if not response.get("success", False):
                raise Exception(f"Backend rejected data: {response.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save imitation learning results: {str(e)}")
            raise

        logger.info("✅ Imitation learning results saved successfully")
        return {"success": True}


# Global backend service instance
backend_service = BackendService()
