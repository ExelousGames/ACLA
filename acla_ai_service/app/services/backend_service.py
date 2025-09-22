"""
Backend integration service for communicating with ACLA backend
"""

from typing import Dict, Any, Optional
import httpx
import asyncio
import logging
import json
import os
import tempfile
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
                
        except Exception as e:
            raise Exception(f"Backend function call failed: {str(e)}\n")

    async def get_all_racing_sessions(self, trackName: Optional[str] = None, carName: Optional[str] = None, chunk_size: int = 1000) -> Dict[str, Any]:
        """
        Get all racing sessions from all users in the database
        
        return structure:
        {   "success": True,
            "download_id": "string", 
            "total_sessions": 10,
            "sessions": [  # list of sessions
                {
                    "sessionId": "string",
                    "metadata": { ... },  # session metadata
                    "data": [ ... ],      # list of telemetry data points
                    "total_telemetry_records": 1000
                },
                ...
            ]
            "summary": {
                "total_sessions_retrieved": 10,
                "total_telemetry_records": 10000
            },
        }
        """
        try:
            # Initialize the download to get metadata about all sessions
            init_data = {
                "trackName": trackName,
                "carName": carName,
                "chunkSize": chunk_size
            }

            try:
                # inital the download the sessions
                init_response = await self.call_backend_function("racing-session/download/init", "POST", init_data)

            except Exception as e:
                raise RuntimeError(f"Failed to initialize racing session download: {str(e)}")

            download_id = init_response.get("downloadId")
            if not download_id:
                raise RuntimeError("No download ID received from initialization")
            
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
                    "total_telemetry_records": len(session_chunks)
                })
            
            return {
                "success": True,
                "download_id": download_id,
                "total_sessions": total_sessions,
                "sessions": all_sessions_data, # session id, metadata, data, total_data_points
                "summary": {
                    "total_sessions_retrieved": len(all_sessions_data),
                    "total_telemetry_records": sum(session.get("total_telemetry_records", 0) for session in all_sessions_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving all racing sessions: {str(e)}")
            raise Exception(f"Failed to retrieve racing sessions: {str(e)}")

    async def send_chunked_data(self, data: Dict[str, Any], endpoint: str, chunk_size: int = 1024 * 1024) -> Dict[str, Any]:
        """Send large data in chunks to a backend endpoint, streaming from a temp file to avoid huge in-memory strings."""
        import json
        import uuid
        from math import ceil
        import numpy as np
        from httpx import Timeout
        
        # Temporarily suppress httpx INFO logging for chunked uploads
        httpx_logger = logging.getLogger("httpx")
        original_level = httpx_logger.level
        httpx_logger.setLevel(logging.WARNING)
        

        def convert_numpy_types(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

            
        # Convert numpy types to native Python types
        serializable_data = convert_numpy_types(data)

        # Serialize to a temp file to avoid building the entire JSON string in memory
        tmp_file_path = None
        session_id = str(uuid.uuid4())
        fd, tmp_file_path = tempfile.mkstemp(prefix=f"chunk-upload-{session_id}-", suffix=".json")
        os.close(fd)
        # Write JSON to file (compact separators reduce size)
        with open(tmp_file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, separators=(',', ':'))

        data_size = os.path.getsize(tmp_file_path)
        # First pass: count chunks by characters to match the sending loop
        total_chunks = 0
        with open(tmp_file_path, 'r', encoding='utf-8') as f:
            while True:
                c = f.read(chunk_size)
                if not c:
                    break
                total_chunks += 1
        print(f"Sending data in {total_chunks} chunks (total size: {data_size} bytes, session: {session_id})")
        
        try:
            # Ensure connection and build auth headers once
            if not await self.ensure_connection():
                raise Exception("Failed to establish backend connection")
            headers = self.get_auth_headers()

            url = f"{self.base_url}:{self.base_port}/{endpoint}"
            timeout = Timeout(connect=10.0, read=180.0, write=180.0, pool=180.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Stream-read file in text mode to avoid splitting multibyte characters
                with open(tmp_file_path, 'r', encoding='utf-8') as f:
                    chunk_index = 0
                    while True:
                        # Read approximately chunk_size characters; not exact bytes, but safe for UTF-8
                        chunk_data = f.read(chunk_size)
                        if not chunk_data:
                            break

                        chunk_request = {
                            "sessionId": session_id,
                            "chunkIndex": chunk_index,
                            "totalChunks": total_chunks,
                            "data": chunk_data,
                            "metadata": {
                                "size": len(chunk_data.encode('utf-8'))
                            }
                        }

                        logger.debug(
                            f"Sending chunk {chunk_index + 1}/{total_chunks} (size: {len(chunk_data.encode('utf-8'))} bytes)"
                        )

                        resp = await client.post(url, json=chunk_request, headers=headers)
                        resp.raise_for_status()
                        response = resp.json()

                        if not response.get("success", False):
                            error_msg = response.get("message", "Unknown error")
                            raise Exception(f"Chunk {chunk_index + 1} failed: {error_msg}")

                        if response.get("isComplete", False):
                            logger.info(
                                f"✅ All chunks sent successfully. Final response: {response.get('message', 'Complete')}"
                            )
                            return response

                        chunk_index += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to send chunked data: {str(e)}")
            raise Exception(f"Failed to send chunked data: {str(e)}")
        finally:
            # Restore original httpx logging level
            httpx_logger.setLevel(original_level)
            # Clean up temp file
            try:
                if tmp_file_path and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
            except Exception:
                pass

    async def save_ai_model(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Save AI model results to backend using chunked transfer"""
        print("[INFO] Saving AI model results to backend...")
        logger.info("Saving AI model results to backend...")

        try:
            # Use chunked upload for large data
            response = await self.send_chunked_data(
                data=results, 
                endpoint="ai-model/save",
                chunk_size=512 * 1024  # 512KB chunks
            )
            
            if not response.get("success", False):
                raise Exception(f"Backend rejected data: {response.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save imitation learning results: {str(e)}")
            raise
        return {"success": True}

    async def initGetActiveModelData(self, trackName: Optional[str], carName: Optional[str], modelType: str) -> Dict[str, Any]:
        """Initialize chunked retrieval of active model data from backend"""
        try:
            # Call the prepare-chunked endpoint to initialize the session
            # trackName and carName are query parameters, not path parameters
            endpoint = f"ai-model/active/{modelType}/prepare-chunked"
            
            # Build query parameters
            query_params = []
            if trackName:
                query_params.append(f"trackName={trackName}")
            if carName:
                query_params.append(f"carName={carName}")
            
            if query_params:
                endpoint += "?" + "&".join(query_params)
            
            print(f"[INFO] Initializing active model data retrieval from endpoint: {endpoint}")
            response = await self.call_backend_function(endpoint, "GET")
            
            if "error" in response:
                return response
            
            # Return the session information
            return {
                "success": response.get("success", False),
                "sessionId": response.get("sessionId"),
                "totalChunks": response.get("totalChunks"),
                "message": response.get("message", "Session initialized successfully")
            }
            
        except Exception as e:
            logger.error(f"Error initializing active model data retrieval: {str(e)}")
            return {"error": f"Failed to initialize active model data retrieval: {str(e)}"}

    async def getActiveModelDataChunk(self, sessionId: str, chunkIndex: int) -> Dict[str, Any]:
        """Get a specific chunk from a prepared chunked session"""
        try:
            # Call the chunked-data endpoint to get the specific chunk
            endpoint = f"ai-model/active/chunked-data/{sessionId}/{chunkIndex}"
            response = await self.call_backend_function(endpoint, "GET")
            
            if "error" in response:
                return response
            
            return response
            
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunkIndex} from session {sessionId}: {str(e)}")
            return {"error": f"Failed to retrieve chunk: {str(e)}"}

    async def getCompleteActiveModelData(self, trackName: Optional[str], carName: Optional[str], modelType: str) -> Dict[str, Any]:
        """Get complete active model data by retrieving all chunks
        
        returned result contains dict_keys(['success', 'sessionId', 'totalChunks', 'data', 'message'])
        
        """ 
        try:
            logger.info(f"Starting retrieval of active model data for {trackName}/{carName}/{modelType}")
            
            # Initialize the chunked session
            init_response = await self.initGetActiveModelData(trackName, carName, modelType)
            logger.info(f"Successfully initialized chunked session: {init_response}")
            
            if not init_response.get("success", False):
                logger.error(f"Failed to initialize chunked session: {init_response}")
                return init_response
            
            session_id = init_response.get("sessionId")
            total_chunks = init_response.get("totalChunks", 0)
            
            logger.info(f"Retrieving active model data in {total_chunks} chunks (session: {session_id})")
            
            # Collect all chunks
            complete_data_chunks = []
            
            for chunk_index in range(total_chunks):
                chunk_response = await self.getActiveModelDataChunk(session_id, chunk_index)
                
                if not chunk_response.get("success", False):
                    error_msg = chunk_response.get("message", "Unknown error")
                    logger.error(f"Failed to retrieve chunk {chunk_index}: {error_msg}")
                    return {"error": f"Failed to retrieve chunk {chunk_index}: {error_msg}"}
                
                chunk_data = chunk_response.get("data")
                if chunk_data is not None:
                    complete_data_chunks.append(chunk_data)
                
                logger.debug(f"Retrieved chunk {chunk_index + 1}/{total_chunks}")
            
            # Reconstruct the complete model data by joining all chunks
            # The chunks contain string portions that need to be concatenated and parsed
            logger.info("🔄 Assembling chunks into complete model data...")
            
            try:
                # Join all chunk data strings to reconstruct the original JSON string
                complete_json_string = ''.join(complete_data_chunks)
                # Parse the complete JSON string back to the original data structure
                complete_model_data = json.loads(complete_json_string)
                
                logger.info("✅ Successfully assembled and parsed complete model data")
                
                return {
                    "success": True,
                    "sessionId": session_id,
                    "totalChunks": total_chunks,
                    "data": complete_model_data,
                    "message": "Complete model data retrieved and assembled successfully"
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse assembled chunk data: {str(e)}")
                return {"error": f"Failed to parse assembled chunk data: {str(e)}"}
            except Exception as e:
                logger.error(f"Failed to assemble chunk data: {str(e)}")
                return {"error": f"Failed to assemble chunk data: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Error retrieving complete active model data: {str(e)}")
            return {"error": f"Failed to retrieve complete active model data: {str(e)}"}
    
# Global backend service instance
backend_service = BackendService()
