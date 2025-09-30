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
import base64
import uuid
import re
import numpy as np
from app.core import settings
from app.models.api_models import ActiveModelData

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
                    
                    # Configure timeout for login request
                    from httpx import Timeout
                    timeout = Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
                    
                    async with httpx.AsyncClient(timeout=timeout) as client:
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
                            
            except asyncio.CancelledError:
                logger.error(f"❌ Backend login was cancelled (attempt {attempt + 1}/{max_retries})")
                self.is_connected = False
                self._connection_established = False
                raise  # Re-raise CancelledError to properly handle cancellation
            except httpx.TimeoutException as e:
                logger.error(f"❌ Backend login timed out: {str(e)} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"🔄 Retrying connection in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                else:
                    self.is_connected = False
                    self._connection_established = False
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
    
    async def call_backend_function(self, endpoint: str, method: str = "GET", data: Optional[Dict] = None, headers: Optional[Dict] = None, timeout_seconds: float = 60.0) -> Dict[str, Any]:
        """Call a backend function with authentication and proper timeout handling"""
        # Ensure we have a valid connection
        if not await self.ensure_connection():
            return {"error": "Failed to establish backend connection"}
        
        # Merge authentication headers with provided headers
        auth_headers = self.get_auth_headers()
        if headers:
            auth_headers.update(headers)
        headers = auth_headers
        
        try:
            # Configure timeout for all requests
            from httpx import Timeout
            timeout = Timeout(connect=10.0, read=timeout_seconds, write=timeout_seconds, pool=timeout_seconds)
            
            async with httpx.AsyncClient(timeout=timeout) as client:
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
                
        except asyncio.CancelledError:
            logger.error(f"Backend request to {endpoint} was cancelled")
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Backend request to {endpoint} timed out after {timeout_seconds}s: {str(e)}")
            return {"error": f"Request timed out after {timeout_seconds} seconds"}
        except httpx.HTTPStatusError as e:
            logger.error(f"Backend request to {endpoint} failed with HTTP {e.response.status_code}: {e.response.text}")
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            logger.error(f"Backend request to {endpoint} failed: {str(e)}")
            raise Exception(f"Backend function call failed: {str(e)}\n")

    async def get_all_racing_sessions_streaming(self, trackName: Optional[str] = None, carName: Optional[str] = None, chunk_size: int = 1000, data_cache=None) -> Dict[str, Any]:
        """
        Stream all racing sessions directly to cache without loading into memory
        
        Args:
            trackName: Optional track name filter
            carName: Optional car name filter  
            chunk_size: Size of chunks to download
            data_cache: HybridDataCache instance to stream data to (uses shared cache if None)
            
        Returns:
            Dictionary with metadata only (no session data in memory)
        """
        if not data_cache:
            # Import shared cache here to avoid circular imports
            from .hybrid_data_cache_service import get_shared_data_cache
            data_cache = get_shared_data_cache()
            
        try:
            # Initialize the download to get metadata about all sessions
            init_data = {
                "trackName": trackName,
                "carName": carName,
                "chunkSize": chunk_size
            }

            try:
                # initial the download the sessions - use longer timeout for data-intensive operations
                init_response = await self.call_backend_function("racing-session/download/init", "POST", init_data, timeout_seconds=120.0)

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
            logger.info(f"Streaming sessions directly to shared cache without loading into memory")
            logger.info(f"Using shared cache for data reuse across all services")
            
            # Create generator and stream sessions directly to cache
            class SessionStreamer:
                def __init__(self, session_metadata, download_id, backend_service):
                    self.session_metadata = session_metadata
                    self.download_id = download_id
                    self.backend_service = backend_service
                
                async def __aiter__(self):
                    """Async iterator for streaming sessions"""
                    for session_meta in self.session_metadata:
                        session_id = session_meta["sessionId"]
                        chunk_count = session_meta["chunkCount"]
                        
                        session_chunks = []
                        
                        # Download all chunks for this session
                        for chunk_index in range(chunk_count):
                            chunk_request = {
                                "downloadId": self.download_id,
                                "sessionId": session_id,
                                "chunkIndex": chunk_index
                            }
                            
                            chunk_response = await self.backend_service.call_backend_function("racing-session/download/chunk", "POST", chunk_request, timeout_seconds=180.0)
                            
                            if "error" in chunk_response:
                                logger.error(f"Failed to download chunk {chunk_index} for session {session_id}: {chunk_response['error']}")
                                continue
                            
                            chunk_data = chunk_response.get("data", [])
                            session_chunks.extend(chunk_data)
                        
                        # Yield session without storing in parent scope
                        yield {
                            "sessionId": session_id,
                            "metadata": session_meta,
                            "data": session_chunks,
                            "total_telemetry_records": len(session_chunks)
                        }
                        
                        # Clear session_chunks to free memory immediately
                        del session_chunks
            
            # Stream to cache
            estimated_size_mb = (total_chunks * chunk_size * 50) / (1024 * 1024)  # Rough estimate
            streamer = SessionStreamer(session_metadata, download_id, self)
            
            cache_success = data_cache.cache_sessions_streaming(
                track_name=trackName or "all_tracks",
                sessions_iterator=streamer,
                estimated_size_mb=estimated_size_mb
            )
            
            if not cache_success:
                raise RuntimeError("Failed to stream sessions to cache")
            
            logger.info(f"Successfully streamed {total_sessions} sessions to cache")
            
            # Return only metadata (no session data in memory)
            return {
                "success": True,
                "download_id": download_id,
                "total_sessions": total_sessions,
                "total_chunks": total_chunks,
                "cached": True,
                "cache_key": trackName or "all_tracks",
                "summary": {
                    "total_sessions_retrieved": total_sessions,
                    "estimated_total_records": total_chunks * chunk_size,
                    "streamed_to_cache": True
                }
            }
            
        except Exception as e:
            logger.error(f"Error streaming racing sessions: {str(e)}")
            raise Exception(f"Failed to stream racing sessions: {str(e)}")

    async def get_all_racing_sessions(self, trackName: Optional[str] = None, carName: Optional[str] = None, chunk_size: int = 1000) -> Dict[str, Any]:
        """
        DEPRECATED: This method loads all sessions into memory. Use get_all_racing_sessions_streaming instead.
        
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
        logger.warning("get_all_racing_sessions is deprecated and loads all data into memory. Use get_all_racing_sessions_streaming instead.")
        
        try:
            # Initialize the download to get metadata about all sessions
            init_data = {
                "trackName": trackName,
                "carName": carName,
                "chunkSize": chunk_size
            }

            try:
                # inital the download the sessions - use longer timeout for data-intensive operations
                init_response = await self.call_backend_function("racing-session/download/init", "POST", init_data, timeout_seconds=120.0)

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
                    
                    chunk_response = await self.call_backend_function("racing-session/download/chunk", "POST", chunk_request, timeout_seconds=180.0)
                    
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
        from math import ceil
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
            elif isinstance(obj, tuple):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif obj is None:
                return None
            elif isinstance(obj, (str, int, float)):
                return obj
            elif hasattr(obj, '__class__') and 'sklearn' in str(type(obj)):
                # If we encounter sklearn objects, this indicates a serialization issue
                raise ValueError(f"Encountered unserialized sklearn object: {type(obj)}. "
                               f"This object should have been serialized before sending to backend.")
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

    async def save_ai_model(self, 
                           model_type: str,
                           track_name: str,
                           car_name: str,
                           model_data: Dict[str, Any],
                           metadata: Optional[Dict[str, Any]] = None,
                           is_active: bool = True) -> Dict[str, Any]:
        """Save AI model results to backend using chunked transfer
        
        Args:
            model_type: Type of the AI model (e.g., "tire_grip_analysis", "imitation_learning")
            track_name: Name of the track the model is for
            car_name: Name of the car the model is for
            model_data: The serialized model data
            metadata: Optional metadata containing model info and timestamps
            is_active: Whether this model should be set as active
        """
        print(f"[INFO] Saving AI model results to backend: {model_type} for {track_name}/{car_name}")
        logger.info(f"Saving AI model results to backend: {model_type} for {track_name}/{car_name}")

        # Structure the data according to the specified format
        structured_data = {
            "modelType": model_type,
            "trackName": track_name,
            "carName": car_name,
            "modelData": model_data,
            "metadata": metadata or {},
            "isActive": is_active
        }

        try:
            # Use chunked upload for large data
            response = await self.send_chunked_data(
                data=structured_data, 
                endpoint="ai-model/save",
                chunk_size=512 * 1024  # 512KB chunks
            )
            
            if not response.get("success", False):
                raise Exception(f"Backend rejected data: {response.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save AI model results: {str(e)}")
            raise
        return {"success": True}

    async def initGetActiveModelData(self, trackName: Optional[str], carName: Optional[str], modelType: str) -> Dict[str, Any]:
        """Initialize chunked retrieval of active model data from backend.
        
        New backend returns the complete structure with metadata immediately,
        plus chunking information for retrieving the modelData field.
        """
        try:
            # Call the prepare-chunked endpoint to get the complete structure
            endpoint = f"ai-model/active/{modelType}/prepare-chunked"
            
            # Build query parameters
            query_params = []
            if trackName:
                query_params.append(f"trackName={trackName}")
            if carName:
                query_params.append(f"carName={carName}")
            
            if query_params:
                endpoint += "?" + "&".join(query_params)
            
            logger.info(f"🔗 Calling new backend endpoint: {endpoint}")
            response = await self.call_backend_function(endpoint, "GET")
            
            if "error" in response:
                return response
            
            # New backend returns { success, data, chunking, message }
            if not response.get("success", False):
                return {"error": response.get("message", "Backend request failed")}
            
            return response  # Return the complete response with data and chunking info
            
        except Exception as e:
            logger.error(f"💥 Error initializing active model data retrieval: {str(e)}")
            return {"error": f"Failed to initialize active model data retrieval: {str(e)}"}

    async def getActiveModelDataChunk(self, sessionId: str, chunkIndex: int, max_retries: int = 3) -> Dict[str, Any]:
        """Get a specific chunk from a prepared chunked session with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Call the chunked-data endpoint to get the specific chunk
                endpoint = f"ai-model/active/chunked-data/{sessionId}/{chunkIndex}"
                response = await self.call_backend_function(endpoint, "GET")
                
                if "error" in response:
                    # If it's a session-related error, don't retry
                    error_msg = response.get("error", "Unknown error")
                    if "session not found" in error_msg.lower() or "expired" in error_msg.lower():
                        return response
                    
                    # For other errors, retry
                    last_error = response.get("error")
                    if attempt < max_retries - 1:
                        logger.warning(f"Chunk {chunkIndex} retrieval failed (attempt {attempt + 1}/{max_retries}): {last_error}")
                        await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                        continue
                    return response
                
                return response
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Error retrieving chunk {chunkIndex} from session {sessionId} (attempt {attempt + 1}/{max_retries}): {last_error}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    continue
        
        return {"error": f"Failed to retrieve chunk after {max_retries} attempts: {last_error}"}

    async def getCompleteActiveModelData(self, trackName: Optional[str], carName: Optional[str], modelType: str) -> ActiveModelData:
        """Get complete active model data - simple and clean approach.
        
        1. Get structure with metadata immediately 
        2. Download chunks and fill modelData
        3. Return complete structure as ActiveModelData
        
        Returns:
            ActiveModelData: Structured model data with all required properties
            
        Raises:
            Exception: If operation fails for any reason
        """ 
        # Temporarily suppress httpx INFO logging for chunked downloads
        httpx_logger = logging.getLogger("httpx")
        original_level = httpx_logger.level
        httpx_logger.setLevel(logging.WARNING)
        
        try:
            # Get the complete structure with metadata from backend
            init_response = await self.initGetActiveModelData(trackName, carName, modelType)
            
            if not init_response.get("success", False):
                error_msg = init_response.get("message", "Failed to initialize")
                raise Exception(f"Backend initialization failed: {error_msg}")
            
            # Extract final structure and chunking info
            final_structure = init_response.get("data", {})
            chunking_info = init_response.get("chunking", {})
            
            session_id = chunking_info.get("sessionId")
            total_chunks = chunking_info.get("totalChunks", 0)
            
            # If no chunks, return structure as ActiveModelData with empty modelData
            if not session_id or total_chunks == 0:
                raise Exception("No chunking information received from backend")
            
            logger.info(f"📦 Will retrieve {total_chunks} chunks (session: {session_id})")
            
            # Retrieve and assemble all chunks
            model_data = await self._retrieve_and_assemble_chunks(session_id, total_chunks)
            if "error" in model_data:
                raise Exception(f"Failed to assemble chunks: {model_data['error']}")
            
            # Fill in the modelData and create structured return
            logger.info("✅ Successfully assembled complete model data with new simplified system, model_data keys: " + ", ".join(model_data.get("data", {}).keys()))
            
            return ActiveModelData(
                modelType=final_structure.get("modelType", modelType),
                trackName=final_structure.get("trackName", trackName or ""),
                carName=final_structure.get("carName", carName or ""),
                isActive=final_structure.get("isActive", True),
                metadata=final_structure.get("metadata", {}),
                modelData=model_data["data"]
            )
            
        except Exception as e:
            logger.error(f"Error retrieving model data: {str(e)}")
            raise Exception(f"Failed to retrieve complete active model data: {str(e)}")
        finally:
            # Restore original httpx logging level
            httpx_logger.setLevel(original_level)

    async def _retrieve_and_assemble_chunks(self, session_id: str, total_chunks: int) -> Dict[str, Any]:
        """Retrieve all chunks and assemble them into model data.
        
        Simple approach: Download base64 chunks, assemble to binary, parse as JSON dict.
        No fallbacks - assembled data must be a valid dict or exception is raised.
        """
        logger.info(f"📦 Retrieving {total_chunks} chunks")
        
        # Create temporary file for binary assembly
        fd, temp_file = tempfile.mkstemp(suffix='.bin', prefix='model_chunks_')
        os.close(fd)
        
        try:
            # Download and assemble all chunks as binary data
            with open(temp_file, 'wb') as f:
                for chunk_index in range(total_chunks):
                    chunk_response = await self.getActiveModelDataChunk(session_id, chunk_index)
                    
                    if not chunk_response.get("success", False):
                        error_msg = chunk_response.get("message", "Unknown error")
                        raise Exception(f"Failed to retrieve chunk {chunk_index}: {error_msg}")
                    
                    chunk_data = chunk_response.get("data")
                    if not isinstance(chunk_data, str):
                        raise Exception(f"Chunk {chunk_index} data must be base64 string, got {type(chunk_data)}")
                    
                    # Decode base64 and write binary data
                    binary_data = base64.b64decode(chunk_data)
                    f.write(binary_data)
                    
                    if chunk_index % 10 == 0 or chunk_index == total_chunks - 1:
                        logger.info(f"📦 Downloaded chunk {chunk_index + 1}/{total_chunks}")
            
            # Parse assembled file as JSON
            file_size = os.path.getsize(temp_file)
            if file_size == 0:
                raise Exception("Assembled file is empty - no data was written")
            
            logger.info(f"📊 Assembled file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
            
            with open(temp_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse JSON - handle "Extra data" by extracting first complete JSON object
            try:
                model_data = json.loads(content)
            except json.JSONDecodeError as json_error:
                if "Extra data" in str(json_error):
                    # Extract first complete JSON object by balancing braces
                    if not content.startswith('{'):
                        raise Exception("Content does not start with JSON object")
                    
                    brace_count = 0
                    end_index = 0
                    
                    for i, char in enumerate(content):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_index = i + 1
                                break
                    
                    if brace_count != 0:
                        raise Exception("Unmatched braces in JSON content")
                    
                    # Parse extracted JSON
                    valid_json = content[:end_index]
                    model_data = json.loads(valid_json)
                    logger.info("🔧 Extracted valid JSON from content with extra data")
                else:
                    raise Exception(f"JSON parsing failed: {json_error}")
            
            # Ensure result is a dictionary
            if not isinstance(model_data, dict):
                raise Exception(f"Assembled data must be a dict, got {type(model_data)}")
            
            logger.info("✅ Successfully assembled model data")
            return {
                "success": True,
                "data": model_data,
                "message": "Model data assembled successfully"
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass


    
# Global backend service instance
backend_service = BackendService()
