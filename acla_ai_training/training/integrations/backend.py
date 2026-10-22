"""Authenticated dataset downloads owned by the training workspace.

Copied from the AI service backend client, keeping only the methods required by
position preparation. Backend routes, authentication, and chunk formats match
that client without importing the live service package.
"""

from typing import Any, Dict, Optional
import asyncio
import base64
import json
import logging
import os
import tempfile

import httpx
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """Backend connection settings, using the existing environment names."""

    backend_server_ip: Optional[str] = None
    backend_proxy_port: Optional[str] = None
    ai_service_username: Optional[str] = None
    ai_service_password: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


class ActiveModelData(BaseModel):
    """Complete active model data returned by the backend."""

    modelType: str
    isActive: bool
    metadata: Dict[str, Any]
    modelData: Dict[str, Any]


settings = BackendSettings()
logger = logging.getLogger(__name__)


class BackendService:
    """Authenticated session, video, and reference-model downloads."""

    def __init__(self):
        # Construct proper URL with protocol
        server_ip = settings.backend_server_ip or "acla_backend_c"
        if not server_ip.startswith(('http://', 'https://')):
            # For development, use http. For production, this should be https
            server_ip = f"http://{server_ip}"

        self.base_url = server_ip
        self.base_port = settings.backend_proxy_port
        self.username = settings.ai_service_username
        self.password = settings.ai_service_password
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
                    timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)

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

    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers with JWT token"""
        if self.jwt_token:
            return {"Authorization": f"Bearer {self.jwt_token}"}
        return {}

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
            timeout = httpx.Timeout(connect=10.0, read=timeout_seconds, write=timeout_seconds, pool=timeout_seconds)

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

    async def iter_session_video(self, session_id: str, video_index: int):
        """Stream a session recording using the configured backend account."""
        if not await self.ensure_connection():
            raise ConnectionError("Failed to establish backend connection")

        timeout = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=60.0)
        url = f"{self.base_url}:{self.base_port}/racing-session/{session_id}/videos/{video_index}"
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("GET", url, headers=self.get_auth_headers()) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    yield chunk

    async def initGetActiveModelData(self, modelType: str) -> Dict[str, Any]:
        """Initialize chunked retrieval of active model data from backend.

        New backend returns the complete structure with metadata immediately,
        plus chunking information for retrieving the modelData field.
        """
        try:
            # Call the prepare-chunked endpoint to get the complete structure
            endpoint = f"ai-model/active/{modelType}/prepare-chunked"

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

    async def getCompleteActiveModelData(self, modelType: str) -> ActiveModelData:
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
            init_response = await self.initGetActiveModelData(modelType)

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


backend_service = BackendService()
