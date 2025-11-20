"""
Configuration settings for ACLA AI Service
"""

import os
from typing import Optional, List
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "ACLA AI Service"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API Configuration
    backend_server_ip: Optional[str] = None
    backend_proxy_port: Optional[str] = None
    

    # Backend Authentication
    backend_username: Optional[str] = None
    backend_password: Optional[str] = None
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    
    # Hugging Face Configuration
    hf_api_token: Optional[str] = None
    hf_username: Optional[str] = None
    hf_training_enabled: bool = False
    
    # CORS Configuration
    allowed_origins: List[str] = ["*"]
    allowed_methods: List[str] = ["*"]
    allowed_headers: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
