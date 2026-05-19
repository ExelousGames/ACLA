"""Backend integration package — HTTP adapter to acla_backend (Node)."""
from app.integrations.backend.client import BackendService, backend_service

__all__ = ["BackendService", "backend_service"]
