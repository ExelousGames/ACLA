"""Download a session and its attached recordings to the training workspace."""

from pathlib import Path as FilePath
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from training.storage.session_download import SESSION_DOWNLOAD_DIR, download_session_pair

router = APIRouter(prefix="/racing-session", tags=["session downloads"])
bearer = HTTPBearer(auto_error=False)


def get_backend_url() -> str:
    from app.infra.config import settings

    host = settings.backend_server_ip or "http://backend"
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return f"{host.rstrip('/')}:{settings.backend_proxy_port or '7001'}/"


def get_download_directory() -> FilePath:
    return SESSION_DOWNLOAD_DIR


@router.post("/{session_id}/download", status_code=201)
async def download_session(
    session_id: Annotated[str, Path(pattern=r"^[0-9a-fA-F]{24}$")],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    backend_url: Annotated[str, Depends(get_backend_url)],
    storage_dir: Annotated[FilePath, Depends(get_download_directory)],
) -> dict:
    """Save telemetry and all attached MP4s using the session owner's backend JWT."""
    if credentials is None:
        raise HTTPException(
            status_code=401, detail="Session owner's backend bearer token is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    timeout = httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=60.0)
    try:
        async with httpx.AsyncClient(
            base_url=backend_url, timeout=timeout,
            headers={"Authorization": f"Bearer {credentials.credentials}"},
        ) as client:
            return await download_session_pair(client, session_id, storage_dir)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Backend session download timed out") from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        raise HTTPException(
            status_code=status if status in (401, 403, 404) else 502,
            detail=f"Backend session download failed (HTTP {status})",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Backend session download failed") from exc
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Invalid backend session data: {exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not save session download to storage") from exc
