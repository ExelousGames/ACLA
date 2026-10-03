"""Predict editable polygons with backend or pretrained YOLO26x segmentation."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import re
import tempfile

import httpx

from . import WORKSPACE_DIR


@asynccontextmanager
async def _get(client, backend_service, path):
    url = f"{backend_service.base_url}:{backend_service.base_port}{path}"
    for attempt in range(2):
        async with client.stream("GET", url, headers=backend_service.get_auth_headers()) as response:
            if response.status_code == 401 and attempt == 0:
                if not await backend_service.establish_connection():
                    raise ConnectionError("Failed to refresh backend authentication")
                continue
            response.raise_for_status()
            yield response
            return


async def download_latest_model(
    *,
    cache_dir: Path = WORKSPACE_DIR / "storage/image_segmentation/backend_models",
    backend_service=None,
) -> tuple[Path, dict]:
    if backend_service is None:
        from app.integrations.backend.client import backend_service

    if not await backend_service.ensure_connection():
        raise ConnectionError("Failed to establish backend connection. Check the backend URL and AI service credentials.")

    timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with _get(client, backend_service, "/ai-model/ultralytics/track-vision") as response:
                metadata = json.loads(await response.aread())
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise ValueError("No trained segmentation model is available. Train and upload a model first.") from exc
            raise

        model_id = metadata.get("id", "")
        digest = metadata.get("sha256", "")
        if (
            metadata.get("task") != "segment"
            or not re.fullmatch(r"[a-f\d]{24}", model_id)
            or not re.fullmatch(r"[a-f\d]{64}", digest)
            or not isinstance(metadata.get("sizeBytes"), int)
            or metadata["sizeBytes"] <= 0
        ):
            raise ValueError("Backend returned invalid segmentation model metadata")

        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = cache_dir / f"{model_id}-{digest}.pt"
        if checkpoint.is_file() and checkpoint.stat().st_size == metadata["sizeBytes"]:
            with checkpoint.open("rb") as cached:
                if hashlib.file_digest(cached, "sha256").hexdigest() == digest:
                    return checkpoint, metadata

        # Publish to the cache only after the entire download is verified.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".part", delete=False) as target:
                temporary = Path(target.name)
                downloaded_hash = hashlib.sha256()
                async with _get(client, backend_service, f"/ai-model/ultralytics/{model_id}/file") as response:
                    async for chunk in response.aiter_bytes():
                        target.write(chunk)
                        downloaded_hash.update(chunk)
                if target.tell() != metadata["sizeBytes"] or downloaded_hash.hexdigest() != digest:
                    raise ValueError("Downloaded model failed its size or SHA-256 check. Try again.")
            temporary.replace(checkpoint)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return checkpoint, metadata


class BackendAutoAnnotator:
    def __init__(self):
        self._checkpoint = None
        self._network = None
        self.model_name = ""

    def predict(self, image, labels: list[str]) -> list[dict]:
        checkpoint, metadata = asyncio.run(download_latest_model())
        if checkpoint != self._checkpoint:
            from ultralytics import YOLO

            network = YOLO(str(checkpoint))
            if network.task != "segment":
                raise ValueError("The backend checkpoint is not a segmentation model")
            self._network = network
            self._checkpoint = checkpoint
        self.model_name = f"{metadata['name']} ({metadata['id']})"
        unknown = set(self._network.names.values()) - set(labels)
        if unknown:
            raise ValueError(
                "Model labels are missing from the editor labels file: "
                + ", ".join(sorted(unknown))
                + ". Launch with --labels pointing to the model's labels file."
            )

        result = self._network.predict(image, conf=0.25, retina_masks=True, verbose=False)[0]
        return _prediction_polygons(result)


def resolve_yolo26x_model(
    cache_dir: Path = WORKSPACE_DIR / "storage/image_segmentation/pretrained",
) -> Path:
    from ultralytics.utils import SETTINGS
    from ultralytics.utils.downloads import attempt_download_asset

    filename = "yolo26x-seg.pt"
    # Share the training cache, and also honor existing desktop/Ultralytics weights.
    for checkpoint in (cache_dir / filename, Path.cwd() / filename, Path(SETTINGS["weights_dir"]) / filename):
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            return checkpoint.resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / filename
    # An interrupted download must not become the next run's local checkpoint.
    with tempfile.TemporaryDirectory(dir=cache_dir) as temporary:
        downloaded = Path(attempt_download_asset(str(Path(temporary) / filename)))
        downloaded.replace(checkpoint)
    return checkpoint.resolve()


class YOLO26xAutoAnnotator:
    model_name = "YOLO26x (yolo26x-seg.pt)"

    def __init__(self):
        self._network = None

    def predict(self, image, labels: list[str]) -> list[dict]:
        if self._network is None:
            from ultralytics import YOLO

            network = YOLO(str(resolve_yolo26x_model()))
            if network.task != "segment":
                raise ValueError("The YOLO26x checkpoint is not a segmentation model")
            self._network = network

        # Propose regions as "other" so the user can relabel them in the editor.
        result = self._network.predict(
            image, conf=0.25, retina_masks=True, verbose=False,
        )[0]
        return _prediction_polygons(result, label_override="other")


def _prediction_polygons(result, *, label_override: str | None = None) -> list[dict]:
    import cv2

    if result.masks is None:
        return []
    polygons = []
    for index, points in enumerate(result.masks.xy):
        if len(points) < 3:
            continue
        contour = points.astype("float32")
        # Douglas-Peucker removes redundant vertices within two image pixels.
        simplified = cv2.approxPolyDP(contour, epsilon=2.0, closed=True).reshape(-1, 2)
        if len(simplified) < 3 or cv2.contourArea(simplified) == 0:
            # Keep tiny or thin regions: remove only collinear points instead.
            simplified = cv2.approxPolyDP(contour, epsilon=0.0, closed=True).reshape(-1, 2)
        if 3 <= len(simplified) < len(points):
            points = simplified
        polygons.append({
            "label": label_override if label_override is not None else result.names[int(result.boxes.cls[index])],
            "points": points.tolist(),
        })
    return polygons
