from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import cv2
import httpx
import numpy as np
import pytest
from PIL import Image

from training.image_segmentation import auto_annotation


@pytest.fixture
def backend(monkeypatch):
    service = SimpleNamespace(
        base_url="http://backend", base_port="7001",
        ensure_connection=AsyncMock(return_value=True),
        establish_connection=AsyncMock(return_value=True),
        get_auth_headers=MagicMock(return_value={"Authorization": "Bearer token"}),
    )
    weights = b"trained segmentation weights"
    metadata = {
        "id": "a" * 24, "name": "track-segments", "task": "segment",
        "sha256": hashlib.sha256(weights).hexdigest(), "sizeBytes": len(weights),
    }
    requests = []

    def handle(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer token"
        if request.url.path == "/ai-model/ultralytics/track-vision":
            return httpx.Response(200, json=metadata)
        assert request.url.path == f"/ai-model/ultralytics/{metadata['id']}/file"
        return httpx.Response(200, content=weights)

    client = httpx.AsyncClient

    def transport(handler):
        monkeypatch.setattr(auto_annotation.httpx, "AsyncClient", lambda **kwargs: client(
            **kwargs, transport=httpx.MockTransport(handler),
        ))

    transport(handle)
    return SimpleNamespace(
        service=service, weights=weights, metadata=metadata, requests=requests,
        handle=handle, transport=transport,
    )


def download(backend, tmp_path):
    return asyncio.run(auto_annotation.download_latest_model(
        backend_service=backend.service, cache_dir=tmp_path,
    ))


def test_latest_is_checked_each_time_and_verified_weights_are_reused(backend, tmp_path):
    first, metadata = download(backend, tmp_path)
    assert first.read_bytes() == backend.weights
    assert metadata == backend.metadata
    second, _ = download(backend, tmp_path)
    assert first == second
    assert len(backend.requests) == 3  # metadata + weights, then just metadata

    backend.metadata["id"] = "b" * 24
    newest, _ = download(backend, tmp_path)
    assert newest != first
    assert newest.read_bytes() == backend.weights
    assert len(backend.requests) == 5


def test_corrupt_cached_weights_are_downloaded_again(backend, tmp_path):
    checkpoint, _ = download(backend, tmp_path)
    checkpoint.write_bytes(b"x" * len(backend.weights))
    assert download(backend, tmp_path)[0].read_bytes() == backend.weights
    assert len(backend.requests) == 4


def test_existing_local_backend_version_skips_weight_download(backend, tmp_path):
    checkpoint = tmp_path / f"{backend.metadata['id']}-{backend.metadata['sha256']}.pt"
    checkpoint.write_bytes(backend.weights)

    assert download(backend, tmp_path)[0] == checkpoint
    assert [request.url.path for request in backend.requests] == ["/ai-model/ultralytics/track-vision"]


def test_changed_backend_digest_downloads_new_weights(backend, tmp_path):
    previous, _ = download(backend, tmp_path)
    backend.weights = b"updated segmentation weights"
    backend.metadata.update(sha256=hashlib.sha256(backend.weights).hexdigest(), sizeBytes=len(backend.weights))

    def handle(request):
        if request.url.path.endswith("/file"):
            backend.requests.append(request)
            return httpx.Response(200, content=backend.weights)
        return backend.handle(request)

    backend.transport(handle)

    checkpoint, _ = download(backend, tmp_path)

    assert checkpoint != previous
    assert checkpoint.read_bytes() == backend.weights
    assert len(backend.requests) == 4


@pytest.mark.parametrize("failure", ["hash", "size", "interrupted"])
def test_failed_download_does_not_leave_a_checkpoint(backend, tmp_path, failure):
    def handle(request):
        if request.url.path.endswith("/file"):
            if failure == "interrupted":
                raise httpx.ReadError("interrupted")
            content = b"x" * (len(backend.weights) if failure == "hash" else 2)
            return httpx.Response(200, content=content)
        return backend.handle(request)

    backend.transport(handle)
    with pytest.raises((ValueError, httpx.ReadError)):
        download(backend, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("endpoint", ["/track-vision", "/file"])
def test_expired_authentication_is_refreshed_for_metadata_and_weights(backend, tmp_path, endpoint):
    rejected = False

    def handle(request):
        nonlocal rejected
        if request.url.path.endswith(endpoint) and not rejected:
            rejected = True
            return httpx.Response(401)
        return backend.handle(request)

    backend.transport(handle)
    assert download(backend, tmp_path)[0].read_bytes() == backend.weights
    backend.service.establish_connection.assert_awaited_once()


def test_missing_model_explains_how_to_make_one_available(backend, tmp_path):
    backend.transport(lambda request: httpx.Response(404))
    with pytest.raises(ValueError, match="Train and upload"):
        download(backend, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_failed_login_does_not_try_to_fetch_a_model(backend, tmp_path):
    backend.service.ensure_connection.return_value = False
    with pytest.raises(ConnectionError, match="credentials"):
        download(backend, tmp_path)
    assert backend.requests == []


@pytest.fixture
def predictor(tmp_path, monkeypatch):
    # Class IDs deliberately differ from the editor's label order.
    result = SimpleNamespace(
        names={0: "grass", 1: "track"},
        boxes=SimpleNamespace(cls=np.array([1, 0, 1])),
        masks=SimpleNamespace(xy=[
            np.array([[0, 0], [40, 0], [40, 20]]),
            np.array([[0, 10], [10, 20], [0, 20]]),
            np.array([[0, 0], [1, 1]]),
        ]),
    )
    network = MagicMock(task="segment", names=result.names)
    network.predict.return_value = [result]
    factory = MagicMock(return_value=network)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    fetch = AsyncMock(return_value=(tmp_path / "latest.pt", {"name": "track-segments", "id": "a" * 24}))
    monkeypatch.setattr(auto_annotation, "download_latest_model", fetch)
    return auto_annotation.BackendAutoAnnotator(), network, factory, fetch


def test_predictions_use_checkpoint_labels_and_original_image_polygons(predictor):
    annotator, network, factory, fetch = predictor
    image = Image.new("RGB", (40, 20))
    expected = [
        {"label": "track", "points": [[0, 0], [40, 0], [40, 20]]},
        {"label": "grass", "points": [[0, 10], [10, 20], [0, 20]]},
    ]
    assert annotator.predict(image, ["track", "grass"]) == expected
    assert annotator.predict(image, ["track", "grass"]) == expected
    assert fetch.await_count == 2
    factory.assert_called_once()
    network.predict.assert_called_with(image, conf=0.25, retina_masks=True, verbose=False)

    fetch.return_value = (fetch.return_value[0].with_name("newer.pt"), fetch.return_value[1])
    annotator.predict(image, ["track", "grass"])
    assert factory.call_count == 2


def test_model_label_mismatch_fails_before_inference(predictor):
    annotator, network, _, _ = predictor
    with pytest.raises(ValueError, match="grass.*--labels"):
        annotator.predict(Image.new("RGB", (40, 20)), ["track"])
    network.predict.assert_not_called()


def test_no_detections_produce_no_polygons(predictor):
    annotator, network, _, _ = predictor
    network.predict.return_value[0].masks = None
    assert annotator.predict(Image.new("RGB", (40, 20)), ["track", "grass"]) == []


@pytest.mark.parametrize("corners", [
    [[10, 10], [110, 10], [110, 70], [10, 70]],
    [[10, 10], [110, 10], [110, 30], [40, 30], [40, 70], [10, 70]],
    [[10, 10], [11, 10], [11, 11], [10, 11]],
    [[10, 10], [410, 10], [410, 11], [10, 11]],
])
@pytest.mark.parametrize("closed", [False, True])
def test_dense_predictions_keep_only_corners_without_losing_regions(predictor, corners, closed):
    annotator, network, _, _ = predictor
    points = np.concatenate([
        np.linspace(start, end, 40, endpoint=False)
        for start, end in zip(corners, corners[1:] + corners[:1])
    ])
    # The contour can start halfway along an edge and repeat its closing point.
    points = np.roll(points, 20, axis=0)
    if closed:
        points = np.concatenate([points, points[:1]])
    result = network.predict.return_value[0]
    result.masks.xy = [points]
    result.boxes.cls = np.array([1])

    polygons = annotator.predict(Image.new("RGB", (420, 80)), ["track", "grass"])

    assert len(polygons) == 1
    assert polygons[0]["label"] == "track"
    assert len(polygons[0]["points"]) == len(corners)
    assert {tuple(point) for point in polygons[0]["points"]} == {tuple(point) for point in corners}
    np.testing.assert_array_equal(result.masks.xy[0], points)


def test_curved_prediction_has_fewer_points_with_small_boundary_error(predictor):
    annotator, network, _, _ = predictor
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    points = np.column_stack([150 + 100 * np.cos(angles), 150 + 100 * np.sin(angles)]).astype(np.float32)
    result = network.predict.return_value[0]
    result.masks.xy = [points]
    result.boxes.cls = np.array([0])

    polygons = annotator.predict(Image.new("RGB", (300, 300)), ["track", "grass"])

    assert len(polygons) == 1
    assert polygons[0]["label"] == "grass"
    simplified = np.array(polygons[0]["points"], dtype=np.float32)
    assert 3 <= len(simplified) < len(points) // 10
    assert cv2.contourArea(simplified) / cv2.contourArea(points) > 0.95
    assert max(abs(cv2.pointPolygonTest(simplified, tuple(map(float, point)), True)) for point in points) <= 2.0


@pytest.fixture
def pretrained_download(tmp_path, monkeypatch):
    cache = tmp_path / "pretrained"
    working = tmp_path / "working"
    weights = tmp_path / "weights"
    working.mkdir()
    weights.mkdir()
    monkeypatch.chdir(working)

    def fetch(filename):
        Path(filename).write_bytes(b"pretrained segmentation weights")
        return filename

    downloader = MagicMock(side_effect=fetch)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", SimpleNamespace(SETTINGS={"weights_dir": str(weights)}))
    monkeypatch.setitem(sys.modules, "ultralytics.utils.downloads", SimpleNamespace(attempt_download_asset=downloader))
    return SimpleNamespace(cache=cache, working=working, weights=weights, downloader=downloader)


@pytest.mark.parametrize("location", ["cache", "working", "weights"])
def test_local_yolo26x_weights_are_used_without_downloading(pretrained_download, location):
    local_dir = getattr(pretrained_download, location)
    local_dir.mkdir(exist_ok=True)
    checkpoint = local_dir / "yolo26x-seg.pt"
    checkpoint.write_bytes(b"existing weights")

    assert auto_annotation.resolve_yolo26x_model(pretrained_download.cache) == checkpoint
    pretrained_download.downloader.assert_not_called()


def test_missing_yolo26x_weights_download_once_into_shared_training_cache(pretrained_download):
    checkpoint = auto_annotation.resolve_yolo26x_model(pretrained_download.cache)
    assert checkpoint == pretrained_download.cache / "yolo26x-seg.pt"
    assert checkpoint.read_bytes() == b"pretrained segmentation weights"
    assert auto_annotation.resolve_yolo26x_model(pretrained_download.cache) == checkpoint
    pretrained_download.downloader.assert_called_once()
    assert list(pretrained_download.cache.iterdir()) == [checkpoint]
    assert list(pretrained_download.working.iterdir()) == []


def test_empty_local_yolo26x_weights_are_replaced(pretrained_download):
    pretrained_download.cache.mkdir()
    checkpoint = pretrained_download.cache / "yolo26x-seg.pt"
    checkpoint.touch()

    assert auto_annotation.resolve_yolo26x_model(pretrained_download.cache) == checkpoint
    assert checkpoint.read_bytes() == b"pretrained segmentation weights"
    pretrained_download.downloader.assert_called_once()


def test_failed_yolo26x_download_leaves_no_cached_checkpoint(pretrained_download):
    def interrupted(filename):
        Path(filename).write_bytes(b"partial weights")
        raise OSError("Download interrupted")

    pretrained_download.downloader.side_effect = interrupted
    with pytest.raises(OSError, match="interrupted"):
        auto_annotation.resolve_yolo26x_model(pretrained_download.cache)
    assert list(pretrained_download.cache.iterdir()) == []


@pytest.fixture
def yolo26x_predictor(tmp_path, monkeypatch):
    result = SimpleNamespace(
        names={0: "person", 2: "car"},
        boxes=SimpleNamespace(cls=np.array([2, 0])),
        masks=SimpleNamespace(xy=[
            np.array([[0, 0], [40, 0], [40, 20]]),
            np.array([[0, 10], [10, 20], [0, 20]]),
        ]),
    )
    network = MagicMock(task="segment", names=result.names)
    network.predict.return_value = [result]
    factory = MagicMock(return_value=network)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    resolve = MagicMock(return_value=tmp_path / "yolo26x-seg.pt")
    monkeypatch.setattr(auto_annotation, "resolve_yolo26x_model", resolve)
    backend_fetch = AsyncMock(side_effect=AssertionError("YOLO26x must not contact the backend"))
    monkeypatch.setattr(auto_annotation, "download_latest_model", backend_fetch)
    return auto_annotation.YOLO26xAutoAnnotator(), network, factory, resolve


@pytest.mark.parametrize("labels", [["track", "car"], ["track", "grass"], []])
def test_yolo26x_keeps_all_model_labels_and_reuses_loaded_model(yolo26x_predictor, labels):
    annotator, network, factory, resolve = yolo26x_predictor
    image = Image.new("RGB", (40, 20))

    for _ in range(2):
        assert annotator.predict(image, labels) == [
            {"label": "car", "points": [[0, 0], [40, 0], [40, 20]]},
            {"label": "person", "points": [[0, 10], [10, 20], [0, 20]]},
        ]

    resolve.assert_called_once()
    factory.assert_called_once_with(str(resolve.return_value))
    network.predict.assert_called_with(image, conf=0.25, retina_masks=True, verbose=False)


def test_yolo26x_with_no_detections_returns_no_polygons(yolo26x_predictor):
    annotator, network, _, _ = yolo26x_predictor
    network.predict.return_value[0].masks = None
    assert annotator.predict(Image.new("RGB", (40, 20)), ["car"]) == []


def test_yolo26x_rejects_non_segmentation_checkpoint(yolo26x_predictor):
    annotator, network, _, _ = yolo26x_predictor
    network.task = "detect"
    with pytest.raises(ValueError, match="not a segmentation model"):
        annotator.predict(Image.new("RGB", (40, 20)), ["car"])
    network.predict.assert_not_called()
