from __future__ import annotations

import asyncio
import json
import sys
from email import policy
from email.parser import BytesParser
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from training import model_publication
from training.image_segmentation import publication
from training.image_segmentation.__main__ import main
from training.image_segmentation.trainer import train_model


@pytest.fixture
def trained(tmp_path, monkeypatch):
    run = tmp_path / "train7"
    weights = run / "weights"
    weights.mkdir(parents=True)
    checkpoint = weights / "best.pt"
    checkpoint.write_bytes(b"saved segmentation weights\x00\xff")
    last = weights / "last.pt"
    last.write_bytes(b"last checkpoint")
    data = tmp_path / "data.yaml"
    data.write_text("names: [track, car]\n")
    saved_model = SimpleNamespace(
        task="segment", names={1: "car", 0: "track"},
        ckpt={
            "train_args": {"model": "yolo11n-seg.pt", "epochs": 2, "imgsz": 128, "batch": 2},
            "train_metrics": {"metrics/mAP50(M)": 0.75},
        },
    )
    network = MagicMock(task="segment")
    network.trainer = SimpleNamespace(best=checkpoint, last=last)
    factory = MagicMock(side_effect=lambda *args, **kwargs: network if kwargs else saved_model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    return SimpleNamespace(
        checkpoint=checkpoint, last=last, data=data, network=network,
        saved_model=saved_model, factory=factory,
    )


def test_checkpoint_upload_sends_binary_class_id_order_metadata_and_auth(trained, monkeypatch):
    backend = SimpleNamespace(
        base_url="http://backend", base_port="7001",
        ensure_connection=AsyncMock(return_value=True),
        get_auth_headers=lambda: {"Authorization": "Bearer test-token"},
    )
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(201, json={"_id": "saved-model"})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        model_publication.httpx, "AsyncClient",
        lambda **kwargs: client_type(transport=httpx.MockTransport(handle), **kwargs),
    )
    record = asyncio.run(publication.upload_checkpoint(
        trained.checkpoint, name="race-boundaries", backend_service=backend,
    ))

    assert record == {"_id": "saved-model"}
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "http://backend:7001/ai-model/ultralytics"
    assert request.headers["Authorization"] == "Bearer test-token"
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
    assert set(parts) == {"file", "metadata"}
    assert parts["file"].get_filename() == "best.pt"
    assert parts["file"].get_payload(decode=True) == trained.checkpoint.read_bytes()
    assert json.loads(parts["metadata"].get_payload(decode=True)) == {
        "name": "race-boundaries", "task": "segment", "classNames": ["track", "car"],
        "metadata": {
            "trainingRunId": "train7", "baseModel": "yolo11n-seg.pt",
            "epochs": 2, "imgsz": 128, "batch": 2,
            "metrics": {"metrics/mAP50(M)": 0.75},
        },
    }


@pytest.mark.parametrize("problem", ["missing", "empty", "extension", "task", "name", "metrics"])
def test_invalid_checkpoint_fails_before_backend_connection(trained, problem):
    backend = SimpleNamespace(ensure_connection=AsyncMock())
    checkpoint = trained.checkpoint
    name = "track-segments"
    if problem == "missing":
        checkpoint = checkpoint.with_name("missing.pt")
    elif problem == "empty":
        checkpoint.write_bytes(b"")
    elif problem == "extension":
        checkpoint = checkpoint.with_suffix(".json")
    elif problem == "task":
        trained.saved_model.task = "detect"
    elif problem == "name":
        name = " "
    else:
        trained.saved_model.ckpt["train_metrics"] = {"map50": float("nan")}

    with pytest.raises(ValueError):
        asyncio.run(publication.upload_checkpoint(checkpoint, name=name, backend_service=backend))
    backend.ensure_connection.assert_not_awaited()


@pytest.mark.parametrize("best_exists", [True, False])
def test_successful_training_publishes_actual_run_checkpoint(trained, monkeypatch, capsys, best_exists):
    if not best_exists:
        trained.checkpoint.unlink()
    upload = AsyncMock(return_value={"_id": "saved-model"})
    monkeypatch.setattr(publication, "upload_checkpoint", upload)

    result = train_model(trained.data, upload_name="race-boundaries")

    assert result is trained.network.train.return_value
    upload.assert_awaited_once_with(
        trained.checkpoint if best_exists else trained.last, name="race-boundaries",
    )
    assert "Model uploaded to backend: saved-model" in capsys.readouterr().out


def test_failed_training_never_uploads(trained, monkeypatch):
    upload = AsyncMock()
    monkeypatch.setattr(publication, "upload_checkpoint", upload)
    trained.network.train.side_effect = RuntimeError("training failed")
    with pytest.raises(RuntimeError, match="training failed"):
        train_model(trained.data)
    upload.assert_not_awaited()


def test_upload_failure_reports_preserved_checkpoint_without_retry(trained, monkeypatch):
    before = trained.checkpoint.read_bytes()
    upload = AsyncMock(side_effect=httpx.ReadTimeout("upload timed out"))
    monkeypatch.setattr(publication, "upload_checkpoint", upload)

    with pytest.raises(RuntimeError, match="Training completed, but backend upload failed") as error:
        train_model(trained.data)

    assert str(trained.checkpoint) in str(error.value)
    assert trained.checkpoint.read_bytes() == before
    upload.assert_awaited_once()


@pytest.mark.parametrize("command", ["train", "train-labelme"])
def test_training_commands_upload_by_default_and_allow_offline_runs(trained, monkeypatch, command):
    from training.image_segmentation import splits

    monkeypatch.setattr(splits, "prepare_training_dataset", lambda *args, **kwargs: trained.data)
    upload = AsyncMock(return_value={"_id": "saved-model"})
    monkeypatch.setattr(publication, "upload_checkpoint", upload)
    args = [command, "--upload-name", "race-boundaries"]
    if command == "train":
        args += ["--data", str(trained.data)]

    assert main(args) == 0
    upload.assert_awaited_once_with(trained.checkpoint, name="race-boundaries")
    upload.reset_mock()
    assert main([*args, "--no-upload"]) == 0
    upload.assert_not_awaited()


def test_upload_command_publishes_without_retraining(trained, monkeypatch, capsys):
    upload = AsyncMock(return_value={"_id": "saved-model"})
    monkeypatch.setattr(publication, "upload_checkpoint", upload)
    assert main(["upload", str(trained.checkpoint), "--name", "race-boundaries"]) == 0
    upload.assert_awaited_once_with(trained.checkpoint, name="race-boundaries")
    trained.network.train.assert_not_called()
    assert "saved-model" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["upload", "train"])
def test_cli_upload_failure_exits_nonzero(trained, monkeypatch, capsys, command):
    monkeypatch.setattr(publication, "upload_checkpoint", AsyncMock(side_effect=ConnectionError("offline")))
    args = ["upload", str(trained.checkpoint)] if command == "upload" else ["train", "--data", str(trained.data)]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert "offline" in capsys.readouterr().err
