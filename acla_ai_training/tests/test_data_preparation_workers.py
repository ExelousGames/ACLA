import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from training.pipelines.training import prepare_training_data as cli
from training.pipelines.training.pipeline import parallel


@pytest.mark.asyncio
async def test_process_map_bounds_reads_and_keeps_order(monkeypatch):
    consumed = []
    second_finished = Event()

    def chunks():
        for value in range(7):
            consumed.append(value)
            yield value

    def worker(value):
        if value == 0:
            assert second_finished.wait(timeout=5)
        if value == 1:
            second_finished.set()
        return value * 2

    def executor(**kwargs):
        assert kwargs["max_workers"] == 2
        assert kwargs["mp_context"].get_start_method() == "spawn"
        return ThreadPoolExecutor(max_workers=kwargs["max_workers"])

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", executor)
    results = []
    async for result in parallel.map_chunks_in_processes(worker, chunks(), max_workers=2):
        results.append(result)
        assert len(consumed) == min(len(results) + 1, 7)
    assert results == list(range(0, 14, 2))


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("stage", ["all", "cleanup", "top_laps", "enrich"])
async def test_cli_passes_worker_count_to_preparation(monkeypatch, workers, stage):
    from training.pipelines.training.config import TrainingPipelineConfig

    service = SimpleNamespace(
        telemetry_store=object(), pipeline_config=TrainingPipelineConfig(),
        backend_service=object(), _top_lap_reference_feature_names=[],
    )
    prepare = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(cli, "Full_dataset_TelemetryMLService", lambda **kwargs: service)
    monkeypatch.setattr(cli, "prepare_training_data", prepare)
    monkeypatch.setattr(sys, "argv", ["prepare_training_data", "--stage", stage, "--workers", str(workers)])
    assert await cli.main() == 0
    prepare.assert_awaited_once_with(
        telemetry_store=service.telemetry_store, config=service.pipeline_config,
        backend_service=service.backend_service, top_lap_reference_feature_names=[],
        stage=stage, max_workers=workers,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", ["0", "-1", "1.5"])
async def test_cli_rejects_invalid_workers_before_initializing_services(monkeypatch, workers):
    service = Mock()
    monkeypatch.setattr(cli, "Full_dataset_TelemetryMLService", service)
    monkeypatch.setattr(sys, "argv", ["prepare_training_data", "--workers", workers])
    with pytest.raises(SystemExit) as error:
        await cli.main()
    assert error.value.code == 2
    service.assert_not_called()
