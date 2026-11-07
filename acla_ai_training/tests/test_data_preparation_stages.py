from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.shared.telemetry import FeatureProcessor
from training.pipelines.training import pipeline
from training.pipelines.training.config import TrainingPipelineConfig
from training.pipelines.training.pipeline import cleaning, enrich
from training.pipelines.training.pipeline.selection import _iter_lap_candidates, select_top_laps


class TelemetryStore:
    def __init__(self, caches=None):
        self.caches = deepcopy(caches or {})

    def has_cached_data(self, cache_key):
        return bool(self.caches.get(cache_key))

    def clear_cache(self, cache_key):
        self.caches.pop(cache_key, None)

    def get_cached_data_chunks(self, cache_key, include_ids=False):
        chunks = self.caches.get(cache_key, {}).items()
        return iter((rows, chunk_id) if include_ids else rows for chunk_id, rows in chunks)

    async def cache_chunks_streaming(self, cache_key, chunks_iterator):
        async for rows, chunk_id in chunks_iterator:
            self.caches.setdefault(cache_key, {})[chunk_id] = deepcopy(rows)
        return self.has_cached_data(cache_key)


def lap(lap_time=10_000.0, *, game="acc", car="car-a", grip=2, valid=1, positions=None):
    if positions is None:
        positions = [i / 20 for i in range(21)]
    return [
        {
            "Static_track": "spa",
            "Static_car_model": car,
            "game_recorded_from": game,
            "Graphics_track_grip_status": grip,
            "Graphics_completed_lap": 1,
            "Graphics_is_valid_lap": valid,
            "Graphics_normalized_car_position": position,
            "Graphics_current_time": i * lap_time / (len(positions) - 1),
            "Physics_speed_kmh": 100.0,
        }
        for i, position in enumerate(positions)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stages", [("all",), ("cleanup", "top_laps", "enrich")])
@pytest.mark.parametrize("workers", [1, 2])
async def test_preparation_selects_persisted_cleaned_laps_before_enrichment(monkeypatch, stages, workers):
    cfg = TrainingPipelineConfig()
    source = {
        "slow": lap(20_000), "fast": lap(10_000), "invalid": lap(5000, valid=0),
        "fragment": lap(1000, positions=[0.02, 0.01, 0.99, 0.98]),
    }
    store = TelemetryStore({cfg.session_data_cache_key: source})
    backend = SimpleNamespace(get_all_racing_sessions_streaming=AsyncMock(return_value={"success": True}))
    executor = Mock(side_effect=ThreadPoolExecutor)
    monkeypatch.setattr(cleaning, "ProcessPoolExecutor", executor)
    enriched_inputs = []

    async def enrich(cache_key, **kwargs):
        assert kwargs["max_workers"] == workers
        assert cache_key == cfg.processed_session_data_cache_key
        assert store.caches[cfg.top_laps_cache_key] == {
            "acc|spa|car-a|grip2": [store.caches[cache_key]["fast"]],
        }
        cleaned = deepcopy(store.caches[cache_key])
        enriched_inputs.append(cleaned)
        store.caches[cfg.enriched_sessions_cache_key] = {
            chunk_id: [{**row, "expert_optimal_speed": 150.0} for row in records]
            for chunk_id, records in cleaned.items()
        }

    monkeypatch.setattr(pipeline, "enriched_contextual_data", enrich)
    for stage in stages:
        result = await pipeline.prepare_training_data(
            telemetry_store=store, config=cfg, backend_service=backend,
            top_lap_reference_feature_names=list(source["fast"][0]), stage=stage,
            max_workers=workers,
        )
        assert result["success"]
        if stage == "cleanup":
            assert not store.has_cached_data(cfg.top_laps_cache_key)
            assert not enriched_inputs
        if stage == "top_laps":
            assert not enriched_inputs
    backend.get_all_racing_sessions_streaming.assert_awaited_once()
    executor.assert_called_once_with(max_workers=workers)
    cleaned = store.caches[cfg.processed_session_data_cache_key]
    assert enriched_inputs == [cleaned]
    assert set(cleaned) == set(source)
    assert [row["Graphics_current_time"] for row in cleaned["fast"]] == list(range(0, 10_001, 500))
    assert [row["Graphics_current_time"] for row in cleaned["slow"]] == list(range(0, 20_001, 500))
    assert store.caches[cfg.session_data_cache_key] == source
    assert all(
        row["expert_optimal_speed"] == 150.0
        for records in store.caches[cfg.enriched_sessions_cache_key].values()
        for row in records
    )


@pytest.mark.asyncio
async def test_cleanup_does_not_require_full_laps_and_invalidates_later_stages(monkeypatch):
    cfg = TrainingPipelineConfig()
    partial_lap = [lap()[0]]
    store = TelemetryStore({
        cfg.session_data_cache_key: {"partial": partial_lap},
        cfg.processed_session_data_cache_key: {"old": lap()},
        cfg.top_laps_cache_key: {"old": [lap()]},
        cfg.enriched_sessions_cache_key: {"old": lap()},
        cfg.segments_cache_key: {"old": [lap()]},
    })
    monkeypatch.setattr(cleaning, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(FeatureProcessor, "split_into_laps", lambda *_: pytest.fail("cleanup selected laps"))
    result = await pipeline.prepare_training_data(
        telemetry_store=store, config=cfg, stage="cleanup",
        backend_service=SimpleNamespace(get_all_racing_sessions_streaming=AsyncMock(return_value={"success": True})),
        top_lap_reference_feature_names=list(partial_lap[0]),
    )
    assert result["success"]
    assert store.caches[cfg.processed_session_data_cache_key] == {"partial": partial_lap}
    assert set(store.caches) == {cfg.session_data_cache_key, cfg.processed_session_data_cache_key}


@pytest.mark.asyncio
async def test_selection_reads_only_cleaned_sessions_and_invalidates_enrichment(monkeypatch):
    cfg = TrainingPipelineConfig()
    cleaned = {
        "slow": lap(2000), "fast": lap(1000), "invalid": lap(500, valid=0),
        "other-game": lap(1500, game="iracing_recorded"),
        "other-car": lap(1700, car="car-b"), "other-grip": lap(1800, grip=5),
    }
    store = TelemetryStore({
        cfg.session_data_cache_key: {"uncleaned-fastest": lap(100)},
        cfg.processed_session_data_cache_key: cleaned,
        cfg.top_laps_cache_key: {"stale-bucket": [lap()]},
        cfg.enriched_sessions_cache_key: {"old": lap()},
        cfg.segments_cache_key: {"old": [lap()]},
    })
    monkeypatch.setattr(FeatureProcessor, "general_cleaning_for_analysis", lambda *_: pytest.fail("selection reran cleanup"))
    backend = SimpleNamespace(get_all_racing_sessions_streaming=AsyncMock())
    result = await pipeline.prepare_training_data(
        telemetry_store=store, config=cfg, stage="top_laps",
        backend_service=backend, top_lap_reference_feature_names=[],
    )
    assert result["success"]
    backend.get_all_racing_sessions_streaming.assert_not_awaited()
    assert store.caches[cfg.top_laps_cache_key] == {
        "acc|spa|car-a|grip2": [cleaned["fast"]],
        "iracing_recorded|spa|car-a|grip2": [cleaned["other-game"]],
        "acc|spa|car-b|grip2": [cleaned["other-car"]],
        "acc|spa|car-a|grip5": [cleaned["other-grip"]],
    }
    assert store.caches[cfg.processed_session_data_cache_key] == cleaned
    assert not store.has_cached_data(cfg.enriched_sessions_cache_key)
    assert not store.has_cached_data(cfg.segments_cache_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,has_cleaned,message", [
    ("top_laps", False, "Run data cleanup"),
    ("enrich", False, "Run data cleanup"),
    ("enrich", True, "Select top laps"),
])
async def test_preparation_requires_prior_stages(monkeypatch, stage, has_cleaned, message):
    cfg = TrainingPipelineConfig()
    store = TelemetryStore({cfg.processed_session_data_cache_key: {"session": lap()}} if has_cleaned else {})
    backend = SimpleNamespace(get_all_racing_sessions_streaming=AsyncMock())
    enrich = AsyncMock()
    monkeypatch.setattr(pipeline, "enriched_contextual_data", enrich)
    result = await pipeline.prepare_training_data(
        telemetry_store=store, config=cfg, backend_service=backend,
        top_lap_reference_feature_names=[], stage=stage,
    )
    assert not result["success"]
    assert message in result["error"]
    backend.get_all_racing_sessions_streaming.assert_not_awaited()
    enrich.assert_not_awaited()


@pytest.mark.asyncio
async def test_selection_rejects_sessions_without_valid_reference_laps():
    cfg = TrainingPipelineConfig()
    store = TelemetryStore({cfg.processed_session_data_cache_key: {"invalid": lap(valid=0)}})
    with pytest.raises(ValueError, match="No top laps"):
        await select_top_laps(cfg.processed_session_data_cache_key, telemetry_store=store, cache_config=cfg)
    assert not store.has_cached_data(cfg.top_laps_cache_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [1, 2])
async def test_selection_preserves_first_lap_on_ties_across_workers(workers):
    cfg = TrainingPipelineConfig()
    first = [{**row, "marker": "first"} for row in lap()]
    tied = [{**row, "marker": "tied"} for row in lap()]
    other_car = lap(car="car-b")
    store = TelemetryStore({cfg.processed_session_data_cache_key: {
        "empty": [], "first": first, "invalid": lap(500, valid=0),
        "tied": tied, "other-car": other_car,
    }})
    await select_top_laps(
        cfg.processed_session_data_cache_key, telemetry_store=store,
        cache_config=cfg, max_workers=workers,
    )
    assert store.caches[cfg.top_laps_cache_key] == {
        "acc|spa|car-a|grip2": [first], "acc|spa|car-b|grip2": [other_car],
    }


def _enrichment_lap():
    return [
        {
            **row,
            **{
                f"Physics_{feature}_{wheel}": 0.05
                for feature in ("slip_angle", "slip_ratio")
                for wheel in ("front_left", "front_right", "rear_left", "rear_right")
            },
        }
        for row in lap()
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [1, 2])
async def test_enrichment_workers_preserve_features_ids_and_protected_sessions(monkeypatch, workers):
    cfg = TrainingPipelineConfig()
    reference = _enrichment_lap()
    sessions = {
        "session-a": reference,
        "session-b": [{**row, "Physics_speed_kmh": 80.0} for row in reference],
        "session-c": [{**row, "Physics_speed_kmh": 90.0} for row in reference],
        "empty": [],
        "no-reference": [{**row, "Static_car_model": "missing-car"} for row in reference],
        "protected:chunk_0": [{"invalid": "must not reach a worker"}],
    }
    store = TelemetryStore({
        cfg.top_laps_cache_key: {"reference": [reference]},
        cfg.processed_session_data_cache_key: sessions,
    })
    publish = AsyncMock()
    monkeypatch.setattr(enrich, "save_ai_model", publish)
    key, model = await enrich.enriched_contextual_data(
        cfg.processed_session_data_cache_key, telemetry_store=store,
        cache_config=cfg, max_workers=workers, protected_session_ids={"protected"},
    )
    assert key == cfg.enriched_sessions_cache_key
    assert list(store.caches[key]) == ["session-a", "session-b", "session-c"]
    for chunk_id, rows in store.caches[key].items():
        expected = await enrich.enrich_sessions_with_context(
            sessions[chunk_id], model, enrich.TireGripAnalysisService(),
        )
        assert rows == expected
        assert rows[0]["expert_optimal_speed"] == 100.0
        assert rows[0]["driver_push_to_limit"] > 0
    assert store.caches[cfg.processed_session_data_cache_key] == sessions
    assert publish.await_count == 2


@pytest.mark.asyncio
async def test_enrichment_propagates_worker_errors(monkeypatch):
    cfg = TrainingPipelineConfig()
    store = TelemetryStore({
        cfg.top_laps_cache_key: {"reference": [_enrichment_lap()]},
        cfg.processed_session_data_cache_key: {"missing-slip-features": lap()},
    })
    monkeypatch.setattr(enrich, "save_ai_model", AsyncMock())
    with pytest.raises(ValueError, match="missing required columns"):
        await enrich.enriched_contextual_data(
            cfg.processed_session_data_cache_key, telemetry_store=store,
            cache_config=cfg, max_workers=2,
        )
    assert not store.has_cached_data(cfg.enriched_sessions_cache_key)


@pytest.mark.asyncio
@pytest.mark.parametrize("workers", [0, -1])
async def test_preparation_rejects_invalid_workers_before_clearing_caches(workers):
    cfg = TrainingPipelineConfig()
    caches = {cfg.processed_session_data_cache_key: {"session": lap()}}
    store = TelemetryStore(caches)
    backend = SimpleNamespace(get_all_racing_sessions_streaming=AsyncMock())
    with pytest.raises(ValueError, match="Workers must be at least 1"):
        await pipeline.prepare_training_data(
            telemetry_store=store, config=cfg, backend_service=backend,
            top_lap_reference_feature_names=[], max_workers=workers,
        )
    assert store.caches == caches
    backend.get_all_racing_sessions_streaming.assert_not_awaited()


@pytest.mark.asyncio
async def test_selection_reports_failed_cache_write():
    cfg = TrainingPipelineConfig()
    store = TelemetryStore({cfg.processed_session_data_cache_key: {"session": lap()}})
    store.cache_chunks_streaming = AsyncMock(return_value=False)
    with pytest.raises(RuntimeError, match="Failed to cache"):
        await select_top_laps(cfg.processed_session_data_cache_key, telemetry_store=store, cache_config=cfg)


@pytest.mark.parametrize("positions", [
    pytest.param([0.02, 0.01, 0.99, 0.98], id="backwards-start-finish"),
    pytest.param([0.0, 1.0], id="endpoints-only"),
    pytest.param([0.0, 0.05, 0.1, 0.85, 0.9, 0.95, 1.0], id="missing-middle"),
    pytest.param([i / 20 for i in range(20, -1, -1)], id="reverse-lap"),
    pytest.param([i / 20 for i in range(21)] + [0.8], id="reversed-before-finish"),
    pytest.param([i / 20 if i != 10 else float("nan") for i in range(21)], id="nonfinite-position"),
    pytest.param([i / 20 if i != 10 else float("inf") for i in range(21)], id="infinite-position"),
    pytest.param([i / 20 for i in range(-1, 22)], id="out-of-range"),
])
def test_lap_candidates_require_continuous_forward_coverage(positions):
    assert list(_iter_lap_candidates(lap(positions=positions), "fragment")) == []


def test_lap_candidates_require_position_data():
    records = lap()
    for row in records:
        del row["Graphics_normalized_car_position"]
    assert list(_iter_lap_candidates(records, "missing-position")) == []


@pytest.mark.parametrize("positions", [
    pytest.param([i / 20 for i in range(21)], id="complete"),
    pytest.param([i / 10 for i in range(11)], id="rounding-at-coverage-limit"),
    pytest.param([0.01 + i * 0.049 for i in range(21)], id="sampled-near-boundaries"),
    pytest.param([i / 20 for i in range(21) for _ in range(2)], id="repeated-positions"),
])
def test_lap_candidates_keep_complete_laps_in_record_order(positions):
    records = lap(positions=positions)
    candidates = list(_iter_lap_candidates(records, "complete"))
    assert len(candidates) == 1
    assert candidates[0]["records"] == records
    assert candidates[0]["lap_time_ms"] == 10_000.0


def test_lap_candidates_keep_consecutive_laps_with_a_stale_lap_counter():
    first_lap = lap(10_000)
    second_lap = lap(9000)
    candidates = list(_iter_lap_candidates(first_lap + second_lap, "session"))

    assert [candidate["records"] for candidate in candidates] == [first_lap, second_lap]


@pytest.mark.asyncio
@pytest.mark.parametrize("fragment_first", [True, False])
async def test_selection_chooses_fastest_complete_lap_over_backwards_fragment(fragment_first):
    cfg = TrainingPipelineConfig()
    fragment = lap(1000, positions=[0.02, 0.01, 0.99, 0.98])
    complete = lap(10_000)
    chunks = [("fragment", fragment), ("slow", lap(20_000)), ("fast", complete)]
    if not fragment_first:
        chunks.reverse()
    store = TelemetryStore({cfg.processed_session_data_cache_key: dict(chunks)})

    await select_top_laps(cfg.processed_session_data_cache_key, telemetry_store=store, cache_config=cfg)

    assert store.caches[cfg.top_laps_cache_key] == {"acc|spa|car-a|grip2": [complete]}


@pytest.mark.asyncio
async def test_selection_does_not_cache_a_backwards_fragment_without_complete_laps():
    cfg = TrainingPipelineConfig()
    store = TelemetryStore({cfg.processed_session_data_cache_key: {
        "fragment": lap(1000, positions=[0.02, 0.01, 0.99, 0.98]),
    }})

    with pytest.raises(ValueError, match="No top laps"):
        await select_top_laps(cfg.processed_session_data_cache_key, telemetry_store=store, cache_config=cfg)

    assert not store.has_cached_data(cfg.top_laps_cache_key)
