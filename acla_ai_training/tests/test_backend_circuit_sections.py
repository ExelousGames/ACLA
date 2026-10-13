from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd
import plotly.graph_objects as go
import pytest

from app.integrations.backend import client as backend_client
from training.shared import circuit_sections
from training.shared.annotation_agent_tools import (
    locate_circuit_section,
    split_lap_by_circuit_sections,
)
from ui.segment_tabs.components import track_sections


@pytest.fixture
def backend_map(monkeypatch):
    summary = {
        "id": "map-id", "game": "acc", "circuit_name": "Backend Circuit",
        "source_track_key": "brands_hatch",
    }
    detail = {
        **summary,
        "centerline_segments": [
            {"id": "section-a", "name": "Backend Hairpin", "tags": ["corner"],
             "start_position": 0.1, "end_position": 0.3},
            {"id": "section-wrap", "name": "Finish Straight", "tags": ["long straight"],
             "start_position": 0.9, "end_position": 0.05},
        ],
    }
    responses = {"circuit-map/list?game=acc": {"list": [summary]}, "circuit-map/map-id": detail}

    async def request(endpoint, **_kwargs):
        return responses[endpoint]

    mock = AsyncMock(side_effect=request)
    monkeypatch.setattr(backend_client, "BackendService", lambda: SimpleNamespace(
        call_backend_function=mock,
    ))
    # A fixed cache clock keeps these tests independent of minute boundaries.
    monkeypatch.setattr(circuit_sections, "time", SimpleNamespace(monotonic=lambda: 120))
    circuit_sections._cached_sections.cache_clear()
    yield SimpleNamespace(summary=summary, detail=detail, responses=responses, request=mock)
    circuit_sections._cached_sections.cache_clear()


def telemetry(positions, track="Brands Hatch"):
    return pd.DataFrame({
        "Static_track": [track] * len(positions),
        "Graphics_normalized_car_position": positions,
        "x": range(len(positions)), "y": range(len(positions)),
    })


@pytest.mark.parametrize("track", ["Brands Hatch", "brands_hatch", "Backend Circuit"])
def test_sections_use_backend_names_and_ranges(backend_map, track):
    sections = circuit_sections.get_circuit_sections(telemetry([0.15], track))
    assert sections == [
        {"id": "section-a", "name": "Backend Hairpin", "normalized_position_range": (0.1, 0.3)},
        {"id": "section-wrap", "name": "Finish Straight", "normalized_position_range": (0.9, 0.05)},
    ]


def test_track_key_takes_priority_over_display_name(backend_map):
    backend_map.responses["circuit-map/list?game=acc"]["list"].insert(0, {
        "id": "wrong-map", "circuit_name": "Brands Hatch", "source_track_key": "another_layout",
    })
    assert circuit_sections.get_circuit_sections(telemetry([0.15]))[0]["id"] == "section-a"


def test_sections_refresh_after_backend_edit(backend_map, monkeypatch):
    df = telemetry([0.15])
    assert circuit_sections.get_circuit_sections(df)[0]["name"] == "Backend Hairpin"
    circuit_sections.get_circuit_sections(df)
    assert backend_map.request.await_count == 2
    backend_map.detail["centerline_segments"][0].update(name="Renamed corner", end_position=0.4)
    monkeypatch.setattr(circuit_sections, "time", SimpleNamespace(monotonic=lambda: 180))
    section = circuit_sections.get_circuit_sections(df)[0]
    assert section["name"] == "Renamed corner"
    assert section["normalized_position_range"] == (0.1, 0.4)


@pytest.mark.parametrize("case", ["missing_map", "cleared_sections", "backend_error"])
def test_no_static_fallback(backend_map, case):
    if case == "missing_map":
        backend_map.responses["circuit-map/list?game=acc"] = {"list": []}
    elif case == "cleared_sections":
        backend_map.detail["centerline_segments"] = []
    else:
        backend_map.responses["circuit-map/list?game=acc"] = {"error": "Backend unavailable"}
    df = telemetry([0.12, 0.15])
    assert circuit_sections.get_circuit_sections(df) == []
    assert not track_sections.track_sections_available(df)
    assert split_lap_by_circuit_sections(
        df, 0, 2, include_interaction_windows=False,
    ).content["segments"] == []


def test_unknown_track_does_not_use_other_maps(backend_map):
    assert circuit_sections.get_circuit_sections(telemetry([0.15], "Unknown Track")) == []
    assert backend_map.request.await_count == 1


def test_missing_track_does_not_fetch_maps(backend_map):
    assert circuit_sections.get_circuit_sections(pd.DataFrame()) == []
    assert backend_map.request.await_count == 0


def test_game_selects_backend_map_catalog(backend_map):
    backend_map.responses["circuit-map/list?game=iracing"] = {"list": [backend_map.summary]}
    df = telemetry([0.15])
    df["game"] = "iracing"
    assert circuit_sections.get_circuit_sections(df)[0]["name"] == "Backend Hairpin"
    assert backend_map.request.call_args_list[0].args == ("circuit-map/list?game=iracing",)


def test_unnamed_section_uses_backend_tags(backend_map):
    del backend_map.detail["centerline_segments"][0]["name"]
    assert circuit_sections.get_circuit_sections(telemetry([0.15]))[0]["name"] == "corner"


@pytest.mark.asyncio
async def test_sections_can_load_inside_async_annotation_provider(backend_map):
    assert circuit_sections.get_circuit_sections(telemetry([0.15]))[0]["id"] == "section-a"


def test_ui_uses_backend_ranges_with_wraparound_and_gaps(backend_map):
    df = telemetry([0.91, 0.98, 0.02, 0.04, 0.07, 0.12, 0.25, 0.4])
    runs = track_sections._section_runs(df)
    assert [(run["section_id"], run["start_index"], run["end_index"]) for run in runs] == [
        ("section-wrap", 0, 3), ("section-a", 5, 6),
    ]
    fig = go.Figure()
    track_sections.add_track_section_trajectory(fig, df, "x", "y")
    assert fig.data[1].name == "Section: Backend Hairpin"
    assert "Section range: 0.1000-0.3000" in fig.data[1].customdata[0]
    bands = go.Figure()
    track_sections.add_track_section_bands(bands, df, "x")
    assert [(shape.x0, shape.x1) for shape in bands.layout.shapes] == [(0, 3), (5, 6)]


def test_lap_split_and_location_use_same_backend_sections(backend_map):
    df = telemetry([0.12, 0.25, 0.5, 0.92, 0.98])
    result = split_lap_by_circuit_sections(df, 0, 5, include_interaction_windows=False).content
    assert [(segment["circuit_section_id"], segment["start_index"], segment["end_index"])
            for segment in result["segments"]] == [("section-a", 0, 3), ("section-wrap", 3, 5)]
    assert result["segments"][0]["circuit_section_name"] == "Backend Hairpin"
    match = locate_circuit_section(df, "brands_hatch", 0, 2)["best_match"]
    assert match["label_id"] == "section-a"
    assert match["name"] == "Backend Hairpin"
    assert match["section_range"] == [0.1, 0.3]


def test_annotation_retains_backend_section_context(backend_map):
    from training.local_annotation_agent.workflow.deterministic import _resolve_circuit_sections

    assert _resolve_circuit_sections(
        telemetry([0.12, 0.25]), "brands_hatch", "section-a", 0, 2, None,
    ) == ("section-a", ["section-a"])


def test_section_overlap_facts_use_backend_names():
    from training.local_annotation_agent.workflow.deterministic_facts import (
        EvaluationContext, FACT_REGISTRY,
    )

    context = EvaluationContext(
        pd.DataFrame(), overlap_section_ids=("backend-pit",),
        section_names={"backend-pit": "Pit"},
    )
    assert FACT_REGISTRY.get("find_section_overlap_names").calculate(context, []) == ["Pit"]
