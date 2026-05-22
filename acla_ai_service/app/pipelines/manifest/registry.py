"""Filesystem-backed Pipeline registry.

Manifests are JSON files in ``app/storage/pipelines/``. One file per
pipeline, named ``<pipeline_id>.json``.

Pipelines start empty — no hardcoded main-dataset fork. The user adds
annotation components in the UI and picks the source for each one
(which is then forked into the annotation's own input dataset).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.pipelines.manifest.models import Pipeline


_STORAGE_ROOT = Path(__file__).resolve().parents[2] / "storage" / "pipelines"
_SLUG_RE = re.compile(r"[^a-z0-9_]+")


def _root() -> Path:
    _STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    return _STORAGE_ROOT


def _path(pipeline_id: str) -> Path:
    return _root() / f"{pipeline_id}.json"


def slugify(text: str) -> str:
    s = _SLUG_RE.sub("_", text.lower().strip()).strip("_")
    return s or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def list_pipelines() -> List[str]:
    return sorted(p.stem for p in _root().glob("*.json"))


def load(pipeline_id: str) -> Optional[Pipeline]:
    path = _path(pipeline_id)
    if not path.exists():
        return None
    with path.open("r") as fh:
        return Pipeline.from_dict(json.load(fh))


def save(pipeline: Pipeline) -> None:
    path = _path(pipeline.id)
    with path.open("w") as fh:
        json.dump(pipeline.to_dict(), fh, indent=2, sort_keys=False)


def delete(pipeline_id: str) -> bool:
    path = _path(pipeline_id)
    if path.exists():
        path.unlink()
        return True
    return False


def create_pipeline(name: str, annotation_prefix: str) -> Pipeline:
    """Create an empty pipeline.

    No nodes are scaffolded — the user adds annotation and training
    components individually in the graph view and picks each one's
    kind from the dropdown.

    ``annotation_prefix`` is retained for compatibility with callers
    but currently unused; per-node ``output_key`` is derived in the UI
    when nodes are added.
    """
    pid = slugify(name)
    pipeline = Pipeline(id=pid, version=1, annotations=[], trainings=[])
    save(pipeline)
    return pipeline


def derive_output_key(annotation_prefix: str, pipeline_id: str, node_id: str) -> str:
    """Per-node output cache_key — used by the UI when adding a node."""
    return f"{annotation_prefix}_{pipeline_id}__{node_id}"


__all__ = [
    "list_pipelines", "load", "save", "delete",
    "create_pipeline", "slugify", "derive_output_key",
]
