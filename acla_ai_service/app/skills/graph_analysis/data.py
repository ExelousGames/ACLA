"""Data layer for graph_analysis.

YAML's ``graphs`` block is the source. Each entry becomes a document
with ``id`` injected from the key. ``cross_graph_guidelines`` is a
scalar dict — accessed via path, not as a collection.
"""

from __future__ import annotations

from typing import Any, Dict


def graphs(raw_body: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    yaml_graphs: Dict[str, Any] = raw_body.get("graphs") or {}
    return {
        str(gid): {"id": str(gid), **(entry or {})}
        for gid, entry in yaml_graphs.items()
    }


COLLECTIONS = {
    "graphs": graphs,
}
