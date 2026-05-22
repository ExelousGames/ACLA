"""Pipeline manifest dataclasses.

An annotation runs in one of three modes:

- ``"fork"`` — ``source_ref`` is copied into the per-node ``input_key``
  and lineage fields (``copied_at``, ``source_updated_at_on_copy``)
  drive the "behind source" badge. The node writes to its own
  ``output_key``.
- ``"secondary_worker"`` — no fork. Reads the *target* sibling's
  output and writes back to the same dataset (e.g. detailed
  annotation adding child segments to a parent's output).
- ``"coworker"`` — no fork. Reads the *target* sibling's input and
  writes to the target's output (e.g. an AI agent assisting a human
  annotator in parallel on the same input → output flow).

For the two no-fork modes, ``source_ref`` is ``"<target_id>.output"``
— the suffix is just the node-reference convention; the actual keys
are looked up via the target's effective input/output.

Training nodes don't fork; their ``input_ref`` resolves to an
annotation's output via :meth:`Pipeline.resolve_source_key`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


MODE_FORK = "fork"
MODE_SECONDARY_WORKER = "secondary_worker"
MODE_COWORKER = "coworker"
_VALID_MODES = {MODE_FORK, MODE_SECONDARY_WORKER, MODE_COWORKER}


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class AnnotationNode:
    id: str                                          # unique within pipeline (stable ref key)
    kind: str                                        # matches a NodeKindSpec.kind
    output_key: str                                  # this annotation's output dataset (fork mode)
    name: Optional[str] = None                       # user-editable display label; falls back to kind.display
    source_ref: Optional[str] = None                 # fork: any cache_key or "<id>.output". Non-fork: "<target_id>.output".
    input_key: Optional[str] = None                  # forked copy this annotation reads from (fork mode)
    copied_at: Optional[str] = None
    source_updated_at_on_copy: Optional[str] = None
    mode: str = MODE_FORK                            # "fork" | "secondary_worker" | "coworker"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnnotationNode":
        # Tolerate the legacy schema (had `input_ref` but no fork lineage).
        source_ref = d.get("source_ref") or d.get("input_ref")
        # Canonicalize legacy kind strings (parent/children/batch → new names).
        from app.pipelines.manifest.node_kinds import canonicalize
        # Migrate legacy boolean `coworker_mode` → tri-state `mode`. The old
        # "coworker" semantics (read + write upstream's output) are exactly
        # today's "secondary_worker".
        if "mode" in d and d["mode"] in _VALID_MODES:
            mode = d["mode"]
        else:
            mode = MODE_SECONDARY_WORKER if d.get("coworker_mode") else MODE_FORK
        return cls(
            id=d["id"],
            kind=canonicalize(d["kind"]),
            output_key=d["output_key"],
            name=d.get("name"),
            source_ref=source_ref,
            input_key=d.get("input_key"),
            copied_at=d.get("copied_at"),
            source_updated_at_on_copy=d.get("source_updated_at_on_copy"),
            mode=mode,
        )


@dataclass
class TrainingNode:
    id: str                                          # unique within pipeline (slug of name)
    kind: str                                        # matches a NodeKindSpec.kind
    input_ref: str                                   # "<annotation_id>.output"
    name: Optional[str] = None                       # user-given display label; falls back to kind.display
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingNode":
        return cls(
            id=d["id"],
            kind=d["kind"],
            input_ref=d["input_ref"],
            name=d.get("name"),
            extra=dict(d.get("extra", {})),
        )


@dataclass
class Pipeline:
    id: str
    version: int = 1
    created_at: str = field(default_factory=_now_iso)
    annotations: List[AnnotationNode] = field(default_factory=list)
    trainings: List[TrainingNode] = field(default_factory=list)

    # ── Lookups ──────────────────────────────────────────────────────────
    def annotation(self, node_id: str) -> AnnotationNode:
        for n in self.annotations:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def training(self, node_id: str) -> TrainingNode:
        for n in self.trainings:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def resolve_source_key(self, ref: Optional[str]) -> Optional[str]:
        """Turn a ``source_ref`` into the actual output cache_key.

        ``"<node_id>.output"`` resolves to that annotation's effective
        output. For a non-fork target (secondary_worker / coworker),
        that's the upstream's effective output (followed recursively)
        — so chaining always lands on the dataset everybody is
        actually writing to.

        Anything else is assumed to already be a cache_key in the store.
        """
        if not ref:
            return None
        if "." in ref:
            node_id, attr = ref.split(".", 1)
            if attr == "output":
                seen: set[str] = set()
                cur = node_id
                while cur not in seen:
                    seen.add(cur)
                    try:
                        target = self.annotation(cur)
                    except KeyError:
                        return None
                    if target.mode == MODE_FORK:
                        return target.output_key
                    # Non-fork: write target is the upstream's output.
                    if not target.source_ref or "." not in target.source_ref:
                        return target.source_ref
                    nxt, nxt_attr = target.source_ref.split(".", 1)
                    if nxt_attr != "output":
                        return target.source_ref
                    cur = nxt
                return None  # cycle
        return ref

    def _resolve_target(self, node: "AnnotationNode") -> Optional["AnnotationNode"]:
        """Sibling node referenced by a non-fork node's source_ref."""
        if not node.source_ref or "." not in node.source_ref:
            return None
        target_id, attr = node.source_ref.split(".", 1)
        if attr != "output":
            return None
        try:
            return self.annotation(target_id)
        except KeyError:
            return None

    def effective_input_key(
        self, node: "AnnotationNode", _seen: Optional[set] = None,
    ) -> Optional[str]:
        """Cache_key this annotation actually reads from.

        - Fork → ``node.input_key`` (the forked copy).
        - Secondary worker → target's effective output (read + write
          the same dataset).
        - Coworker → target's effective *input* (read what the target
          reads; write where the target writes).
        """
        if node.mode == MODE_FORK:
            return node.input_key
        if node.mode == MODE_SECONDARY_WORKER:
            return self.resolve_source_key(node.source_ref)
        # MODE_COWORKER: chase target's input recursively (with cycle guard).
        seen = _seen if _seen is not None else set()
        if node.id in seen:
            return None
        seen = seen | {node.id}
        target = self._resolve_target(node)
        if target is None:
            return None
        return self.effective_input_key(target, seen)

    def effective_output_key(self, node: "AnnotationNode") -> Optional[str]:
        """Cache_key this annotation actually writes to.

        - Fork → ``node.output_key``.
        - Secondary worker / coworker → the upstream target's output
          (resolved recursively via ``resolve_source_key``).
        """
        if node.mode == MODE_FORK:
            return node.output_key
        return self.resolve_source_key(node.source_ref)

    # ── Serialization ────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "created_at": self.created_at,
            "annotations": [n.to_dict() for n in self.annotations],
            "trainings": [n.to_dict() for n in self.trainings],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pipeline":
        # Note: legacy manifests had a top-level ``inputs`` list. We drop
        # it on load — each annotation now owns its own forked input.
        return cls(
            id=d["id"],
            version=int(d.get("version", 1)),
            created_at=d.get("created_at", _now_iso()),
            annotations=[AnnotationNode.from_dict(x) for x in d.get("annotations", [])],
            trainings=[TrainingNode.from_dict(x) for x in d.get("trainings", [])],
        )


__all__ = ["AnnotationNode", "TrainingNode", "Pipeline"]
