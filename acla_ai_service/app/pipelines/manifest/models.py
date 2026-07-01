"""Pipeline manifest dataclasses.

An annotation runs in one of three modes:

- ``"source"`` — copies ``source_ref`` to a private input dataset and
  writes annotations to its own ``output_key``.
- ``"secondary_worker"`` — reads the *target* sibling's
  output and writes annotations to its own ``output_key``.
- ``"coworker"`` — reads the *target* sibling's input and
  writes to the target's output (e.g. an AI agent assisting a human
  annotator in parallel on the same input → output flow).

For the two worker modes, ``source_ref`` is the target annotation id;
the actual keys are looked up via the target's effective input/output.

Training nodes read annotation output via :meth:`Pipeline.resolve_source_key`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


MODE_SOURCE = "source"
MODE_SECONDARY_WORKER = "secondary_worker"
MODE_COWORKER = "coworker"
_VALID_MODES = {MODE_SOURCE, MODE_SECONDARY_WORKER, MODE_COWORKER}
_SOURCE_MODE_ALIASES = {"copy", "fork"}
_LEGACY_OUTPUT_SUFFIX = ".output"


def _now_iso() -> str:
    return datetime.now().isoformat()


@dataclass
class AnnotationNode:
    id: str                                          # unique within pipeline (stable ref key)
    kind: str                                        # matches a NodeKindSpec.kind
    output_key: str = ""                             # this annotation's own output dataset. Empty = not configured yet; the annotation page's popup sets it on first open.
    name: Optional[str] = None                       # user-editable display label; falls back to kind.display
    source_ref: Optional[str] = None                 # source: any cache_key or annotation id. Worker modes: target annotation id.
    mode: str = MODE_SOURCE                          # "source" | "secondary_worker" | "coworker"
    output_dir: Optional[str] = None                 # filesystem dir holding output_key's lance dataset; None = default lance store dir
    protection_reference: Optional[Dict[str, str]] = None  # selected schema-backed output->input row reference for source-copy refresh.

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnnotationNode":
        # Tolerate the legacy schema (had `input_ref`).
        source_ref = d.get("source_ref") or d.get("input_ref")
        # Canonicalize legacy kind strings (parent/children/batch → new names).
        from app.pipelines.manifest.node_kinds import canonicalize
        # Tolerate old manifest values, but never preserve copy/fork as
        # active modes.
        raw_mode = d.get("mode")
        if raw_mode in _SOURCE_MODE_ALIASES:
            mode = MODE_SOURCE
        elif raw_mode in _VALID_MODES:
            mode = raw_mode
        elif d.get("coworker_mode"):
            mode = MODE_COWORKER
        else:
            raise ValueError(f"Invalid annotation mode for {d['id']!r}: {raw_mode!r}")
        return cls(
            id=d["id"],
            kind=canonicalize(d["kind"]),
            output_key=d.get("output_key", "") or "",
            name=d.get("name"),
            source_ref=source_ref,
            mode=mode,
            output_dir=d.get("output_dir"),
            protection_reference=d.get("protection_reference"),
        )


@dataclass
class TrainingNode:
    id: str                                          # unique within pipeline (slug of name)
    kind: str                                        # matches a NodeKindSpec.kind
    input_ref: str                                   # annotation id whose output trains the model
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

        A node id resolves to that annotation's effective output.
        Coworkers share the upstream target's effective output; secondary
        workers expose their own output. Legacy ``"<node_id>.output"``
        refs are still accepted on load/runtime but are not written back.

        Anything else is assumed to already be a cache_key in the store.
        """
        if not ref:
            return None
        target_id = self._target_id_from_ref(ref)
        if target_id:
            try:
                target = self.annotation(target_id)
            except KeyError:
                return None
            return self.effective_output_key(target)
        return ref

    def _resolve_target(self, node: "AnnotationNode") -> Optional["AnnotationNode"]:
        """Sibling node referenced by a worker node's source_ref."""
        target_id = self._target_id_from_ref(node.source_ref)
        if not target_id:
            return None
        try:
            return self.annotation(target_id)
        except KeyError:
            return None

    def _target_id_from_ref(self, ref: Optional[str]) -> Optional[str]:
        if not ref:
            return None
        node_ids = {node.id for node in self.annotations}
        if ref in node_ids:
            return ref
        if ref.endswith(_LEGACY_OUTPUT_SUFFIX):
            node_id = ref[:-len(_LEGACY_OUTPUT_SUFFIX)]
            if node_id in node_ids:
                return node_id
        return None

    def _serialize_ref(self, ref: Optional[str]) -> Optional[str]:
        target_id = self._target_id_from_ref(ref)
        return target_id if target_id else ref

    def effective_input_key(
        self, node: "AnnotationNode", _seen: Optional[set] = None,
    ) -> Optional[str]:
        """Cache_key this annotation actually reads from.

        - Source → copied input dataset once ``output_key`` is configured,
          otherwise the resolved source so first-time setup can open.
        - Secondary worker → target's effective output.
        - Coworker → target's effective *input* (read what the target
          reads; write where the target writes).
        """
        if node.mode == MODE_SOURCE:
            if node.output_key:
                from app.pipelines.manifest.source_copy import source_copy_key
                return source_copy_key(node.output_key)
            return self.resolve_source_key(node.source_ref)
        if node.mode == MODE_SECONDARY_WORKER:
            target = self._resolve_target(node)
            if target is None:
                return None
            return self.effective_output_key(target)
        # MODE_COWORKER: chase target's input recursively (with cycle guard).
        seen = _seen if _seen is not None else set()
        if node.id in seen:
            return None
        seen = seen | {node.id}
        target = self._resolve_target(node)
        if target is None:
            return None
        return self.effective_input_key(target, seen)

    def effective_output_key(
        self, node: "AnnotationNode", _seen: Optional[set] = None,
    ) -> Optional[str]:
        """Cache_key this annotation actually writes to.

        - Source → ``node.output_key``.
        - Secondary worker → ``node.output_key``.
        - Coworker → the upstream target's output.
        """
        if node.mode in {MODE_SOURCE, MODE_SECONDARY_WORKER}:
            return node.output_key or None
        seen = _seen if _seen is not None else set()
        if node.id in seen:
            return None
        seen = seen | {node.id}
        target = self._resolve_target(node)
        if target is None:
            return None
        return self.effective_output_key(target, seen)

    # ── Serialization ────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "created_at": self.created_at,
            "annotations": [
                {
                    **n.to_dict(),
                    "source_ref": self._serialize_ref(n.source_ref),
                }
                for n in self.annotations
            ],
            "trainings": [
                {
                    **n.to_dict(),
                    "input_ref": self._serialize_ref(n.input_ref),
                }
                for n in self.trainings
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Pipeline":
        # Note: legacy manifests had a top-level ``inputs`` list. We drop
        # it on load — each annotation now owns its own source_ref.
        pipeline = cls(
            id=d["id"],
            version=int(d.get("version", 1)),
            created_at=d.get("created_at", _now_iso()),
            annotations=[AnnotationNode.from_dict(x) for x in d.get("annotations", [])],
            trainings=[TrainingNode.from_dict(x) for x in d.get("trainings", [])],
        )
        for node in pipeline.annotations:
            node.source_ref = pipeline._serialize_ref(node.source_ref)
        for node in pipeline.trainings:
            node.input_ref = pipeline._serialize_ref(node.input_ref)
        return pipeline


__all__ = [
    "MODE_SOURCE",
    "MODE_SECONDARY_WORKER",
    "MODE_COWORKER",
    "AnnotationNode",
    "TrainingNode",
    "Pipeline",
]
