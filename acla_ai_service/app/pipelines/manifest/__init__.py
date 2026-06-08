"""Pipeline manifests — JSON definitions of an annotation+training workflow.

Public surface used by the UI:

- :class:`Pipeline`, :class:`AnnotationNode`, :class:`TrainingNode` —
  manifest dataclasses (``models``).
- :mod:`registry` — list / load / save / create / delete pipelines in
  ``app/storage/pipelines/<id>.json``.
- :mod:`node_kinds` — registry of annotation/training kinds; extend it
  to add new node types.
"""

from app.pipelines.manifest.models import (
    MODE_SOURCE,
    AnnotationNode,
    Pipeline,
    TrainingNode,
)
from app.pipelines.manifest.protection import (
    collect_protected_session_ids,
    is_protected_chunk_id,
    session_id_from_chunk_id,
)
from app.pipelines.manifest.source_copy import (
    ProtectionReference,
    source_copy_key,
    sync_source_copy,
)
from app.pipelines.manifest import node_kinds, registry

__all__ = [
    "MODE_SOURCE",
    "AnnotationNode",
    "Pipeline",
    "TrainingNode",
    "collect_protected_session_ids",
    "is_protected_chunk_id",
    "session_id_from_chunk_id",
    "ProtectionReference",
    "source_copy_key",
    "sync_source_copy",
    "node_kinds",
    "registry",
]
