"""Pipeline manifests — JSON definitions of an annotation+training workflow.

Public surface used by the UI:

- :class:`Pipeline`, :class:`AnnotationNode`, :class:`TrainingNode` —
  manifest dataclasses (``models``).
- :mod:`registry` — list / load / save / create / delete pipelines in
  ``app/storage/pipelines/<id>.json``.
- :func:`fork_dataset` — Lance-level copy used when an annotation's
  source is selected/changed.
- :func:`compare_against_source` — backs the git-style "N records behind"
  badge per forked input.
- :mod:`node_kinds` — registry of annotation/training kinds; extend it
  to add new node types.
"""

from app.pipelines.manifest.models import (
    AnnotationNode,
    Pipeline,
    TrainingNode,
)
from app.pipelines.manifest.forking import (
    fork_dataset,
    compare_against_source,
    SourceComparison,
)
from app.pipelines.manifest import node_kinds, registry

__all__ = [
    "AnnotationNode",
    "Pipeline",
    "TrainingNode",
    "fork_dataset",
    "compare_against_source",
    "SourceComparison",
    "node_kinds",
    "registry",
]
