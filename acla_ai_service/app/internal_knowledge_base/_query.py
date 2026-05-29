"""Document-store query engine for the skill registry.

Skills are loaded as documents (nested dict trees). Callers pull data
out via path-based ``get`` or predicate-based ``find`` — same shape as
a Mongo-style key-value store. Skill-specific method names disappear
from the caller surface; everything is a query.

Path syntax
-----------
Dotted paths into the document tree::

    sub_label_annotation.labels.MSP1
    sub_label_annotation.labels.MSP1.description
    lap_annotation.global_rules
    graph_analysis.cross_graph_guidelines.brake_and_speed

Path segments containing dots can be escaped with a backslash::

    sub_label_annotation.category_guidelines.Main\\ Labels

(Or just use the bracket form in callers if it gets ugly: pass the path
as a list to ``get_path``.)

Filter syntax
-------------
``find(collection_path, **filters)`` filters a collection (a dict whose
values are documents) by Mongo-style predicates. ``id`` is always added
to each document from its key, so ``find(..., id="MSP1")`` works.

Plain values are exact-match (``$eq``)::

    find("sub_label_annotation.labels", type="sub", parent="MSP")

Operator dicts get the full vocabulary::

    find("sub_label_annotation.labels", id={"$in": ["MSP1", "MSP2"]})
    find("sub_label_annotation.labels", description={"$regex": "trail brake"})
    find("sub_label_annotation.labels", parent={"$ne": None})
    find("sub_label_annotation.labels", annotation_guideline={"$exists": True})

Top-level logical combinators::

    find("sub_label_annotation.labels", **{
        "$or": [{"type": "main"}, {"type": "segment_type"}],
    })
    find("sub_label_annotation.labels", **{
        "$not": {"type": "circuit_section"},
    })
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _op_eq(v: Any, x: Any) -> bool:
    return v == x


def _op_ne(v: Any, x: Any) -> bool:
    return v != x


def _op_in(v: Any, x: Any) -> bool:
    try:
        return v in x
    except TypeError:
        return False


def _op_nin(v: Any, x: Any) -> bool:
    try:
        return v not in x
    except TypeError:
        return True


def _op_gt(v: Any, x: Any) -> bool:
    try:
        return v > x
    except TypeError:
        return False


def _op_gte(v: Any, x: Any) -> bool:
    try:
        return v >= x
    except TypeError:
        return False


def _op_lt(v: Any, x: Any) -> bool:
    try:
        return v < x
    except TypeError:
        return False


def _op_lte(v: Any, x: Any) -> bool:
    try:
        return v <= x
    except TypeError:
        return False


def _op_regex(v: Any, x: Any) -> bool:
    if not isinstance(v, str):
        return False
    try:
        return re.search(x, v) is not None
    except re.error:
        return False


def _op_contains(v: Any, x: Any) -> bool:
    """Substring (strings) or membership (lists/sets/tuples/dicts)."""
    if v is None:
        return False
    try:
        return x in v
    except TypeError:
        return False


def _op_exists(v: Any, x: Any) -> bool:
    """``True`` matches "field is present and not None"; ``False`` matches absent."""
    present = v is not None
    return present == bool(x)


def _op_startswith(v: Any, x: Any) -> bool:
    return isinstance(v, str) and v.startswith(x)


def _op_endswith(v: Any, x: Any) -> bool:
    return isinstance(v, str) and v.endswith(x)


def _op_size(v: Any, x: Any) -> bool:
    try:
        return len(v) == int(x)
    except (TypeError, ValueError):
        return False


def _op_all(v: Any, x: Any) -> bool:
    """Every element of x is in v (v is list-like)."""
    if v is None:
        return False
    try:
        return all(item in v for item in x)
    except TypeError:
        return False


OPERATORS: Dict[str, Callable[[Any, Any], bool]] = {
    "$eq":         _op_eq,
    "$ne":         _op_ne,
    "$in":         _op_in,
    "$nin":        _op_nin,
    "$gt":         _op_gt,
    "$gte":        _op_gte,
    "$lt":         _op_lt,
    "$lte":        _op_lte,
    "$regex":      _op_regex,
    "$contains":   _op_contains,
    "$exists":     _op_exists,
    "$startswith": _op_startswith,
    "$endswith":   _op_endswith,
    "$size":       _op_size,
    "$all":        _op_all,
}


# ---------------------------------------------------------------------------
# Path access
# ---------------------------------------------------------------------------

_MISSING = object()


def _split_path(path: str) -> List[str]:
    """Split a dotted path. Backslash-escaped dots stay in the segment."""
    if not path:
        return []
    parts: List[str] = []
    buf: List[str] = []
    i = 0
    while i < len(path):
        c = path[i]
        if c == "\\" and i + 1 < len(path) and path[i + 1] == ".":
            buf.append(".")
            i += 2
            continue
        if c == ".":
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1
    parts.append("".join(buf))
    return parts


def get_path(root: Mapping[str, Any], segments: Iterable[str], default: Any = None) -> Any:
    """Walk a dotted path through nested dicts; return ``default`` on miss."""
    cur: Any = root
    for seg in segments:
        if isinstance(cur, Mapping) and seg in cur:
            cur = cur[seg]
        elif isinstance(cur, list):
            try:
                cur = cur[int(seg)]
            except (ValueError, IndexError):
                return default
        else:
            return default
    return cur


def get(root: Mapping[str, Any], path: str, default: Any = None) -> Any:
    return get_path(root, _split_path(path), default)


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

def _match_clause(doc: Mapping[str, Any], field: str, spec: Any) -> bool:
    """Match a single field against either a value (eq) or an operator dict."""
    if isinstance(spec, Mapping) and any(k.startswith("$") for k in spec):
        for op_name, op_arg in spec.items():
            op = OPERATORS.get(op_name)
            if op is None:
                raise ValueError(f"Unknown operator: {op_name!r}")
            field_val = get(doc, field, _MISSING)
            if field_val is _MISSING:
                field_val = None
            if not op(field_val, op_arg):
                return False
        return True
    # Plain value → exact match
    field_val = get(doc, field, _MISSING)
    if field_val is _MISSING:
        return spec is None
    return field_val == spec


def matches(doc: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    """All clauses must match (AND). Special keys: $or, $and, $not."""
    for key, spec in filters.items():
        if key == "$and":
            if not all(matches(doc, sub) for sub in spec):
                return False
        elif key == "$or":
            if not any(matches(doc, sub) for sub in spec):
                return False
        elif key == "$not":
            if matches(doc, spec):
                return False
        else:
            if not _match_clause(doc, key, spec):
                return False
    return True


# ---------------------------------------------------------------------------
# Collection iteration
# ---------------------------------------------------------------------------

def iter_collection(collection: Any) -> List[Dict[str, Any]]:
    """Normalise a collection to a list of dict documents, injecting ``id``.

    Accepts:
      * ``dict[id, doc]`` — keys become each doc's ``id`` field
      * ``list[doc]`` — passed through (docs are expected to already carry id)
    """
    if collection is None:
        return []
    if isinstance(collection, Mapping):
        out: List[Dict[str, Any]] = []
        for k, v in collection.items():
            if isinstance(v, Mapping):
                doc = dict(v)
                doc.setdefault("id", k)
            else:
                doc = {"id": k, "value": v}
            out.append(doc)
        return out
    if isinstance(collection, list):
        return [d for d in collection if isinstance(d, Mapping)]
    return []


def find(
    collection: Any,
    filters: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    docs = iter_collection(collection)
    if not filters:
        return docs
    return [d for d in docs if matches(d, filters)]
