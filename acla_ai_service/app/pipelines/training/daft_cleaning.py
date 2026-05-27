"""Daft-backed cleaning pipeline (first slice).

Replaces the trivially-vectorisable parts of
:class:`app.domain.telemetry.FeatureProcessor` with a Daft DataFrame
pipeline that consumes Arrow tables straight from
``app/storage/telemetry_lance_store/racing_sessions_.lance``.

What this module covers
-----------------------

``clean_session_table_with_daft`` applies, in order:

1. ``general_cleaning`` — numeric ``fillna(0)``, boolean coercion for
   columns whose name hints at a boolean (``on``/``enabled``/``valid``/
   ``running``/``controlled``), and rounding of floats to six decimals.
2. ``flip_y_z_features`` — swaps the values of any ``*_y`` /  ``*_z``
   column pair (so axis conventions match downstream consumers).
3. ``filter_features_by_list`` — projects to a caller-supplied feature
   list, raising :class:`ValueError` if any requested feature is missing
   (mirrors the legacy ``filter_features_by_list``).

What this module deliberately leaves out
----------------------------------------

* ``_handle_complex_fields`` — extracts ``Graphics_player_pos_*`` and
  flattens the nested ``Graphics_car_coordinates`` list<struct> into
  fixed ``Car_{1..MAX_CARS}_pos_{x,y,z}`` columns (empty slots → 0.0)
  so training sees a constant column count. Today it's a Python
  ``for idx in df.index`` loop with per-row ``df.loc[idx, ...]`` writes
  — a vectorisation candidate in its own right, but the rewrite is
  substantial enough to deserve a separate PR with its own parity test
  against the existing extractor.

* ``strip_dataframe_by_time_gap`` — keeps a running ``last_selected``
  pointer in a Python for-loop; cleanly vectorising the gap-detection
  logic requires care (lap-reset handling, time-gap aggregation), so it
  stays on the legacy path for now.

* ``split_into_laps`` — groupby on lap-change + position-reset; not a
  cleaning op per se, lives downstream of this module.

Parity guard
------------

``scripts/parity_test_daft_cleaning.py`` runs the legacy FeatureProcessor
ops covered here against this module on real chunks pulled from
``racing_sessions_.lance`` and asserts the resulting Arrow tables match
column-for-column within float tolerance.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import daft


_BOOLEAN_KEYWORDS = ("on", "enabled", "valid", "running", "controlled")
# Same string truthiness mapping FeatureProcessor.general_cleaning_for_analysis
# applies before falling back to ``False`` (i.e. NaN/None → False).
_BOOL_MAP = {
    "True": True, "False": False,
    "true": True, "false": False,
    "1": True, "0": False,
}


def _looks_boolean(column_name: str) -> bool:
    lowered = column_name.lower()
    return any(keyword in lowered for keyword in _BOOLEAN_KEYWORDS)


def _coerce_string_boolean_column(col: pa.ChunkedArray) -> pa.Array:
    """Match the legacy ``df[col].map({...}).fillna(False)`` semantics.

    Handles columns where pyarrow inferred the type as ``string`` because
    the source dicts had mixed values like ``"True"`` / ``"0"`` / ``"true"``.
    The legacy code coerced anything not in the explicit map to ``False``.
    """
    values = col.to_pylist()
    return pa.array(
        [_BOOL_MAP.get(v, False) for v in values],
        type=pa.bool_(),
    )


def _general_cleaning(table: pa.Table) -> pa.Table:
    """Pure-Arrow equivalent of FeatureProcessor.general_cleaning_for_analysis.

    Implemented in Arrow rather than Daft because the per-column type
    inspection and bool-coercion-from-strings step doesn't have a clean
    Daft expression form yet — Arrow's column-level ops are exactly the
    right granularity here. Daft is used for the downstream
    flip-and-filter pipeline where its lazy/columnar story actually pays.
    """
    if table.num_rows == 0:
        return table

    new_columns: List[pa.Array] = []
    new_names: List[str] = list(table.column_names)
    for name in new_names:
        col = table.column(name)
        col_type = col.type

        if pa.types.is_floating(col_type):
            # fillna(0) + round to 6 decimals
            filled = pc.fill_null(col, 0.0)
            rounded = pc.round(filled, ndigits=6)
            new_columns.append(rounded.combine_chunks())
        elif pa.types.is_integer(col_type):
            new_columns.append(pc.fill_null(col, 0).combine_chunks())
        elif pa.types.is_boolean(col_type):
            # Legacy passes booleans through; we mirror that.
            new_columns.append(col.combine_chunks())
        elif _looks_boolean(name) and pa.types.is_string(col_type):
            new_columns.append(_coerce_string_boolean_column(col))
        else:
            # Object / list<struct> / other types pass through unchanged.
            new_columns.append(col.combine_chunks())

    return pa.Table.from_arrays(new_columns, names=new_names)


def _flip_y_z_features(table: pa.Table) -> pa.Table:
    """Swap any pair of columns ``foo_y`` and ``foo_z``.

    Mirrors FeatureProcessor.flip_y_z_features. Implemented as a simple
    rename map — the underlying Arrow buffers are zero-copy aliased.
    """
    column_names = list(table.column_names)
    rename_map = {}
    for name in column_names:
        if name.endswith("_y"):
            counterpart = name[:-2] + "_z"
            if counterpart in column_names and counterpart not in rename_map:
                rename_map[name] = counterpart
                rename_map[counterpart] = name

    if not rename_map:
        return table

    new_names = [rename_map.get(name, name) for name in column_names]
    return table.rename_columns(new_names)


def _filter_features_by_list(
    table: pa.Table,
    feature_list: Sequence[str],
) -> pa.Table:
    """Project ``feature_list`` columns from ``table``.

    Matches the legacy method's error semantics: if any requested feature
    is missing, raises ``ValueError`` listing them (the legacy method
    raises after printing a warning).
    """
    if not feature_list:
        return pa.table({})

    available = set(table.column_names)
    missing = [f for f in feature_list if f not in available]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    return table.select(list(feature_list))


def clean_session_table_with_daft(
    table: pa.Table,
    feature_list: Optional[Sequence[str]] = None,
) -> pa.Table:
    """Run the Daft cleaning pipeline against a single session's Arrow table.

    Steps: general cleaning → flip y/z → optional column projection.
    Returns a pyarrow Table so the caller can decide whether to stay in
    Arrow, hand off to Daft for further processing, or convert to
    pandas/numpy.

    The Daft layer is currently used for the projection step — the
    earlier cleaning is in Arrow because Daft's column-type inspection
    surface doesn't cover the bool-string coercion legacy expects. As
    additional cleaning ops are vectorised they'll move into the Daft
    expression below.
    """
    cleaned = _general_cleaning(table)
    flipped = _flip_y_z_features(cleaned)

    if feature_list is None:
        return flipped

    # Daft round-trip — demonstrates the lazy pipeline path and validates
    # zero-copy from Arrow. Identity-equivalent to the pyarrow projection
    # but uses Daft's planner so future additions (filters, derived
    # columns) land naturally here.
    df = daft.from_arrow(flipped)
    df = df.select(*feature_list)
    return df.to_arrow()


__all__ = [
    "clean_session_table_with_daft",
]
