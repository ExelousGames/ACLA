"""Reusable input/output key selector for annotation components."""

from __future__ import annotations

from typing import Callable, Optional

import streamlit as st


KeyOption = dict[str, str]
KeyReference = dict[str, str]


def _option_paths(options: list[KeyOption]) -> list[str]:
    return [option["path"] for option in options]


def _select_index(options: list[KeyOption], preferred_path: str) -> int:
    paths = _option_paths(options)
    return paths.index(preferred_path) if preferred_path in paths else 0


def _format_option(option: KeyOption) -> str:
    label = option["label"]
    path = option["path"]
    if not path:
        return label
    return label if label == path else f"{label} · {path}"


def _same_reference(
    left: Optional[KeyReference],
    right: KeyReference,
) -> bool:
    if not left:
        return False
    return (
        left.get("output_path") == right.get("output_path")
        and left.get("input_path") == right.get("input_path")
    )


def render_annotation_key_reference(
    *,
    component_id: str,
    output_options: list[KeyOption],
    input_options: list[KeyOption],
    default_input: KeyOption,
    saved_reference: Optional[KeyReference],
    input_preview: Callable[[str], str],
    save_reference: Callable[[KeyReference], None],
) -> tuple[bool, Optional[KeyReference]]:
    """Render and persist one annotation component's output/input key pair."""
    saved_output = str((saved_reference or {}).get("output_path") or "")
    saved_input = str((saved_reference or {}).get("input_path") or "")

    placeholder = {"label": "-- pick field --", "path": ""}
    output_display_options = [placeholder] + output_options
    output_index = _select_index(output_display_options, saved_output)

    input_paths = set(_option_paths(input_options))
    preferred_input = saved_input if saved_input in input_paths else default_input["path"]
    input_index = _select_index(input_options, preferred_input)

    cols = st.columns(2)
    with cols[0]:
        picked_output = st.selectbox(
            "Output foreign-key field",
            output_display_options,
            index=output_index,
            key=f"key_ref_output_{component_id}",
            format_func=_format_option,
        )
    with cols[1]:
        picked_input = st.selectbox(
            "Input primary key",
            input_options,
            index=input_index,
            key=f"key_ref_input_{component_id}",
            format_func=_format_option,
        )
        st.caption(input_preview(picked_input["path"]))

    if not picked_output["path"]:
        if saved_reference:
            st.warning("The saved protection fields are no longer valid for this input schema.")
        else:
            st.info("Pick the output foreign-key field to enable Copy from source.")
        return False, None

    picked = {
        "label": f"{picked_output['label']} -> {picked_input['label']}",
        "output_path": picked_output["path"],
        "input_path": picked_input["path"],
    }
    st.caption(
        "Protected copied rows are matched by this component's output foreign key and input primary key.",
        help=(
            "Pick the output field that stores the input reference, then "
            "pick the input field that uniquely identifies source rows. "
            "Referenced input entries are preserved during Update from source."
        ),
    )
    if not _same_reference(saved_reference, picked):
        save_reference(dict(picked))
        st.rerun()

    return True, picked
