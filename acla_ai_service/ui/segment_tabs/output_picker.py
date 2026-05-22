"""First-time output-location popup for copy-mode annotation nodes.

When a user opens an annotation page from a freshly-created node, the
node has no ``output_key`` yet — the popup here collects the destination
*directory* and *filename* and writes them back onto the node before the
page itself renders. Existing nodes (created before this change) skip
the popup because they already carry an ``output_key``.

The popup is only shown for ``mode == MODE_COPY``. Secondary-worker and
coworker nodes write to the target's output, so they don't need their
own file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st

from app.pipelines.manifest.models import MODE_COPY, AnnotationNode, Pipeline
from app.pipelines.manifest.registry import save as save_pipeline, slugify

from .shared import get_store, register_output_dir


def _default_dir() -> str:
    """Return the default Lance store dir as a filesystem path."""
    return str(Path(get_store().store_dir).resolve())


def _suggested_name(pipeline: Pipeline, node: AnnotationNode) -> str:
    """Reasonable default cache_key when the popup first appears.

    Matches the previous auto-derived shape so a user who just clicks
    "Save & Continue" gets the same key the old form would have made.
    """
    return f"manual_segment_annotations_{pipeline.id}__{node.id}"


def needs_output_setup(node: AnnotationNode) -> bool:
    """Whether the popup should fire for this node on this open."""
    return node.mode == MODE_COPY and not node.output_key


def render_output_picker(
    pipeline: Pipeline, node: AnnotationNode,
) -> Optional[str]:
    """Show the modal and, on save, persist the chosen output location.

    Returns the resulting ``output_key`` (also written onto ``node`` and
    saved to disk) when the user confirms, or ``None`` while the dialog
    is still open. The caller should ``st.rerun()`` after a non-``None``
    return so the annotation page picks up the new key.
    """
    result_box: dict = st.session_state.setdefault(
        f"_output_picker_result_{node.id}", {}
    )

    @st.dialog("Configure annotation output", width="large")
    def _dialog() -> None:
        default_dir = _default_dir()
        default_name = _suggested_name(pipeline, node)

        st.markdown(
            f"**Node:** `{node.id}` · **Mode:** Copy from source\n\n"
            "Choose where the output dataset for this annotation will be saved."
        )

        location_choice = st.radio(
            "Location",
            options=["Lance storage (default)", "Custom directory"],
            index=0,
            key=f"_output_picker_choice_{node.id}",
            horizontal=True,
            help=f"Lance storage lives at `{default_dir}`.",
        )
        use_custom = location_choice == "Custom directory"

        if use_custom:
            directory = st.text_input(
                "Directory",
                value=st.session_state.get(
                    f"_output_picker_dir_{node.id}", default_dir,
                ),
                key=f"_output_picker_dir_{node.id}",
            )
            cleaned_dir = directory.strip()
        else:
            st.caption(f"Saving in the Lance store: `{default_dir}`")
            cleaned_dir = default_dir

        filename = st.text_input(
            "Filename / dataset name",
            value=st.session_state.get(
                f"_output_picker_name_{node.id}", default_name,
            ),
            key=f"_output_picker_name_{node.id}",
            help="The cache key. The Lance dataset will live at "
                 "`<directory>/<filename>.lance`.",
        )

        cleaned_name = slugify(filename) if filename else ""

        if cleaned_name and cleaned_name != filename:
            st.caption(f"Stored as: `{cleaned_name}` (slugified)")

        if cleaned_dir and cleaned_name:
            st.caption(
                f"Will write to: `{Path(cleaned_dir) / (cleaned_name + '.lance')}`"
            )

        col_save, col_cancel = st.columns([1, 1])
        with col_save:
            if st.button(
                "Save & Continue", type="primary",
                use_container_width=True,
                disabled=not (cleaned_dir and cleaned_name),
                key=f"_output_picker_save_{node.id}",
            ):
                node.output_key = cleaned_name
                if not use_custom or Path(cleaned_dir).resolve() == Path(default_dir).resolve():
                    # Falls in the Lance store — leave output_dir unset so
                    # the manifest stays tidy and the store uses its default.
                    node.output_dir = None
                else:
                    node.output_dir = cleaned_dir
                    try:
                        Path(cleaned_dir).mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        st.error(f"Could not create directory: {exc}")
                        return
                save_pipeline(pipeline)
                register_output_dir(node.output_key, node.output_dir)
                result_box["output_key"] = node.output_key
                st.rerun()
        with col_cancel:
            if st.button(
                "Cancel", use_container_width=True,
                key=f"_output_picker_cancel_{node.id}",
            ):
                # Drop the node-active flag so the caller falls back to
                # the pipeline view instead of staring at a blank page.
                for k in ("active_view", "pipeline_routed_view",
                          "pipeline_active_node_id"):
                    st.session_state.pop(k, None)
                st.rerun()

    if "output_key" in result_box:
        ok = result_box.pop("output_key")
        return ok

    _dialog()
    return None
