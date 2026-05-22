"""Streamlit custom component for editing templated multi-turn training entries.

The component is implemented as a single static ``index.html`` (no build
step). Python sends the initial ``entries`` / ``variables`` / ``approved``
on mount; the JS side owns all editing state and posts back a payload
only when the user clicks Save or Approve inside the iframe. Python
distinguishes saves from approvals by the ``action`` field, and tracks
``nonce`` to avoid re-acting on the same payload after a Streamlit rerun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components


_COMPONENT_DIR = Path(__file__).parent
_component_func = components.declare_component(
    "llm_entry_editor", path=str(_COMPONENT_DIR),
)


def llm_entry_editor(
    *,
    entries: List[Dict[str, Any]],
    variables: Dict[str, str],
    approved: bool,
    key: str,
) -> Optional[Dict[str, Any]]:
    """Render the editor; returns the last action payload from JS (or None).

    Payload shape::

        {
            "action": "save" | "approve",
            "entries": [{entry_id, turns: [{role, template}]}],
            "approved": bool,
            "nonce": int,           # ms timestamp, monotonically increasing
        }
    """
    return _component_func(
        entries=entries,
        variables=variables,
        approved=approved,
        key=key,
        default=None,
    )
