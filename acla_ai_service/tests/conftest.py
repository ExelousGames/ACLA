"""Test scaffolding.

Stubs out heavy third-party deps that the project's package-level ``__init__``
files transitively import. Tests in this directory exercise pure helpers in
the agent framework that have no need for OpenAI / FastAPI / pandas.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _stub(name: str) -> None:
    if name not in sys.modules:
        sys.modules[name] = MagicMock()


# Block transitive imports triggered by ``app.services.__init__`` (openai),
# ``app.models`` (pandas, sentence_transformers), and any others that the
# agent_framework / describe_graphs modules don't actually need at import.
for mod in (
    "openai",
    "openai.types",
    "openai.types.chat",
    "fastapi",
    "fastapi.responses",
):
    _stub(mod)
