"""Agent specs auto-registered into AGENT_REGISTRY on package import.

Importing this package side-effectfully registers every agent below, so any
consumer that does ``from app.services.llm.agents import ...`` (or just
``import app.services.llm.agents``) gets a populated registry.

Import order matters: leaf agents must register before agents that
``delegates_to`` them so the registry is populated when a parent's spec is
compiled.
"""

# Leaves first.
from app.services.llm.agents import zoom                       # noqa: F401
from app.services.llm.agents import label_verifier             # noqa: F401

# Meta-agents that delegate to the leaves.
from app.services.llm.agents import describe_graphs            # noqa: F401

# Root agent that orchestrates everything.
from app.services.llm.agents import annotation_root            # noqa: F401
