"""Shared application modules used across service boundaries.

- ``contracts``                 AgentRequest/AgentResponse and callbacks.
- domain catalog/telemetry modules moved here from the old domain package.
"""

from app.shared.contracts import *  # noqa: F401,F403
