"""Configuration: app settings and infrastructure cache config.

Single source of truth — replaces app/core/ and app/config/, which
were collapsed in refactor/hexagonal-v2 Step 12.
"""
from app.infra.config.settings import settings

__all__ = ["settings"]
