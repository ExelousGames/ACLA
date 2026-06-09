"""Annotation provider registry models.

These models describe AI services for the annotation pipeline only. They are
not used by the racing-engineer chatbot or voice pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


PromptMode = Literal["local_pipeline", "tool_agent"]
RunnerKind = Literal["local_vlm", "claude_cli", "openai_compatible"]


@dataclass(frozen=True)
class ProviderModel:
    id: str
    label: str
    max_context: Optional[int] = None
    max_new_tokens: Optional[int] = None


@dataclass(frozen=True)
class ProviderOption:
    key: str
    label: str
    kind: Literal["text", "number", "checkbox", "select"]
    default: Any = None
    help: str = ""
    options: List[Any] = field(default_factory=list)
    advanced: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None


@dataclass(frozen=True)
class AnnotationProvider:
    id: str
    label: str
    runner: RunnerKind
    prompt_mode: PromptMode
    models: List[ProviderModel] = field(default_factory=list)
    options: List[ProviderOption] = field(default_factory=list)
    required_settings: List[str] = field(default_factory=list)
    default_model: str = ""
    description: str = ""
    configured: bool = True

    def model_ids(self) -> List[str]:
        return [m.id for m in self.models]

    def default_model_id(self) -> str:
        if self.default_model:
            return self.default_model
        if self.models:
            return self.models[0].id
        return ""

    def option_defaults(self) -> Dict[str, Any]:
        return {opt.key: opt.default for opt in self.options}
