"""Shared data structure for prompt/response training examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PromptResponseExample:
    """Normalized prompt/response payload used across LLM training pipelines."""

    prompt: str
    response: str
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        """Return a serializable dictionary representation of the example."""

        # Construct messages list for chat-based training (AutoTrain preferred format)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.prompt})
        messages.append({"role": "assistant", "content": self.response})

        # Construct text field for legacy/simple SFT
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")
        parts.append(f"User: {self.prompt}")
        parts.append(f"Assistant: {self.response}")
        full_text = "\n\n".join(parts)

        #This structure allows AutoTrain to automatically detect the correct columns (text or messages) without manual mapping, while preserving your metadata in a separate column that won't interfere with training.
        record = {
            "text": full_text,
            "messages": messages,
            "prompt": self.prompt,
            "response": self.response,
        }
        if self.system_prompt:
            record["system_prompt"] = self.system_prompt
        if self.metadata:
            record["metadata"] = self.metadata
        return record

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "PromptResponseExample":
        """Construct an example from a serialized record, validating structure."""

        if not isinstance(record, dict):
            raise TypeError("Prompt/response record must be a dictionary")

        prompt = record.get("prompt")
        response = record.get("response")
        system_prompt = record.get("system_prompt")
        metadata = record.get("metadata", {})

        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError("Prompt/response records require string prompt and response fields")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string when provided")
        if not isinstance(metadata, dict):
            metadata = {}

        # Use shallow copies so downstream mutations do not affect shared state
        safe_metadata = dict(metadata)

        return cls(
            prompt=prompt,
            response=response,
            system_prompt=system_prompt,
            metadata=safe_metadata,
        )
