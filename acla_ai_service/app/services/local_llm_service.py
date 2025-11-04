"""Local LLM training and inference utilities for telemetry forecasting.

This module replaces the legacy transformer-only pipeline with a HuggingFace-
compatible workflow that can be fine-tuned locally (via LoRA adapters) and
used for inference to generate both future telemetry and human-readable
explanations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:  # pragma: no cover - guarding runtime deps
    raise ImportError(
        "transformers is required for LocalTelemetryLLM. Please install `transformers`."
    ) from exc

try:
    from peft import (
        LoraConfig,
        PeftConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
except ImportError as exc:  # pragma: no cover - guarding runtime deps
    raise ImportError("peft is required for LoRA fine-tuning. Install `peft`.") from exc

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LocalLLMConfig:
    """Configuration options for loading and training the local LLM."""

    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None

    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_gradient_checkpointing: bool = True
    use_lora: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    max_seq_length: int = 1024
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    weight_decay: float = 0.01
    num_train_epochs: int = 3
    max_steps: Optional[int] = None

    logging_steps: int = 20
    save_steps: int = 200
    eval_steps: Optional[int] = None
    save_total_limit: int = 3

    bf16: bool = torch.cuda.is_available()
    fp16: bool = False
    dataloader_num_workers: int = 2
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Generation defaults
    generation_max_new_tokens: int = 256
    generation_temperature: float = 0.9
    generation_top_p: float = 0.95
    generation_do_sample: bool = True


@dataclass
class GenerationRequest:
    """Payload for inference requests."""

    system_prompt: str
    user_prompt: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _extract_messages(record: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """Extract system/user/assistant text from a chat-formatted dictionary."""

    messages = record.get("messages")
    if not isinstance(messages, list):
        return None

    system_messages: List[str] = []
    user_messages: List[str] = []
    assistant_messages: List[str] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if role == "system":
            system_messages.append(content)
        elif role == "user":
            user_messages.append(content)
        elif role == "assistant":
            assistant_messages.append(content)

    if not user_messages or not assistant_messages:
        return None

    system_text = "\n\n".join(system_messages)
    user_text = "\n\n".join(user_messages)
    assistant_text = assistant_messages[-1]
    return system_text, user_text, assistant_text


class TelemetryPromptDataset(Dataset):
    """PyTorch dataset for instruction tuning using prompt/completion pairs."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: "AutoTokenizer",
        max_seq_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples: List[Tuple[List[int], List[int]]] = []
        self.metadata: List[Dict[str, Any]] = []

        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

        with jsonl_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Skipping malformed JSON at line %s", line_number)
                    continue

                parsed = _extract_messages(record)
                if not parsed:
                    continue

                prompt_text, response_text = self._format_prompt(parsed)
                sample = self._build_sample(prompt_text, response_text)
                if sample is None:
                    continue

                input_ids, labels = sample
                self.samples.append((input_ids, labels))
                prompt_length = len(input_ids) - sum(1 for value in labels if value != -100)
                self.metadata.append({
                    "line_number": line_number,
                    "prompt_tokens": prompt_length,
                    "total_tokens": len(input_ids),
                })

        if not self.samples:
            raise ValueError("No valid samples were parsed from the dataset")

        LOGGER.info("Loaded %d samples for fine-tuning", len(self.samples))

    def _format_prompt(self, parsed: Tuple[str, str, str]) -> Tuple[str, str]:
        system_text, user_text, assistant_text = parsed

        system_block = f"[SYSTEM]\n{system_text}\n[/SYSTEM]\n\n" if system_text else ""
        user_block = f"[USER]\n{user_text}\n[/USER]\n\n"
        assistant_prefix = "[ASSISTANT]\n"

        prompt = f"{system_block}{user_block}{assistant_prefix}"
        response = f"{assistant_text}{self.tokenizer.eos_token}"
        return prompt, response

    def _build_sample(self, prompt_text: str, response_text: str) -> Optional[Tuple[List[int], List[int]]]:
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        response_ids = self.tokenizer(
            response_text,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + response_ids
        if len(input_ids) > self.max_seq_length:
            LOGGER.debug("Skipping sample exceeding max_seq_length=%s", self.max_seq_length)
            return None

        labels = [-100] * len(prompt_ids) + response_ids
        return input_ids, labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self.samples[idx]
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_tensor)
        return {
            "input_ids": input_tensor,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
        }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


class LocalTelemetryLLM:
    """High-level wrapper for local LLM fine-tuning and inference."""

    def __init__(self, config: Optional[LocalLLMConfig] = None) -> None:
        self.config = config or LocalLLMConfig()
        self.tokenizer = None
        self.model = None

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _ensure_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return

        tokenizer_name = self.config.tokenizer_name or self.config.base_model
        LOGGER.info("Loading tokenizer %s", tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=self.config.cache_dir,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.padding_side != "right":
            self.tokenizer.padding_side = "right"

    def _load_model(self, for_training: bool, adapter_path: Optional[Path] = None) -> torch.nn.Module:
        self._ensure_tokenizer()

        torch_dtype = torch.bfloat16 if self.config.bf16 else (
            torch.float16 if self.config.fp16 else torch.float32
        )

        LOGGER.info("Loading base model %s", self.config.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            cache_dir=self.config.cache_dir,
            load_in_8bit=self.config.load_in_8bit,
            load_in_4bit=self.config.load_in_4bit,
            torch_dtype=None if (self.config.load_in_8bit or self.config.load_in_4bit) else torch_dtype,
            device_map="auto",
        )

        if self.config.use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(**(self.config.gradient_checkpointing_kwargs or {}))

        if adapter_path:
            adapter_path = Path(adapter_path)
            LOGGER.info("Loading LoRA adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
            return model

        if for_training and self.config.use_lora:
            LOGGER.info("Wrapping model with LoRA (r=%s, alpha=%s)", self.config.lora_r, self.config.lora_alpha)
            if self.config.load_in_8bit or self.config.load_in_4bit:
                model = prepare_model_for_kbit_training(model)

            target_modules = self.config.lora_target_modules
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_path: Path,
        output_dir: Path,
        eval_dataset_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Fine-tune the local LLM on the provided dataset."""

        self.model = self._load_model(for_training=True)
        self.model.train()

        dataset = TelemetryPromptDataset(
            jsonl_path=dataset_path,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        eval_dataset: Optional[TelemetryPromptDataset] = None
        if eval_dataset_path:
            eval_dataset = TelemetryPromptDataset(
                jsonl_path=eval_dataset_path,
                tokenizer=self.tokenizer,
                max_seq_length=self.config.max_seq_length,
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=self.config.max_steps or -1,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=False,
            report_to=["tensorboard"],
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        train_result = trainer.train()
        trainer.save_state()

        self._save_model(output_dir)

        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset)
        if eval_dataset is not None:
            metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        return metrics

    def _save_model(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.use_lora:
            LOGGER.info("Saving LoRA adapter to %s", output_dir)
            if isinstance(self.model, PeftModel):
                self.model.save_pretrained(output_dir)
            else:
                raise RuntimeError("Model is expected to be a PeftModel when use_lora=True")
        else:
            LOGGER.info("Saving full model to %s", output_dir)
            self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def load_for_inference(self, adapter_path: Optional[Path] = None) -> None:
        """Load base model (and optional adapter) for inference."""

        self.model = self._load_model(for_training=False, adapter_path=adapter_path)
        self.model.eval()

    def generate(self, request: GenerationRequest) -> str:
        """Generate telemetry narrative using the fine-tuned model."""

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_for_inference() first.")

        self._ensure_tokenizer()

        prompt = self._format_generation_prompt(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": request.max_new_tokens or self.config.generation_max_new_tokens,
            "temperature": request.temperature or self.config.generation_temperature,
            "top_p": request.top_p or self.config.generation_top_p,
            "do_sample": request.do_sample if request.do_sample is not None else self.config.generation_do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    @staticmethod
    def _format_generation_prompt(system_prompt: str, user_prompt: str) -> str:
        system_block = f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n\n" if system_prompt else ""
        user_block = f"[USER]\n{user_prompt}\n[/USER]\n\n"
        assistant_prefix = "[ASSISTANT]\n"
        return f"{system_block}{user_block}{assistant_prefix}"


__all__ = [
    "LocalLLMConfig",
    "GenerationRequest",
    "TelemetryPromptDataset",
    "LocalTelemetryLLM",
]
