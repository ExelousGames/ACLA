"""Local LLM training and inference utilities for telemetry forecasting.

This module replaces the legacy transformer-only pipeline with a HuggingFace-
compatible workflow that can be fine-tuned locally (via LoRA adapters) and
used for inference to generate both future telemetry and human-readable
explanations.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from collections.abc import Mapping

from app.core.config import settings

import torch
from torch.utils.data import Dataset

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
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

# Determine persistent model cache directory
# app/services/llm/local_llm_service.py -> ../../../models/huggingface_cache
_SERVICE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HF_CACHE = str(_SERVICE_ROOT / "models" / "huggingface_cache")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LocalLLMConfig:
    """Configuration options for loading and training the local LLM."""

    base_model: str = "mistralai/Ministral-3-14B-Base-2512"
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = DEFAULT_HF_CACHE
    gguf_file: Optional[str] = None
    default_adapter: Optional[str] = None
    provider: str = "transformers"  # 'transformers', 'llama_cpp'

    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    use_gradient_checkpointing: bool = True
    use_lora: bool = True
    device_map: Union[str, Dict[str, Union[int, str]]] = "auto"
    max_memory: Optional[Dict[str, Union[int, str]]] = None
    offload_folder: Optional[str] = None
    offload_state_dict: bool = False
    low_cpu_mem_usage: bool = True

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

    max_seq_length: int = 2353642
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

    use_hf_datasets: bool = True
    hf_streaming: bool = False
    hf_num_proc: Optional[int] = None

    # Generation defaults
    generation_max_new_tokens: int = 256
    generation_temperature: float = 0.9
    generation_top_p: float = 0.95
    generation_do_sample: bool = True


@dataclass
class GenerationRequest:
    """Payload for inference requests."""

    user_prompt: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    do_sample: Optional[bool] = None
    model_id: Optional[str] = None
    api_token: Optional[str] = None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _extract_messages(record: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    """Extract system/user/assistant text from a normalized prompt/response record."""
    if isinstance(record, Mapping):
        print("record is Mapping:")
        record = dict(record)
    if not isinstance(record, dict):
        return None

    prompt_text = record.get("prompt")
    response_text = record.get("response") or record.get("completion")
    if not isinstance(prompt_text, str):
        return None
    if not isinstance(response_text, str):
        return None

    system_text = record.get("system_prompt")
    if not isinstance(system_text, str):
        system_text = ""

    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        meta_system = metadata.get("system_prompt")
        if isinstance(meta_system, str) and not system_text:
            system_text = meta_system

    return system_text, prompt_text, response_text


def _format_prompt_and_response(
    tokenizer: "AutoTokenizer",
    parsed: Tuple[str, str, str],
) -> Tuple[str, str]:
    """Create prompt/response text blocks for a parsed example."""

    system_text, user_text, assistant_text = parsed

    system_block = f"[SYSTEM]\n{system_text}\n[/SYSTEM]\n\n" if system_text else ""
    user_block = f"[USER]\n{user_text}\n[/USER]\n\n"
    assistant_prefix = "[ASSISTANT]\n"

    prompt = f"{system_block}{user_block}{assistant_prefix}"
    response = f"{assistant_text}{tokenizer.eos_token}"
    return prompt, response


def _build_tokenized_sample(
    tokenizer: "AutoTokenizer",
    prompt_text: str,
    response_text: str,
    max_seq_length: int,
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    """Tokenize prompt/response blocks into training arrays."""

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    response_ids = tokenizer(
        response_text,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_seq_length:
        total_len = len(input_ids)
        prompt_len = len(prompt_ids)
        response_len = len(response_ids)
        print(
            "Skipping sample exceeding max_seq_length=%s (total_tokens=%s, prompt_tokens=%s, response_tokens=%s)",
            max_seq_length,
            total_len,
            prompt_len,
            response_len,
        )
        raise ValueError(
            "Skipping sample exceeding max_seq_length="
            f"{max_seq_length} (total_tokens={total_len}, prompt_tokens={prompt_len}, response_tokens={response_len})"
        )

    labels = [-100] * len(prompt_ids) + response_ids
    attention_mask = [1] * len(input_ids)
    return input_ids, labels, attention_mask


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
        self.samples: List[Tuple[List[int], List[int], List[int]]] = []
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

                prompt_text, response_text = _format_prompt_and_response(self.tokenizer, parsed)
                sample = _build_tokenized_sample(
                    self.tokenizer,
                    prompt_text,
                    response_text,
                    self.max_seq_length,
                )
                if sample is None:
                    continue

                input_ids, labels, attention_mask = sample
                self.samples.append((input_ids, labels, attention_mask))
                response_tokens = sum(1 for value in labels if value != -100)
                prompt_length = len(input_ids) - response_tokens
                self.metadata.append({
                    "line_number": line_number,
                    "prompt_tokens": prompt_length,
                    "total_tokens": len(input_ids),
                })

        if not self.samples:
            raise ValueError("No valid samples were parsed from the dataset")

        LOGGER.info("Loaded %d samples for fine-tuning", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = self.samples[idx]
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_tensor = torch.tensor(attention_mask, dtype=torch.long)
        return {
            "input_ids": input_tensor,
            "attention_mask": attention_tensor,
            "labels": labels_tensor,
        }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


class LocalTelemetryLLM:
    """High-level wrapper for local LLM fine-tuning and inference."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LocalTelemetryLLM, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[LocalLLMConfig] = None) -> None:
        if getattr(self, "_initialized", False):
            return

        self.config = config or LocalLLMConfig()
        self.tokenizer = None
        self.processor = None
        self.model = None
        self._initialized = True

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------
    def _raise_missing_local_resource(self, resource_name: str, cause: Exception) -> None:
        """Raise a helpful error when required local files are missing or download fails."""
        
        hint = (
            f"Failed to load or automatic download failed for '{resource_name}'. "
            f"Original error: {cause}"
        )
        # We don't want to enforce manual downloads, transformers can do it
        # Try adjusting your model ID, HF_API_TOKEN or connection
        raise RuntimeError(hint) from cause

    def _ensure_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return

        tokenizer_name = self.config.tokenizer_name or self.config.base_model
        LOGGER.info("Loading tokenizer %s", tokenizer_name)

        tokenizer_kwargs = {
            "cache_dir": self.config.cache_dir,
            "token": settings.hf_api_token,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.gguf_file:
            tokenizer_kwargs["gguf_file"] = self.config.gguf_file

        # Try loading AutoProcessor first (recommended for multimodal models like Mistral 3)
        try:
            self.processor = AutoProcessor.from_pretrained(
                tokenizer_name,
                **tokenizer_kwargs,
            )
            if hasattr(self.processor, "tokenizer"):
                self.tokenizer = self.processor.tokenizer
                LOGGER.info("Loaded AutoProcessor and extracted tokenizer")
        except Exception as e:
            LOGGER.debug("AutoProcessor load failed or not applicable: %s", e)
            self.processor = None

        if self.tokenizer is not None:
             # Ensure padding settings are correct even if loaded via processor
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.padding_side != "right":
                self.tokenizer.padding_side = "right"
            return
        
        # Try loading fast tokenizer first, fall back to slow tokenizer if it fails
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,
                **tokenizer_kwargs,
            )
        except OSError as os_error:
            self._raise_missing_local_resource(tokenizer_name, os_error)
        except Exception as e:
            msg = str(e)
            if "ModelWrapper" in msg:
                LOGGER.warning(
                    "Tokenization schema error detected. Your `transformers` version may be too old for this model (e.g. requires Tekken support)."
                )

            LOGGER.warning(
                "Failed to load fast tokenizer for %s: %s. Attempting slow tokenizer fallback.",
                tokenizer_name,
                msg
            )
            
            fallback_kwargs = {
                "use_fast": False,
                **tokenizer_kwargs,
            }

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name,
                    **fallback_kwargs,
                )
            except OSError as fallback_os_error:
                self._raise_missing_local_resource(tokenizer_name, fallback_os_error)
            except Exception as e2:
                LOGGER.error("Failed to load slow tokenizer as well: %s", str(e2))
                raise
        
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
        load_kwargs: Dict[str, Any] = {
            "cache_dir": self.config.cache_dir,
            "dtype": None if (self.config.load_in_8bit or self.config.load_in_4bit) else torch_dtype,
            "device_map": self.config.device_map,
            "token": settings.hf_api_token,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.gguf_file:
            load_kwargs["gguf_file"] = self.config.gguf_file

        if self.config.load_in_8bit or self.config.load_in_4bit:
            LOGGER.info("Configuring quantization: 8bit=%s, 4bit=%s", self.config.load_in_8bit, self.config.load_in_4bit)
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.config.load_in_8bit,
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype if self.config.load_in_4bit else None
            )
            load_kwargs["quantization_config"] = quantization_config

        if self.config.max_memory:
            load_kwargs["max_memory"] = self.config.max_memory

        if self.config.offload_folder:
            offload_path = Path(self.config.offload_folder)
            offload_path.mkdir(parents=True, exist_ok=True)
            load_kwargs["offload_folder"] = str(offload_path)

        if self.config.offload_state_dict:
            load_kwargs["offload_state_dict"] = True

        if self.config.low_cpu_mem_usage is not None:
            load_kwargs["low_cpu_mem_usage"] = self.config.low_cpu_mem_usage

        try:
            LOGGER.info("Attempting to load model with AutoModelForCausalLM")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                **load_kwargs,
            )
        except (OSError, ValueError, RuntimeError) as e:
            LOGGER.info("AutoModelForCausalLM failed (%s), falling back to AutoModelForImageTextToText", str(e))
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    self.config.base_model,
                    **load_kwargs,
                )
            except OSError as load_error:
                self._raise_missing_local_resource(self.config.base_model, load_error)

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

        use_hf_datasets = self.config.use_hf_datasets
        streaming_mode = use_hf_datasets and self.config.hf_streaming

        if streaming_mode and not self.config.max_steps:
            raise ValueError(
                "Streaming Hugging Face datasets require 'max_steps' to be set in LocalLLMConfig."
            )

        if use_hf_datasets:
            dataset = self._prepare_hf_prompt_dataset(
                dataset_path=dataset_path,
                streaming=self.config.hf_streaming,
            )
        else:
            dataset = TelemetryPromptDataset(
                jsonl_path=dataset_path,
                tokenizer=self.tokenizer,
                max_seq_length=self.config.max_seq_length,
            )

        eval_dataset: Optional[Any] = None
        if eval_dataset_path:
            if use_hf_datasets:
                eval_dataset = self._prepare_hf_prompt_dataset(
                    dataset_path=eval_dataset_path,
                    streaming=False,
                )
            else:
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
            eval_strategy="steps" if eval_dataset is not None else "no",
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

        class DeviceAwareTrainer(Trainer):
            """Trainer that respects pre-sharded device maps during model move."""

            def _should_skip_device_move(
                self,
                model: torch.nn.Module,
                *,
                _visited: Optional[set[int]] = None,
            ) -> bool:
                if _visited is None:
                    _visited = set()

                model_id = id(model)
                if model_id in _visited:
                    return False
                _visited.add(model_id)

                # HuggingFace populates `hf_device_map` when dispatching with accelerate.
                device_map = getattr(model, "hf_device_map", None)
                if device_map:
                    return True

                # LoRA models wrap the base model; check nested structures to avoid recursion.
                base_model = getattr(model, "base_model", None)
                if base_model and self._should_skip_device_move(base_model, _visited=_visited):  # type: ignore[arg-type]
                    return True

                inner_model = getattr(base_model, "model", None) if base_model else None
                if inner_model and self._should_skip_device_move(inner_model, _visited=_visited):  # type: ignore[arg-type]
                    return True

                return False

            def _move_model_to_device(self, model, device):  # type: ignore[override]
                if self._should_skip_device_move(model):
                    LOGGER.info("Skipping automatic model.to(%s); model already dispatched via device map", device)
                    return model
                return super()._move_model_to_device(model, device)

        trainer = DeviceAwareTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Log dataset size before training
        dataset_size = getattr(dataset, "num_rows", None)
        if dataset_size is None and hasattr(dataset, "__len__"):
            try:
                dataset_size = len(dataset)
            except TypeError:
                dataset_size = "unknown"
        
        LOGGER.info("="*60)
        LOGGER.info("Starting LLM fine-tuning")
        LOGGER.info(f"Training samples: {dataset_size}")
        LOGGER.info(f"Epochs: {self.config.num_train_epochs}")
        LOGGER.info(f"Max steps: {self.config.max_steps or 'auto'}")
        LOGGER.info("="*60)

        try:
            train_result = trainer.train()
            trainer.save_state()
            LOGGER.info("="*60)
            LOGGER.info("Training completed successfully")
            LOGGER.info("="*60)
        except Exception as e:
            LOGGER.error("="*60)
            LOGGER.error(f"TRAINING FAILED: {type(e).__name__}: {str(e)}")
            LOGGER.error("="*60)
            raise RuntimeError(f"Training execution failed: {e}") from e

        try:
            self._save_model(output_dir)
            LOGGER.info(f"Model saved successfully to {output_dir}")
        except Exception as e:
            LOGGER.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Model save failed: {e}") from e

        metrics = train_result.metrics

        train_samples = getattr(dataset, "num_rows", None)
        if train_samples is None and hasattr(dataset, "__len__"):
            try:
                train_samples = len(dataset)
            except TypeError:
                train_samples = None
        if train_samples is not None:
            metrics["train_samples"] = int(train_samples)

        if eval_dataset is not None:
            eval_samples = getattr(eval_dataset, "num_rows", None)
            if eval_samples is None and hasattr(eval_dataset, "__len__"):
                try:
                    eval_samples = len(eval_dataset)
                except TypeError:
                    eval_samples = None
            if eval_samples is not None:
                metrics["eval_samples"] = int(eval_samples)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        return metrics

    def _prepare_hf_prompt_dataset(self, *, dataset_path: Path, streaming: bool):
        """Load and preprocess a prompt dataset using Hugging Face datasets."""

        try:
            from datasets import load_dataset  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required when use_hf_datasets=True. Install `datasets`."
            ) from exc

        # Early validation: check if dataset file exists and has content
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        # Count lines in JSONL file for early validation
        line_count = 0
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    line_count += 1
        
        LOGGER.info(f"Dataset file has {line_count} raw lines")
        if line_count == 0:
            raise ValueError(f"Dataset file is empty: {dataset_path}")
        if line_count < 10:
            LOGGER.warning(
                f"Dataset file has only {line_count} lines. "
                f"This may result in insufficient training data after filtering."
            )

        data_files = {"train": str(dataset_path)}
        dataset = load_dataset(
            "json",
            data_files=data_files,
            split="train",
            streaming=streaming,
        )

        def has_valid_messages(example: Dict[str, Any]) -> bool:
            return _extract_messages(example) is not None

        dataset = dataset.filter(
            has_valid_messages,
            batched=False,
            load_from_cache_file=not streaming,
        )

        def tokenize_example(example: Dict[str, Any]) -> Dict[str, Any]:
            try:
                parsed = _extract_messages(example)
                if not parsed:
                    print("[DEBUG] Failed to extract messages from example")
                    return {
                        "skip": True,
                        "input_ids": [],
                        "labels": [],
                        "attention_mask": [],
                    }

                prompt_text, response_text = _format_prompt_and_response(self.tokenizer, parsed)
                if not prompt_text or not response_text:
                    print(f"[DEBUG] Empty prompt or response: prompt={bool(prompt_text)}, response={bool(response_text)}")
                    return {
                        "skip": True,
                        "input_ids": [],
                        "labels": [],
                        "attention_mask": [],
                    }
                      
                sample = _build_tokenized_sample(
                    self.tokenizer,
                    prompt_text,
                    response_text,
                    self.config.max_seq_length,
                )

                if sample is None:
                    print("[DEBUG] Tokenization returned None")
                    return {
                        "skip": True,
                        "input_ids": [],
                        "labels": [],
                        "attention_mask": [],
                    }

                input_ids, labels, attention_mask = sample
                return {
                    "skip": False,
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
            except Exception as tokenize_error:
                print(f"[ERROR] Tokenization failed: {type(tokenize_error).__name__}: {tokenize_error}")
                import traceback
                traceback.print_exc()
                return {
                    "skip": True,
                    "input_ids": [],
                    "labels": [],
                    "attention_mask": [],
                }

        columns_to_remove = list(getattr(dataset, "column_names", [])) or None

        print(f"[DEBUG] Starting tokenization map operation...")
        print(f"[DEBUG] Columns to remove: {columns_to_remove}")
        print(f"[DEBUG] Streaming mode: {streaming}")
        print(f"[DEBUG] Num proc: {None if (streaming or self.config.hf_num_proc is None) else self.config.hf_num_proc}")
        print(f"[DEBUG] Dataset type: {type(dataset)}")
        sys.stdout.flush()
        
        # Test tokenize on first example before running full map
        print("[DEBUG] Testing tokenize_example on first row...")
        sys.stdout.flush()
        try:
            first_example = next(iter(dataset))
            print(f"[DEBUG] First example keys: {first_example.keys() if hasattr(first_example, 'keys') else type(first_example)}")
            sys.stdout.flush()
            
            test_result = tokenize_example(first_example)
            print(f"[DEBUG] Test tokenization result: skip={test_result.get('skip')}, has_input_ids={len(test_result.get('input_ids', [])) > 0}")
            sys.stdout.flush()
        except Exception as test_error:
            print(f"[ERROR] Test tokenization failed: {test_error}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
        
        print("[DEBUG] About to call dataset.map()...")
        sys.stdout.flush()
        
        # Performance target: Process 1000+ examples in <30 seconds with multiprocessing
        print("[DEBUG] Using manual iteration instead of dataset.map() to avoid hanging...")
        sys.stdout.flush()
        
        try:
            tokenized_examples = []
            for idx, example in enumerate(dataset):
                print(f"[DEBUG] Tokenizing example {idx + 1}...")
                sys.stdout.flush()
                result = tokenize_example(example)
                if not result.get("skip", False):
                    tokenized_examples.append(result)
                    print(f"[DEBUG] Example {idx + 1} tokenized successfully (not skipped)")
                else:
                    print(f"[DEBUG] Example {idx + 1} skipped")
                sys.stdout.flush()
            
            print(f"[DEBUG] Manual tokenization complete: {len(tokenized_examples)} examples kept")
            sys.stdout.flush()
            
            if len(tokenized_examples) == 0:
                raise ValueError("All examples were skipped during tokenization")
            
            # Convert to HF Dataset
            from datasets import Dataset as HFDataset
            dataset = HFDataset.from_dict({
                "input_ids": [ex["input_ids"] for ex in tokenized_examples],
                "labels": [ex["labels"] for ex in tokenized_examples],
                "attention_mask": [ex["attention_mask"] for ex in tokenized_examples],
            })
            print(f"[DEBUG] Created new dataset with {len(dataset)} examples")
            sys.stdout.flush()
            
        except KeyboardInterrupt:
            print(f"[ERROR] Tokenization interrupted by user")
            raise
        except Exception as manual_error:
            print(f"[ERROR] Manual tokenization failed: {type(manual_error).__name__}: {manual_error}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to tokenize dataset: {manual_error}") from manual_error
        
        print(f"[DEBUG] Skipping skip filter (already done in manual iteration)...")
        sys.stdout.flush()

        print(f"[DEBUG] Converting dataset to torch format...")
        sys.stdout.flush()
        dataset = dataset.with_format(type="torch")
        print(f"[DEBUG] Dataset converted to torch format successfully")
        sys.stdout.flush()

        print(f"[DEBUG] Streaming mode: {streaming}")
        print(f"[DEBUG] About to validate dataset size...")
        
        if streaming:
            iterator = dataset.take(1)
            try:
                next(iter(iterator))
            except StopIteration as stop_error:
                raise ValueError("No valid samples were parsed from the dataset") from stop_error
        else:
            num_rows = getattr(dataset, "num_rows", 0)
            print(f"[DEBUG] Dataset has num_rows attribute: {num_rows}")
            LOGGER.info(f"Dataset preparation complete: {num_rows} samples after filtering and tokenization")
            
            if num_rows == 0:
                print(f"[ERROR] Dataset is empty!")
                raise ValueError("No valid samples were parsed from the dataset")
            if num_rows < 10:
                print(f"[ERROR] Dataset has insufficient samples: {num_rows}")
                LOGGER.error(
                    f"INSUFFICIENT TRAINING DATA: Only {num_rows} samples found. "
                    f"Minimum required is 10 samples, recommended 100+ for meaningful fine-tuning."
                )
                raise ValueError(
                    f"Insufficient training data: {num_rows} samples. "
                    f"Need at least 10 samples, recommended 100+ for effective fine-tuning. "
                    f"Please annotate more examples in the Streamlit UI before attempting to train."
                )

        return dataset

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
        if self.model is not None:
            # If adapter_path is provided, we might need to load it. 
            # For now, we assume if model is loaded, it's sufficient, 
            # or the user accepts the current state due to Singleton constraint.
            # In a full implementation, we would check if the adapter is attached.
            LOGGER.info("Model already loaded. Skipping reload.")
            self.model.eval()
            return

        self.model = self._load_model(for_training=False, adapter_path=adapter_path)
        self.model.eval()

    def generate(self, request: GenerationRequest) -> str:
        """Generate telemetry narrative using the fine-tuned model."""

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_for_inference() first.")

        self._ensure_tokenizer()

        prompt = self._format_generation_prompt(
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
            "max_length": None,
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
    def _format_generation_prompt(user_prompt: str) -> str:
        user_block = f"[USER]\n{user_prompt}\n[/USER]\n\n"
        assistant_prefix = "[ASSISTANT]\n"
        return f"{user_block}{assistant_prefix}"


__all__ = [
    "LocalLLMConfig",
    "GenerationRequest",
    "TelemetryPromptDataset",
    "LocalTelemetryLLM",
]
