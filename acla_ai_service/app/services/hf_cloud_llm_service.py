"""Hugging Face Cloud LLM training service.

This module provides a training backend that offloads the training process to
Hugging Face's infrastructure (e.g., via AutoTrain or Spaces).
"""

from __future__ import annotations

import logging
import json
import csv
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

try:
    from huggingface_hub import HfApi, create_repo, upload_file
except ImportError as exc:
    raise ImportError(
        "huggingface_hub is required for HuggingFaceCloudLLM. Please install it."
    ) from exc

from app.core.config import settings
from app.services.local_llm_service import LocalLLMConfig

LOGGER = logging.getLogger(__name__)


class HuggingFaceCloudLLM:
    """Wrapper for Hugging Face Cloud training."""

    def __init__(self, config: Optional[LocalLLMConfig] = None) -> None:
        self.config = config or LocalLLMConfig()
        self.api = HfApi(token=settings.hf_api_token)
        self.username = settings.hf_username

        if not settings.hf_api_token:
            LOGGER.warning("HF_API_TOKEN not set. Cloud training will fail.")

    def upload_dataset_for_training(
        self,
        dataset_path: Path,
        output_dir: Path,
        eval_dataset_path: Optional[Path] = None,
        repo_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload dataset to Hugging Face for AutoTrain (does not trigger training)."""

        if not settings.hf_api_token:
            raise ValueError("HF_API_TOKEN is required for cloud training.")

        cleaned_dataset_path = self._clean_dataset_for_training(dataset_path)
        cleaned_eval_path = self._clean_dataset_for_training(eval_dataset_path) if eval_dataset_path else None
        
        if not repo_id:
            if not self.username:
                user_info = self.api.whoami()
                self.username = user_info["name"]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = f"acla-telemetry-training-{timestamp}"
            repo_id = f"{self.username}/{dataset_name}"

        LOGGER.info(f"Preparing to upload dataset to Hugging Face: {repo_id}")

        # 1. Create Dataset Repository
        try:
            create_repo(
                repo_id=repo_id,
                token=settings.hf_api_token,
                repo_type="dataset",
                private=True,
                exist_ok=True
            )
        except Exception as e:
            LOGGER.error(f"Failed to create HF dataset repo: {e}")
            raise

        # 2. Upload Dataset File
        try:
            upload_file(
                path_or_fileobj=str(cleaned_dataset_path),
                path_in_repo="train.csv",
                repo_id=repo_id,
                repo_type="dataset",
                token=settings.hf_api_token
            )
            if cleaned_eval_path:
                upload_file(
                    path_or_fileobj=str(cleaned_eval_path),
                    path_in_repo="eval.csv",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=settings.hf_api_token
                )
        except Exception as e:
            LOGGER.error(f"Failed to upload dataset to HF: {e}")
            raise

        LOGGER.info(f"Dataset uploaded successfully to https://huggingface.co/datasets/{repo_id}")

        # 3. Return Info (No actual training triggered)
        
        training_info = {
            "cloud_provider": "huggingface",
            "dataset_repo_id": repo_id,
            "dataset_url": f"https://huggingface.co/datasets/{repo_id}",
            "status": "dataset_uploaded",
            "instructions": "Use Hugging Face AutoTrain to train the model using this dataset.",
            "base_model": self.config.base_model
        }

        # We return a structure compatible with the orchestrator's expectations,
        # but indicating that training is offloaded.
        return {
            "train_loss": 0.0, # Placeholder
            "epoch": 0.0,
            "cloud_training_info": training_info
        }

    def _clean_dataset_for_training(self, dataset_path: Path) -> Path:
        """Create a temporary, cleaned version of the dataset for training (CSV format)."""
        cleaned_records = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    
                    # Check if 'text' field already exists
                    if "text" in record and isinstance(record["text"], str):
                        cleaned_records.append(record["text"])
                        continue

                    # Prepare fields from components
                    system_prompt = record.get("system_prompt", "")
                    prompt = record.get("prompt", "")
                    response = record.get("response", "")

                    if prompt and response:
                        # Construct 'text' column for AutoTrain SFT compatibility
                        # Using a generic format: ### System: ... ### User: ... ### Assistant: ...
                        text_parts = []
                        if system_prompt:
                            text_parts.append(f"### System:\n{system_prompt}")
                        text_parts.append(f"### User:\n{prompt}")
                        text_parts.append(f"### Assistant:\n{response}")
                        
                        full_text = "\n\n".join(text_parts)
                        cleaned_records.append(full_text)
                except (json.JSONDecodeError, KeyError):
                    LOGGER.warning(f"Skipping malformed line in {dataset_path}")

        cleaned_path = dataset_path.parent / f"{dataset_path.stem}_cleaned.csv"
        with cleaned_path.open("w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text"])
            for text in cleaned_records:
                writer.writerow([text])
        
        LOGGER.info(f"Created cleaned dataset for training at: {cleaned_path}")
        return cleaned_path

    def load_for_inference(self, adapter_path: Optional[Path] = None) -> None:
        """Not implemented for cloud trainer - inference should happen via API or local download."""
        LOGGER.warning("load_for_inference called on HuggingFaceCloudLLM - this is a training-only backend.")
        pass

    def generate(self, request: Any) -> str:
        """Generate text using Hugging Face Inference Endpoint via OpenAI client."""
        token = settings.hf_api_token
        
        if not token:
            print("[WARN] HF_API_TOKEN not found in settings. Cloud inference might fail for private models.")
        else:
            print(f"[INFO] Using HF_API_TOKEN from settings (starts with {token[:4]}...)")
        
        model_id = getattr(request, "model_id", None) or self.config.base_model
        
        # Use the specific endpoint URL provided
        base_url = "https://rw76u1tvv878sk5m.us-east-1.aws.endpoints.huggingface.cloud/v1/"
        
        try:
            from openai import OpenAI
        except ImportError:
            return "Error: openai package is not installed. Please install it with `pip install openai`."

        client = OpenAI(
            base_url=base_url,
            api_key=token
        )

        messages = []
        if hasattr(request, "system_prompt") and request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        if hasattr(request, "user_prompt") and request.user_prompt:
            messages.append({"role": "user", "content": request.user_prompt})

        try:
            print(f"[INFO] Sending request to HF Inference Endpoint: {base_url} for model {model_id}")
            
            chat_completion = client.chat.completions.create(
                model=model_id,
                messages=messages,
                stream=True,
                max_tokens=request.max_new_tokens or 256,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.95,
            )

            full_response = ""
            for message in chat_completion:
                content = message.choices[0].delta.content
                if content:
                    full_response += content
            
            return full_response

        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] HF Cloud generation failed: {error_msg}")
            return f"Error generating explanation from cloud: {e}"
