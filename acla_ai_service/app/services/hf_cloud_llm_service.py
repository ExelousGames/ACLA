"""Hugging Face Cloud LLM training service.

This module provides a training backend that offloads the training process to
Hugging Face's infrastructure (e.g., via AutoTrain or Spaces).
"""

from __future__ import annotations

import logging
import json
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
                path_in_repo="train.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                token=settings.hf_api_token
            )
            if cleaned_eval_path:
                upload_file(
                    path_or_fileobj=str(cleaned_eval_path),
                    path_in_repo="eval.jsonl",
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
        """Create a temporary, cleaned version of the dataset for training."""
        cleaned_records = []
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    
                    # Prepare fields
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

                        cleaned_record = {
                            "text": full_text,
                        }
                        cleaned_records.append(cleaned_record)
                except (json.JSONDecodeError, KeyError):
                    LOGGER.warning(f"Skipping malformed line in {dataset_path}")

        cleaned_path = dataset_path.parent / f"{dataset_path.stem}_cleaned.jsonl"
        with cleaned_path.open("w", encoding="utf-8") as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        LOGGER.info(f"Created cleaned dataset for training at: {cleaned_path}")
        return cleaned_path

    def load_for_inference(self, adapter_path: Optional[Path] = None) -> None:
        """Not implemented for cloud trainer - inference should happen via API or local download."""
        LOGGER.warning("load_for_inference called on HuggingFaceCloudLLM - this is a training-only backend.")
        pass

    def generate(self, request: Any) -> str:
        """Generate text using Hugging Face Inference API."""
        if not settings.hf_api_token:
            raise ValueError("HF_API_TOKEN is required for cloud generation.")
        
        # If we have a repo_id from initialization or config, use it.
        # Otherwise, we might need to pass it in the request or set it on the instance.
        # For now, let's assume the user provides the model ID in the request or we use a default.
        
        # We'll use the base model from config if available, or a default.
        # Ideally, the UI should allow selecting the model.
        model_id = getattr(request, "model_id", None) or self.config.base_model
        
        # If model_id is a path (local), we can't use cloud inference easily unless we deploy it.
        # But here we assume the user wants to use a HF Hub model.
        
        # Construct the payload
        # The request object is likely GenerationRequest from local_llm_service
        # which has system_prompt, user_prompt, etc.
        
        prompt = f"[SYSTEM]\n{request.system_prompt}\n[/SYSTEM]\n\n[USER]\n{request.user_prompt}\n[/USER]\n\n[ASSISTANT]\n"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": request.max_new_tokens or 256,
                "temperature": request.temperature or 0.7,
                "top_p": request.top_p or 0.95,
                "do_sample": request.do_sample if request.do_sample is not None else True,
                "return_full_text": False
            }
        }
        
        import requests
        api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = {"Authorization": f"Bearer {settings.hf_api_token}"}
        
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"]
            else:
                return str(result)
                
        except Exception as e:
            error_msg = str(e)
            if "410 Client Error: Gone" in error_msg or "404 Client Error: Not Found" in error_msg:
                 return f"Error: Model not found or not accessible ({model_id}). If you are using a dataset ID, please use the trained model ID instead."
            LOGGER.error(f"HF Cloud generation failed: {e}")
            return f"Error generating explanation from cloud: {e}"
