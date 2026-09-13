"""High-level orchestration for serving telemetry guidance LLMs."""

from __future__ import annotations

import asyncio
import base64
import io
import shutil
import zipfile
from collections import OrderedDict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import traceback

from app.integrations.backend.schemas import ActiveModelData
from app.local_llm.local_llm import LocalLLMConfig, LocalTelemetryLLM
from app.storage.cache import model_cache_service


class TelemetryLLMOrchestrator:
	"""Coordinates loading and inference for the telemetry LLM."""

	def __init__(
		self,
		*,
		llm_config: Optional[LocalLLMConfig] = None,
		adapter_directory: Path,
	) -> None:
		self.llm_config = llm_config or LocalLLMConfig()
		self.adapter_directory = Path(adapter_directory)
		self.model_cache = model_cache_service

		self.adapter_directory.mkdir(parents=True, exist_ok=True)

		self._model_fetch_locks: Dict[str, asyncio.Event] = {}
		self._lock_creation_lock = asyncio.Lock()

	def _deserialize_llm_model(self, payload: Dict[str, Any]) -> LocalTelemetryLLM:
		encoded = payload.get("adapter_zip_base64")
		adapter_name = payload.get("adapter_directory_name") or f"adapter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

		if not encoded:
			raise ValueError("Adapter payload missing 'adapter_zip_base64'")

		target_dir = self.adapter_directory / adapter_name
		if target_dir.exists():
			shutil.rmtree(target_dir)
		target_dir.mkdir(parents=True, exist_ok=True)

		raw_bytes = base64.b64decode(encoded)
		buffer = io.BytesIO(raw_bytes)
		with zipfile.ZipFile(buffer, mode="r") as zip_file:
			zip_file.extractall(path=target_dir)

		llm = LocalTelemetryLLM(config=self.llm_config)
		llm.load_for_inference(adapter_path=target_dir)
		return llm

	# ------------------------------------------------------------------
	# Inference cache management
	# ------------------------------------------------------------------
	async def get_llm_for_inference(
		self,
		*,
		force_refresh: bool = False,
		model_subtype: str = "llm_adapter_data",
		provider: str = "local",
		model_id: Optional[str] = None,
	) -> Tuple[Optional[LocalTelemetryLLM], Optional[Dict[str, Any]]]:
		if provider == "hf_local":
			if not model_id:
				return None, {"error": "Model ID is required for Hugging Face Local provider"}
			
			try:
				# Check cache first
				cached_result = self.model_cache.get(
					model_type="hf_local",
					model_subtype=model_id,
				)
				if cached_result and not force_refresh:
					return cached_result[0], cached_result[1]

				# Load model locally
				print(f"[INFO] Loading HF model locally: {model_id}"); print(f"config: {self.llm_config}")
				config = replace(self.llm_config, base_model=model_id)
				llm = LocalTelemetryLLM(config=config)
				
				# Run in thread to avoid blocking event loop during heavy load
				await asyncio.to_thread(llm.load_for_inference, adapter_path=None)
				
				metadata = {"provider": "hf_local", "model_id": model_id}
				self.model_cache.put(
					model_type="hf_local",
					data=llm,
					metadata=metadata,
					model_subtype=model_id,
				)
				return llm, metadata
			except Exception as e:
				print(f"[ERROR] Failed to load local HF model: {e}")
				traceback.print_exc()
				return None, {"error": f"Failed to load local HF model: {e}"}

		if force_refresh:
			try:
				self.model_cache.invalidate(
					model_type=provider,
					model_subtype=model_subtype,
				)
			except Exception as invalidate_error:
				print(f"[WARNING] Failed to invalidate LLM cache entry: {invalidate_error}")

		try:
			cached_result = self.model_cache.get(
				model_type=provider,
				model_subtype=model_subtype,
			)
			if cached_result and not force_refresh:
				return cached_result[0], cached_result[1]
			
			adapter_name = self.llm_config.model.adapter or "telemetry_descriptions_v1_train_20260314_001617"
			if model_id and (self.adapter_directory / model_id).exists():
				adapter_name = model_id
				
			target_dir = self.adapter_directory / adapter_name
			
			if not target_dir.exists():
				return None, {"error": f"Adapter directory {target_dir} not found"}
				
			print(f"[INFO] Loading local adapter model for inference: {target_dir}")
			llm = LocalTelemetryLLM(config=self.llm_config)
			await asyncio.to_thread(llm.load_for_inference, adapter_path=target_dir)
			
			metadata = {"provider": provider, "model_subtype": model_subtype, "adapter": adapter_name}
			self.model_cache.put(
				model_type=provider,
				data=llm,
				metadata=metadata,
				model_subtype=model_subtype,
			)
			return llm, metadata
		except Exception as fetch_error:
			print(f"[WARNING] No active {provider} model available: {fetch_error}")
			return None, {"error": str(fetch_error)}

	# ------------------------------------------------------------------
	# AI Operations Extensions
	# ------------------------------------------------------------------

	async def generate_inference(
		self,
		provider: str,
		model_id: str,
		request_data: Any
	) -> Dict[str, Any]:
		"""Perform inference using the currently cached model."""
		llm, metadata = await self.get_llm_for_inference(
			force_refresh=False,
			model_subtype=model_id,
			provider=provider,
			model_id=model_id
		)
		if llm is None:
			return {
				"status": "error",
				"message": metadata.get("error", "Model not found or failed to load") if metadata else "Model not found",
			}
		
		try:
			if hasattr(llm, "generate"):
				# Convert to native generation request if the LocalLLM expects it
				if provider == "hf_local":
					from app.local_llm.local_llm import GenerationRequest
					if isinstance(request_data, dict):
						request_data.pop("system_prompt", None)
						req = GenerationRequest(**request_data)
					else:
						req = request_data
					result = await asyncio.to_thread(llm.generate, req)
				else:
					result = await asyncio.to_thread(llm.generate, request_data)
				return {"status": "success", "result": result}
			else:
				return {
					"status": "error",
					"message": f"Generate method missing on {provider} LLM."
				}
		except Exception as e:
			traceback.print_exc()
			return {"status": "error", "message": str(e)}

	async def terminate_llm(
		self,
		provider: str,
		model_id: str
	) -> Dict[str, Any]:
		"""Remove a model from cache and free related resources (e.g., VRAM)."""
		actual_type = provider
		
		cached_result = self.model_cache.get(model_type=actual_type, model_subtype=model_id)
		
		if cached_result:
			llm, _ = cached_result
			if hasattr(llm, "cleanup"):
				llm.cleanup()
		
		self.model_cache.invalidate(model_type=actual_type, model_subtype=model_id)
		
		if provider == "hf_local":
			import gc
			try:
				import torch
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
			except ImportError:
				pass
				
		return {
			"status": "success",
			"message": f"Terminated {provider} model: {model_id}"
		}

	async def check_progress(
		self,
		provider: str,
		model_id: str
	) -> Dict[str, Any]:
		"""Check status of the specified model."""
		actual_type = provider
		cached_result = self.model_cache.get(model_type=actual_type, model_subtype=model_id)
		
		if cached_result:
			return {
				"status": "ready",
				"provider": provider,
				"model_id": model_id,
				"message": "Model is loaded and ready for inference."
			}
		else:
			return {
				"status": "not_loaded",
				"provider": provider,
				"model_id": model_id,
				"message": "Model is not currently loaded in cache."
			}


__all__ = ["TelemetryLLMOrchestrator"]
