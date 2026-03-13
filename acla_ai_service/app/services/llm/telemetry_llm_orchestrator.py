"""High-level orchestration for training and serving telemetry guidance LLMs."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import zipfile
from collections import OrderedDict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import traceback

from app.models import ActiveModelData

from ..backend_service import backend_service
from .local_llm_service import LocalLLMConfig, LocalTelemetryLLM
from ..model_cache_service import model_cache_service
from ..zarr_telemetry_store import get_shared_zarr_store


class TelemetryLLMOrchestrator:
	"""Coordinates fine-tuning and inference for the telemetry LLM."""

	def __init__(
		self,
		*,
		llm_config: Optional[LocalLLMConfig] = None,
		adapter_directory: Path,
		dataset_directory: Path,
	) -> None:
		self.llm_config = llm_config or LocalLLMConfig()
		self.adapter_directory = Path(adapter_directory)
		self.dataset_directory = Path(dataset_directory)
		self.backend_service = backend_service
		self.model_cache = model_cache_service
		self.data_cache = get_shared_zarr_store()

		self.adapter_directory.mkdir(parents=True, exist_ok=True)
		self.dataset_directory.mkdir(parents=True, exist_ok=True)

		self._model_fetch_locks: Dict[str, asyncio.Event] = {}
		self._lock_creation_lock = asyncio.Lock()

	def _initialize_dataset_summary(self, dataset_path: Path) -> Dict[str, Any]:
		return {
			"dataset_path": str(dataset_path),
			"total_examples": 0,
			"annotated_examples": 0,
			"annotation_ratio": 0.0,
		}

	def _update_summary_from_line(
		self,
		line: str,
		summary: Dict[str, Any],
	) -> None:
		summary["total_examples"] += 1
		try:
			record = json.loads(line)
		except json.JSONDecodeError:
			return

		metadata = record.get("metadata") or {}
		if metadata.get("annotation_complete"):
			summary["annotated_examples"] += 1

	def _finalize_summary(self, summary: Dict[str, Any]) -> None:
		total = summary.get("total_examples", 0)
		annotated = summary.get("annotated_examples", 0)
		summary["annotation_ratio"] = (annotated / total) if total else 0.0

	def _summarize_dataset(self, dataset_path: Path) -> Dict[str, Any]:
		"""Collect lightweight statistics for an existing dataset file."""

		summary = self._initialize_dataset_summary(dataset_path)

		if not dataset_path.exists():
			return summary

		try:
			with dataset_path.open("r", encoding="utf-8") as jsonl_file:
				for raw_line in jsonl_file:
					line = raw_line.strip()
					if not line:
						continue
					self._update_summary_from_line(line, summary)
		except Exception as error:  # pragma: no cover - logging safety
			print(f"[WARNING] Failed to summarize dataset {dataset_path}: {error}")
			return summary

		self._finalize_summary(summary)
		return summary

	# ------------------------------------------------------------------
	# Training helpers
	# ------------------------------------------------------------------
	async def train_from_dataset(
		self,
		*,
		dataset_path: Path,
		dataset_stats: Optional[Dict[str, Any]] = None,
		cleanup_dataset_file: bool = False,
	) -> Dict[str, Any]:
		dataset_path = Path(dataset_path)
		dataset_stats = dataset_stats or self._summarize_dataset(dataset_path)

		training_artifacts = await self._train_llm(dataset_path=dataset_path)

		serialized_adapter = training_artifacts["serialized_adapter"]
		adapter_dir: Path = training_artifacts["adapter_dir"]
		metrics = training_artifacts["metrics"]

		metadata_payload = {
			"training_metrics": metrics,
			"dataset_stats": dataset_stats,
			"adapter_directory": adapter_dir.name,
			"llm_config": training_artifacts["config"],
			"generated_dataset": str(dataset_path),
			"training_timestamp": datetime.now().isoformat(),
		}

		print(
			f"[INFO] LLM fine-tuning complete. Saving adapter '{adapter_dir.name}' to backend"
		)

		# await self.backend_service.save_ai_model(
		# 	model_type="llm_guidance_v1",
		# 	model_data=serialized_adapter,
		# 	metadata=metadata_payload,
		# 	is_active=True,
		# )
		print("[WARN] Model saving temporarily disabled by user request.")

		if cleanup_dataset_file:
			try:
				dataset_path.unlink(missing_ok=True)
			except Exception as cleanup_error:
				print(f"[WARNING] Failed to delete dataset file {dataset_path}: {cleanup_error}")

		return {
			"success": True,
			"training_metrics": metrics,
			"adapter_directory": adapter_dir.name,
			"serialized_adapter": serialized_adapter,
		}

	async def _train_llm(self, dataset_path: Path) -> Dict[str, Any]:
		return await asyncio.to_thread(self._train_local_llm_sync, Path(dataset_path))

	def _train_local_llm_sync(self, dataset_path: Path) -> Dict[str, Any]:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		dataset_identifier = (dataset_path.stem or "telemetry").replace(" ", "_")
		adapter_dir = self.adapter_directory / f"{dataset_identifier}_{timestamp}"
		adapter_dir.mkdir(parents=True, exist_ok=True)

		print(f"[INFO] Initializing LocalTelemetryLLM for training...")
		print(f"[INFO] Dataset: {dataset_path}")
		print(f"[INFO] Output directory: {adapter_dir}")
		
		try:
			llm = LocalTelemetryLLM(config=self.llm_config)
			print(f"[INFO] Starting LLM training (this may take a while)...")
			metrics = llm.train(
				dataset_path=dataset_path,
				output_dir=adapter_dir,
				eval_dataset_path=None,
			)
			print(f"[INFO] Training completed successfully")
		except ValueError as ve:
			print(f"[ERROR] LLM training validation error: {ve}")
			raise
		except Exception as e:
			print(f"[ERROR] LLM training failed: {type(e).__name__}: {e}")
			traceback.print_exc()
			raise

		serialized_adapter = self._serialize_adapter_directory(adapter_dir)
		serialized_adapter["adapter_directory_name"] = adapter_dir.name

		return {
			"metrics": metrics,
			"adapter_dir": adapter_dir,
			"serialized_adapter": serialized_adapter,
			"config": asdict(self.llm_config),
		}

	def _serialize_adapter_directory(self, adapter_dir: Path) -> Dict[str, Any]:
		buffer = io.BytesIO()
		with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
			for file_path in adapter_dir.rglob("*"):
				if file_path.is_file():
					zip_file.write(file_path, arcname=file_path.relative_to(adapter_dir))

		buffer.seek(0)
		encoded = base64.b64encode(buffer.read()).decode("utf-8")

		return {
			"adapter_zip_base64": encoded,
			"created_at": datetime.now().isoformat(),
		}

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
					model_type="llm_guidance_v1",
					model_subtype=model_subtype,
				)
			except Exception as invalidate_error:
				print(f"[WARNING] Failed to invalidate LLM cache entry: {invalidate_error}")

		try:
			return await self._get_cached_model_or_fetch(
				model_type="llm_guidance_v1",
				model_subtype=model_subtype,
				deserializer_func=self._deserialize_llm_model,
			)
		except Exception as fetch_error:
			print(f"[WARNING] No active LLM guidance model available: {fetch_error}")
			return None, {"error": str(fetch_error)}

	async def _fetch_and_cache_model(
		self,
		model_type: str,
		*,
		model_subtype: str = "complete_model_data",
		deserializer_func=None,
	) -> Tuple[Any, Dict[str, Any]]:
		print(f"[DEBUG] Fetching model from backend: {model_type}")

		model_response: ActiveModelData = await self.backend_service.getCompleteActiveModelData(
			model_type
		)

		cache_metadata = {
			"model_type": model_response.modelType,
			"is_active": model_response.isActive,
			"fetched_at": datetime.now().isoformat(),
			"backend_metadata": model_response.metadata,
			"model_subtype": model_subtype,
		}

		if deserializer_func is None:
			raise ValueError("deserializer_func is required to deserialize model data")

		print(f"[DEBUG] Deserializing model instance: {model_type}")
		model_instance = deserializer_func(model_response.modelData)
		if model_instance is None:
			raise RuntimeError("Deserializer function returned None - must return model instance")

		print(f"[DEBUG] Caching model instance: {model_type}")
		self.model_cache.put(
			model_type=model_type,
			data=model_instance,
			metadata=cache_metadata,
			model_subtype=model_subtype,
		)

		print(f"[DEBUG] Successfully cached model instance: {model_type}")
		return model_instance, cache_metadata

	async def _get_cached_model_or_fetch(
		self,
		model_type: str,
		*,
		model_subtype: str = "complete_model_data",
		deserializer_func=None,
	) -> Tuple[Any, Dict[str, Any]]:
		cache_key = self.model_cache._generate_cache_key(  # type: ignore[attr-defined]
			model_type=model_type,
			model_subtype=model_subtype,
		)

		is_fetching_thread = False
		model_instance = None
		metadata: Dict[str, Any] = {}

		try:
			cache_info = self.model_cache.get_cache_info(
				model_type=model_type,
				model_subtype=model_subtype,
			)

			if cache_info and not cache_info.get("is_expired", True):
				print(
					f"[DEBUG] Cache hit for {cache_key} - age: {cache_info.get('ttl_remaining_seconds', 'N/A')}s remaining"
				)
				cached_result = self.model_cache.get(
					model_type=model_type,
					model_subtype=model_subtype,
				)
				if cached_result:
					model_instance, metadata = cached_result
					return model_instance, metadata
			else:
				print(f"[DEBUG] Cache miss or expired for {cache_key}")

			async with self._lock_creation_lock:
				cached_result = self.model_cache.get(
					model_type=model_type,
					model_subtype=model_subtype,
				)
				if cached_result:
					model_instance, metadata = cached_result
					return model_instance, metadata

				if cache_key in self._model_fetch_locks:
					fetch_event = self._model_fetch_locks[cache_key]
				else:
					fetch_event = asyncio.Event()
					self._model_fetch_locks[cache_key] = fetch_event
					is_fetching_thread = True

			if not model_instance and not is_fetching_thread:
				fetch_event = self._model_fetch_locks.get(cache_key)
				if fetch_event:
					print(f"[DEBUG] Waiting for fetch lock release: {cache_key}")
					await fetch_event.wait()
					cached_result = self.model_cache.get(
						model_type=model_type,
						model_subtype=model_subtype,
					)
					if cached_result:
						model_instance, metadata = cached_result
						return model_instance, metadata
					raise RuntimeError(f"Cache still empty after waiting for {cache_key}")

			if not model_instance and is_fetching_thread:
				try:
					model_instance, metadata = await self._fetch_and_cache_model(
						model_type,
						model_subtype=model_subtype,
						deserializer_func=deserializer_func,
					)
				except Exception as fetch_error:
					self._emergency_cleanup_fetch_lock(cache_key)
					raise fetch_error
				finally:
					self._cleanup_fetch_lock(cache_key)

			if model_instance is None:
				raise RuntimeError("Failed to obtain model instance from cache or backend")

			return model_instance, metadata

		except Exception as error:
			if is_fetching_thread:
				self._emergency_cleanup_fetch_lock(cache_key)
			raise error

	def _cleanup_fetch_lock(self, cache_key: str) -> None:
		if cache_key in self._model_fetch_locks:
			try:
				self._model_fetch_locks[cache_key].set()
				del self._model_fetch_locks[cache_key]
				print(f"[DEBUG] Released fetch lock for {cache_key}")
			except Exception as cleanup_error:
				print(f"[WARNING] Error cleaning up fetch lock: {cleanup_error}")

	def _emergency_cleanup_fetch_lock(self, cache_key: str) -> None:
		if cache_key in self._model_fetch_locks:
			try:
				self._model_fetch_locks[cache_key].set()
				del self._model_fetch_locks[cache_key]
				print(f"[INFO] Emergency cleanup of fetch lock for {cache_key}")
			except Exception as cleanup_error:
				print(f"[WARNING] Error during emergency lock cleanup: {cleanup_error}")

	# ------------------------------------------------------------------
	# Operational helpers
	# ------------------------------------------------------------------
	def clear_llm_cache(self) -> None:
		self.model_cache.invalidate(model_type="llm_guidance_v1")

	def get_fetch_locks_status(self) -> Dict[str, Any]:
		return {
			"active_locks": list(self._model_fetch_locks.keys()),
			"lock_count": len(self._model_fetch_locks),
			"timestamp": datetime.now().isoformat(),
		}

	def get_cache_debug_info(self) -> Dict[str, Any]:
		cache_stats = self.model_cache.get_stats()
		lock_status = self.get_fetch_locks_status()
		return {
			"cache_stats": cache_stats,
			"fetch_locks": lock_status,
			"timestamp": datetime.now().isoformat(),
		}

	def print_cache_debug_info(self) -> None:
		debug_info = self.get_cache_debug_info()

		print("\n" + "=" * 60)
		print("CACHE DEBUG INFORMATION")
		print("=" * 60)

		cache_stats = debug_info["cache_stats"]
		print(f"Cache Size: {cache_stats['cache_size']}/{cache_stats['max_cache_size']}")
		print(f"Memory Usage: {cache_stats['memory_usage_mb']:.2f}/{cache_stats['max_memory_mb']} MB")
		print(f"Hit Rate: {cache_stats['hit_rate']:.2%}")
		print(f"Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
		print(f"Evictions: {cache_stats['evictions']}, Cleanups: {cache_stats['cleanups']}")

		print(f"\nActive Fetch Locks: {debug_info['fetch_locks']['lock_count']}")
		for lock_key in debug_info['fetch_locks']['active_locks']:
			print(f"  - {lock_key}")

		print("\nCached Models:")
		for entry in cache_stats.get('entries', []):
			print(f"  - {entry['key']} ({entry['size_mb']:.2f} MB, accessed {entry['access_count']} times)")
			if entry.get('ttl_remaining_seconds'):
				print(f"    TTL remaining: {entry['ttl_remaining_seconds']:.0f}s")

		print("=" * 60 + "\n")

	async def clear_stuck_fetch_locks(self, max_age_minutes: int = 10) -> Dict[str, Any]:
		cleared_locks: List[str] = []
		try:
			async with self._lock_creation_lock:
				for cache_key in list(self._model_fetch_locks.keys()):
					self._model_fetch_locks[cache_key].set()
					cleared_locks.append(cache_key)
				self._model_fetch_locks.clear()

			return {
				"success": True,
				"cleared_locks": cleared_locks,
				"cleared_count": len(cleared_locks),
				"timestamp": datetime.now().isoformat(),
			}

		except Exception as error:
			return {
				"success": False,
				"error": str(error),
				"cleared_locks": cleared_locks,
				"cleared_count": len(cleared_locks),
				"timestamp": datetime.now().isoformat(),
			}

	# ------------------------------------------------------------------
	# AI Operations Extensions
	# ------------------------------------------------------------------
	async def initialize_and_persist_model(
		self,
		provider: str,
		model_id: str
	) -> Dict[str, Any]:
		"""Explicitly load a model and persist it in cache for inference."""
		llm, metadata = await self.get_llm_for_inference(
			force_refresh=True,
			model_subtype=model_id,
			provider=provider,
			model_id=model_id
		)
		if llm is None:
			return {
				"status": "error",
				"message": metadata.get("error", "Unknown initialization error") if metadata else "Unknown initialization error",
			}
		return {
			"status": "success",
			"message": f"{provider} model {model_id} loaded.",
			"metadata": metadata,
		}

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
					from .local_llm_service import GenerationRequest
					if isinstance(request_data, dict):
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
		actual_type = provider if provider == "hf_local" else "llm_guidance_v1"
		
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
		actual_type = provider if provider == "hf_local" else "llm_guidance_v1"
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
