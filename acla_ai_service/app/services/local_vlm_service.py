"""Local VLM inference service for telemetry analysis.

This module provides functionality to convert CSV telemetry data into visual
graphs and analyze them using Vision Language Models (VLM).
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from app.core.config import settings

try:
    from transformers import (
        AutoProcessor,
        AutoModelForVision2Seq,
        BitsAndBytesConfig,
    )
except ImportError as exc:
    raise ImportError(
        "transformers is required for LocalVLMService. Please install `transformers`."
    ) from exc

LOGGER = logging.getLogger(__name__)

# Determine persistent model cache directory
_SERVICE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE = str(_SERVICE_ROOT / "models" / "huggingface_cache")


@dataclass
class LocalVLMConfig:
    """Configuration options for loading the local VLM."""

    # Default to LLaVA 1.5 7B, a strong open-source VLM
    base_model: str = "llava-hf/llava-1.5-7b-hf"
    cache_dir: Optional[str] = DEFAULT_HF_CACHE

    # Quantization settings for memory efficiency
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    
    device_map: Union[str, Dict[str, Union[int, str]]] = "auto"
    low_cpu_mem_usage: bool = True

    # Generation defaults
    max_new_tokens: int = 512
    temperature: float = 0.2
    do_sample: bool = True


class LocalVLMService:
    """Service for generating graphs from data and analyzing them with VLM."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LocalVLMService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[LocalVLMConfig] = None) -> None:
        if getattr(self, "_initialized", False):
            return

        self.config = config or LocalVLMConfig()
        self.processor = None
        self.model = None
        self._initialized = True

    def _load_model(self) -> None:
        """Load the VLM model and processor if not already loaded."""
        if self.model is not None:
            return

        LOGGER.info("Loading VLM model: %s", self.config.base_model)

        quantization_config = None
        if self.config.load_in_4bit or self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.base_model,
                cache_dir=self.config.cache_dir,
                token=settings.hf_api_token,
                trust_remote_code=True,
            )

            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.base_model,
                cache_dir=self.config.cache_dir,
                quantization_config=quantization_config,
                device_map=self.config.device_map,
                token=settings.hf_api_token,
                trust_remote_code=True,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
            )

        except Exception as e:
            LOGGER.error("Failed to load VLM model: %s", e)
            raise RuntimeError(f"Could not load VLM model {self.config.base_model}") from e

    def create_graph_from_csv(self, csv_data: str) -> Image.Image:
        """Converts CSV string data to a matplotlib plot image."""
        try:
            # Use non-interactive backend to avoid GUI issues
            plt.switch_backend("Agg")

            # Parse CSV data
            df = pd.read_csv(io.StringIO(csv_data))

            if df.empty:
                raise ValueError("CSV data is empty")

            # Create plot
            plt.figure(figsize=(10, 6))

            # Identify numeric columns to plot
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in CSV to plot")

            # Simple heuristic: look for a time-like column for x-axis
            x_col = None
            potential_x = ["time", "timestamp", "step", "index", "epoch"]
            for col in df.columns:
                if any(p in col.lower() for p in potential_x):
                    x_col = col
                    break

            if x_col:
                # Plot other numeric columns against x_col
                for col in numeric_cols:
                    if col != x_col:
                        plt.plot(df[x_col], df[col], label=col)
                plt.xlabel(x_col)
            else:
                # Fallback: plot against index
                for col in numeric_cols:
                    plt.plot(df.index, df[col], label=col)
                plt.xlabel("Index")

            plt.legend()
            plt.title("Telemetry Data Visualization")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()

            # Save plot to in-memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close()

            # Convert to PIL Image
            return Image.open(buf).convert("RGB")

        except Exception as e:
            LOGGER.error("Error generating graph from CSV: %s", e)
            raise

    def analyze_data(self, csv_data: str, prompt: str) -> str:
        """Generates a graph from CSV data and runs VLM inference on it.

        Args:
            csv_data: String content of the CSV file.
            prompt: User question or prompt about the data.

        Returns:
            The model's textual analysis of the graph and data.
        """
        self._load_model()
        
        if not prompt:
            prompt = "Describe the trends in this data."

        # 1. Generate visualization
        image = self.create_graph_from_csv(csv_data)

        # 2. Format prompt for VLM (LLaVA-style default)
        # Note: LLaVA expects <image> token in the user prompt
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"

        # 3. Process inputs
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        # 4. Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
            )

        # 5. Decode output
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # 6. Post-processing: extract assistant part
        if "ASSISTANT:" in generated_text:
            generated_text = generated_text.split("ASSISTANT:")[-1].strip()

        return generated_text


__all__ = [
    "LocalVLMConfig",
    "LocalVLMService",
]
