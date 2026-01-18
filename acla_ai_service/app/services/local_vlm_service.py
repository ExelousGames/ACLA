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
from typing import Dict, Optional, Union, Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from app.core.config import settings

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        TextIteratorStreamer,
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


@dataclass
class LocalVLMConfig:
    """Configuration options for loading the local VLM."""

    # Default to InternVL3-14B
    base_model: str = "OpenGVLab/InternVL3_5-30B-A3B-Flash"
    cache_dir: Optional[str] = DEFAULT_HF_CACHE

    # Quantization settings for memory efficiency
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    
    device_map: Union[str, Dict[str, Union[int, str]]] = "auto"
    low_cpu_mem_usage: bool = True

    # Generation defaults
    max_new_tokens: int = 4096
    temperature: float = 0.4
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
        self.tokenizer = None
        self.model = None
        self._initialized = True

    def _load_model(self) -> None:
        """Load the VLM model and tokenizer if not already loaded."""
        if self.model is not None:
            return

        LOGGER.info("Loading VLM model: %s", self.config.base_model)

        # InternVL3-14B specific loading logic
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            "trust_remote_code": True,
            "cache_dir": self.config.cache_dir,
            "token": settings.hf_api_token,
        }

        # Handle quantization if specified
        skip_modules = ["pooling_before_gating"]
        if self.config.load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=skip_modules
            )
        elif self.config.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=skip_modules
            )
        else:
            # If not quantized, use flash attention if available
            try:
                import flash_attn
                kwargs["use_flash_attn"] = True
            except ImportError:
                kwargs["use_flash_attn"] = False

        if self.config.device_map is not None:
             kwargs["device_map"] = self.config.device_map

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model,
                trust_remote_code=True,
                use_fast=False,
                cache_dir=self.config.cache_dir,
                token=settings.hf_api_token,
            )

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.model = AutoModel.from_pretrained(
                self.config.base_model,
                **kwargs
            ).eval()
            
            # If not using device_map (accelerate), move to cuda explicitly if available and not quantized
            if not self.config.load_in_8bit and not self.config.load_in_4bit and self.config.device_map is None:
                if torch.cuda.is_available():
                    self.model = self.model.cuda()

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

    def create_trajectory_graph_from_csv(self, csv_data: str) -> Image.Image:
        """Converts CSV string data to a 2D trajectory plot image."""
        try:
            # Use non-interactive backend
            plt.switch_backend("Agg")

            df = pd.read_csv(io.StringIO(csv_data))
            if df.empty:
                 # If empty, return a blank white image or raise
                 # Returning a small blank image is safer for optional components
                 return Image.new('RGB', (100, 100), color='white')

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            # Player
            has_player = False
            if 'Graphics_player_pos_x' in df.columns and 'Graphics_player_pos_y' in df.columns:
                has_player = True
                xs = df['Graphics_player_pos_x']
                ys = df['Graphics_player_pos_y']
                ax.plot(xs, ys, label='Player', color='blue')
                if len(xs) > 0:
                    ax.scatter(xs.iloc[0], ys.iloc[0], color='green', marker='o', label='Start')
                    ax.scatter(xs.iloc[-1], ys.iloc[-1], color='red', marker='x', label='End')

            # Expert
            if 'expert_optimal_player_pos_x' in df.columns and 'expert_optimal_player_pos_y' in df.columns:
                exs = df['expert_optimal_player_pos_x']
                eys = df['expert_optimal_player_pos_y']
                ax.plot(exs, eys, label='Expert', color='orange', linestyle='--')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title("2D Trajectory")
            ax.legend()
            ax.grid(True)
            ax.axis('equal')

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close()
            return Image.open(buf).convert("RGB")
        except Exception as e:
            LOGGER.error("Error generating trajectory graph: %s", e)
            # Create a fallback image with error text or just plain
            return Image.new('RGB', (200, 100), color='gray')

    def analyze_data(
        self, 
        csv_data: str, 
        prompt: str, 
        trajectory_csv_data: Optional[str] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> tuple[str, Image.Image]:
        """Generates a graph from CSV data and runs VLM inference on it.

        Args:
            csv_data: String content of the CSV file for telemetry traces.
            prompt: User question or prompt about the data.
            trajectory_csv_data: Optional string content of CSV for trajectory data.
            status_callback: Optional callback to report progress.

        Returns:
            A tuple containing:
            - The model's textual analysis of the graph and data.
            - The generated graph image (combined if trajectory is included).
        """
        if status_callback:
            status_callback("Checking model status...")
        self._load_model()
        
        if not prompt:
            prompt = "Describe the trends in this data."

        # 1. Generate visualization(s)
        if status_callback:
            status_callback("Generating telemetry visualization...")
        image_telem = self.create_graph_from_csv(csv_data)
        
        # Prepare inputs
        if status_callback:
            status_callback("Preprocessing telemetry image...")
        pixel_values1 = load_image(image_telem, max_num=12).to(torch.bfloat16)
        if torch.cuda.is_available():
            pixel_values1 = pixel_values1.cuda()
            
        final_image = image_telem
        pixel_values = pixel_values1
        num_patches_list = [pixel_values1.size(0)]
        
        question = f'<image>\n{prompt}'

        if trajectory_csv_data:
            if status_callback:
                status_callback("Generating and preprocessing trajectory visualization...")
            image_traj = self.create_trajectory_graph_from_csv(trajectory_csv_data)
            
            # Prepare second image inputs
            pixel_values2 = load_image(image_traj, max_num=12).to(torch.bfloat16)
            if torch.cuda.is_available():
                pixel_values2 = pixel_values2.cuda()
            
            # Concatenate pixel values for multi-image input
            pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
            num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
            
            # Update prompt to reference both images
            question = f'Image-1: <image>\nImage-2: <image>\nImage-1 shows the telemetry data traces. Image-2 shows the 2D trajectory of the vehicle (Top-down view).\n{prompt}'
            w_t, h_t = image_telem.size
            w_tr, h_tr = image_traj.size
            max_w = max(w_t, w_tr)
            total_h = h_t + h_tr
            
            combined = Image.new('RGB', (max_w, total_h), (255, 255, 255))
            combined.paste(image_telem, ((max_w - w_t) // 2, 0))
            combined.paste(image_traj, ((max_w - w_tr) // 2, h_t))
            final_image = combined

        generation_config = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.2,
        )

        # 4. Generate response using chat method with streaming if callback provided
        if status_callback:
            status_callback("Running VLM inference...")
            
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_config['streamer'] = streamer
            
            thread = threading.Thread(target=self.model.chat, kwargs={
                'tokenizer': self.tokenizer,
                'pixel_values': pixel_values,
                'question': question,
                'generation_config': generation_config,
                'num_patches_list': num_patches_list,
                'history': None,
                'return_history': False
            })
            thread.start()
            
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                status_callback(f"Generating: {generated_text}")
                
            thread.join()
        else:
            generated_text = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                question, 
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False
            )

        return generated_text, final_image
