"""Utilities to generate images with the Qwen-Image diffusion pipeline.

This helper wraps a preconfigured Diffusers pipeline (Qwen/Qwen-Image) with
reasonable defaults for scheduler, LoRA acceleration, and compilation. It
exposes a simple callable class that returns a PIL.Image for a text prompt.

Notes
-----
- The pipeline is moved to CUDA and compiled for speed. A CUDA-capable GPU and
  the corresponding PyTorch build are required for execution.
- LoRA weights from "lightx2v/Qwen-Image-Lightning" are loaded and fused to
  reduce inference steps (8 by default) while maintaining quality.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import math
import random

import numpy as np
import torch
from PIL import Image
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

MAX_SEED = np.iinfo(np.int32).max

# Common width x height presets keyed by aspect ratio label.
ASPECT_RATIO_SIZES: Mapping[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1152, 640),
    "9:16": (640, 1152),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "3:2": (1024, 688),
    "2:3": (688, 1024),
}


def get_random_aspect_ratio() -> str:
    """Return a random aspect ratio key from ASPECT_RATIO_SIZES.

    This is used when no aspect ratio is provided to the generator.
    """
    return random.choice(list(ASPECT_RATIO_SIZES.keys()))


class QwenImage:
    """Text-to-image generator backed by the Qwen-Image Diffusers pipeline.

    Parameters
    ----------
    model_path:
        Hugging Face model identifier or local path. Defaults to
        "Qwen/Qwen-Image" when None.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        model_path = model_path or "Qwen/Qwen-Image"

        # FlowMatch Euler discrete scheduler settings tuned for Qwen-Image.
        # See: https://huggingface.co/docs/diffusers
        scheduler_config: Dict[str, Any] = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        self.pipe = DiffusionPipeline.from_pretrained(
            model_path, scheduler=scheduler, torch_dtype=torch.bfloat16
        ).to("cuda")

        # Load LoRA weights for acceleration.
        self.pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning",
            weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors",
        )
        self.pipe.fuse_lora()
        self.pipe.enable_xformers_memory_efficient_attention()

        # Compile transformer blocks for performance (requires PyTorch 2.x).
        self.pipe.transformer = torch.compile(
            self.pipe.transformer, mode="default", fullgraph=True, dynamic=True
        )

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = " ",
        aspect_ratio: Optional[str] = None,
        num_inference_steps: int = 8,
        seed: int = 0,
        randomize_seed: bool = True,
        guidance_scale: float = 1.0,
    ) -> Image.Image:
        """Generate an image from a text prompt.

        Parameters
        ----------
        prompt:
            Positive text prompt describing the desired image.
        negative_prompt:
            Negative prompt to discourage undesired attributes.
        aspect_ratio:
            One of the keys in ASPECT_RATIO_SIZES (e.g., "16:9"). If not
            provided, a random supported ratio is chosen.
        num_inference_steps:
            Number of denoising steps. Lower values are faster with the loaded
            Lightning LoRA (defaults to 8).
        seed:
            Seed for reproducibility. Ignored if randomize_seed is True.
        randomize_seed:
            When True, a random seed in [0, MAX_SEED] is used.
        guidance_scale:
            Guidance strength; this model uses true_cfg_scale.

        Returns
        -------
        PIL.Image.Image
            The generated image.
        """
        # Resolve target image size from aspect ratio preset. Fallback to 1:1 if
        # an unsupported ratio is provided.
        if aspect_ratio:
            width, height = ASPECT_RATIO_SIZES.get(aspect_ratio, (1024, 1024))
        else:
            aspect_ratio = get_random_aspect_ratio()
            width, height = ASPECT_RATIO_SIZES[aspect_ratio]

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        # Create a CUDA RNG for deterministic sampling when a seed is set.
        generator = torch.Generator(device="cuda").manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=guidance_scale,  # Use true_cfg_scale for this model
        ).images[0]

        return image
