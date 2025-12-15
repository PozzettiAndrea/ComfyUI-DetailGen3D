"""
LoadDetailGen3D_TextureModel node for loading MV-Adapter pipeline.
"""

import os
import sys
import torch
from typing import Optional

# Add vendor to path
_VENDOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)


# Type identifier for the texture model
DETAILGEN3D_TEXTURE_MODEL = "DETAILGEN3D_TEXTURE_MODEL"


class LoadDetailGen3D_TextureModel:
    """
    Load MV-Adapter pipeline for texture generation.

    Downloads and initializes the SDXL base model with MV-Adapter weights
    for image-guided multi-view generation and texture baking.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("STRING", {
                    "default": "stabilityai/stable-diffusion-xl-base-1.0",
                    "tooltip": "Base SDXL model from HuggingFace"
                }),
            },
            "optional": {
                "adapter_path": ("STRING", {
                    "default": "huanngzh/mv-adapter",
                    "tooltip": "MV-Adapter weights from HuggingFace"
                }),
                "vae_model": ("STRING", {
                    "default": "",
                    "tooltip": "Optional custom VAE model path"
                }),
                "dtype": (["float16", "bfloat16", "float32"], {
                    "default": "float16",
                    "tooltip": "Model precision (float16 recommended for most GPUs)"
                }),
            }
        }

    RETURN_TYPES = (DETAILGEN3D_TEXTURE_MODEL,)
    RETURN_NAMES = ("texture_model",)
    OUTPUT_TOOLTIPS = (
        "MV-Adapter pipeline for texture generation",
    )
    FUNCTION = "load_model"
    CATEGORY = "DetailGen3D"
    DESCRIPTION = "Load MV-Adapter model for mesh texturing. Requires ~12GB VRAM."

    def load_model(
        self,
        base_model: str,
        adapter_path: str = "huanngzh/mv-adapter",
        vae_model: str = "",
        dtype: str = "float16",
    ):
        """
        Load MV-Adapter pipeline.

        Args:
            base_model: Base SDXL model path
            adapter_path: MV-Adapter weights path
            vae_model: Optional custom VAE
            dtype: Model precision

        Returns:
            Tuple of (pipeline dict,)
        """
        print(f"[DetailGen3D] Loading texture model...")
        print(f"[DetailGen3D] Base model: {base_model}")
        print(f"[DetailGen3D] Adapter: {adapter_path}")

        # Import MV-Adapter components
        try:
            from diffusers import AutoencoderKL, DDPMScheduler
            from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
            from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
            from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
        except ImportError as e:
            raise ImportError(
                f"Failed to import MV-Adapter components: {e}\n"
                "Make sure all dependencies are installed."
            )

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model_dtype = dtype_map.get(dtype, torch.float16)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[DetailGen3D] Using device: {device}, dtype: {dtype}")

        # Load VAE if provided
        pipe_kwargs = {"torch_dtype": model_dtype}
        if vae_model and vae_model.strip():
            print(f"[DetailGen3D] Loading custom VAE: {vae_model}")
            pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(
                vae_model, torch_dtype=model_dtype
            )

        # Load base pipeline
        print(f"[DetailGen3D] Loading SDXL pipeline (this may take a while)...")
        pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(base_model, **pipe_kwargs)

        # Setup scheduler
        pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=DDPMScheduler,
        )

        # Initialize MV-Adapter
        num_views = 6
        pipe.init_custom_adapter(
            num_views=num_views,
            self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
        )

        # Load adapter weights
        weight_name = "mvadapter_ig2mv_sdxl.safetensors"
        print(f"[DetailGen3D] Loading adapter weights: {weight_name}")
        pipe.load_custom_adapter(adapter_path, weight_name=weight_name)

        # Move to device
        pipe.to(device=device, dtype=model_dtype)
        if hasattr(pipe, 'cond_encoder') and pipe.cond_encoder is not None:
            pipe.cond_encoder.to(device=device, dtype=model_dtype)

        # Enable memory optimizations
        pipe.enable_vae_slicing()
        if hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()

        print(f"[DetailGen3D] Texture model loaded successfully")

        # Return as dict with pipeline and config
        model_dict = {
            "pipeline": pipe,
            "device": device,
            "dtype": model_dtype,
            "num_views": num_views,
        }

        return (model_dict,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LoadDetailGen3D_TextureModel": LoadDetailGen3D_TextureModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDetailGen3D_TextureModel": "Load DetailGen3D Texture Model",
}
