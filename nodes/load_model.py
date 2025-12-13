"""
LoadDetailGen3DModel node for loading DetailGen3D inference pipeline.
"""

import os
import torch
from pathlib import Path
from typing import Any

from .utils import (
    _MODEL_CACHE,
    get_detailgen3d_models_path,
    get_device,
    get_dtype,
    DETAILGEN3D_MODEL,
)


class LoadDetailGen3DModel:
    """
    Load DetailGen3D model for enhancing 3D meshes with geometric detail.

    Downloads model from HuggingFace if not cached locally.
    Models are cached globally to avoid reloading.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_source": (["hf"], {
                    "default": "hf",
                    "tooltip": "Model source (hf = VAST-AI/DetailGen3D from HuggingFace)"
                }),
                "dtype": (["float16", "float32", "auto"], {
                    "default": "float16",
                    "tooltip": "Model precision: float16 (recommended), float32 (slowest), auto (detect based on GPU)"
                }),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HuggingFace token for private/gated repositories (leave empty for public models)"
                }),
            }
        }

    RETURN_TYPES = (DETAILGEN3D_MODEL, DETAILGEN3D_MODEL, DETAILGEN3D_MODEL)
    RETURN_NAMES = ("model", "encoder", "decoder")
    OUTPUT_TOOLTIPS = (
        "DetailGen3D pipeline - use with DetailGen3D_Generate",
        "VAE encoder - use with DetailGen3D_PrepareMesh",
        "Decoder reference - used internally"
    )
    FUNCTION = "load_model"
    CATEGORY = "DetailGen3D"
    DESCRIPTION = "Load DetailGen3D model for enhancing 3D meshes with geometric detail."

    def load_model(self, model_source: str, dtype: str, hf_token: str = ""):
        """
        Load the DetailGen3D model.

        Args:
            model_source: Model source to load from
            dtype: Model precision
            hf_token: HuggingFace token for private repos (optional)

        Returns:
            Tuple of (pipeline, encoder, decoder) - all point to same pipeline
        """
        print(f"[DetailGen3D] Loading DetailGen3D model...")

        # Check CUDA availability
        device = get_device()
        torch_dtype = get_dtype(dtype)

        if device.type == "cpu":
            print("[DetailGen3D] WARNING: CUDA not available, running on CPU will be extremely slow!")
        else:
            gpu_props = torch.cuda.get_device_properties(0)
            vram_gb = gpu_props.total_memory / (1024**3)
            print(f"[DetailGen3D] Using GPU: {gpu_props.name} ({vram_gb:.1f} GB VRAM)")

        # Create cache key
        cache_key = f"{model_source}_{dtype}"

        # Return cached model if available
        if cache_key in _MODEL_CACHE:
            print(f"[DetailGen3D] Using cached model")
            pipeline = _MODEL_CACHE[cache_key]
            return (pipeline, pipeline, pipeline)

        # Get or download checkpoint
        checkpoint_path = self._get_or_download_checkpoint(model_source, hf_token)

        # Import pipeline from vendored library
        try:
            from detailgen3d.pipelines.pipeline_detailgen3d import DetailGen3DPipeline
        except ImportError as e:
            raise ImportError(
                f"Failed to import DetailGen3DPipeline: {e}\n"
                "Please ensure the vendored library is properly installed."
            ) from e

        # Load the pipeline
        try:
            print(f"[DetailGen3D] Loading pipeline from {checkpoint_path}...")
            pipeline = DetailGen3DPipeline.from_pretrained(
                str(checkpoint_path)
            ).to(device, dtype=torch_dtype)

            print(f"[DetailGen3D] Pipeline loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load DetailGen3D pipeline: {e}") from e

        # Cache the pipeline
        _MODEL_CACHE[cache_key] = pipeline
        print(f"[DetailGen3D] Model loaded and cached")

        # Return same pipeline 3 times (for different stages)
        return (pipeline, pipeline, pipeline)

    @classmethod
    def _get_or_download_checkpoint(cls, model_source: str, hf_token: str = "") -> Path:
        """
        Get checkpoint path, downloading if necessary.

        Args:
            model_source: Model source tag
            hf_token: HuggingFace token for authentication (optional)

        Returns:
            Path to checkpoint directory
        """
        models_dir = get_detailgen3d_models_path()
        checkpoint_dir = models_dir / model_source

        # Check if checkpoint already exists
        # Look for model_index.json which is the standard diffusers marker
        if checkpoint_dir.exists():
            if (checkpoint_dir / "model_index.json").exists():
                print(f"[DetailGen3D] Using cached model at {checkpoint_dir}")
                return checkpoint_dir

        # Download checkpoint
        print(f"[DetailGen3D] Downloading model '{model_source}'...")

        try:
            cls._download_checkpoint(model_source, checkpoint_dir, hf_token)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download checkpoint: {e}\n"
                "Please check your internet connection and try again."
            ) from e

        return checkpoint_dir

    @classmethod
    def _download_checkpoint(cls, model_source: str, target_dir: Path, hf_token: str = ""):
        """
        Download checkpoint from HuggingFace.

        Args:
            model_source: Model source tag
            target_dir: Target directory for download
            hf_token: HuggingFace token for authentication (optional)
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        # Map model sources to HuggingFace repo IDs
        repo_mapping = {
            "hf": "VAST-AI/DetailGen3D",
        }

        if model_source not in repo_mapping:
            raise ValueError(f"Unknown model source: {model_source}")

        repo_id = repo_mapping[model_source]

        try:
            from huggingface_hub import snapshot_download

            print(f"[DetailGen3D] Downloading from HuggingFace: {repo_id} (this may take a while)")

            # Download all files from the repo
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                token=hf_token or None,
            )

            print(f"[DetailGen3D] Download complete: {target_dir}")

        except ImportError:
            raise ImportError(
                "huggingface_hub is required for downloading checkpoints. "
                "Please install it: pip install huggingface-hub"
            )
        except Exception as e:
            # Clean up partial download
            import shutil
            if target_dir.exists():
                shutil.rmtree(target_dir)
            raise RuntimeError(f"Download failed: {e}") from e


# Node mappings
NODE_CLASS_MAPPINGS = {
    "LoadDetailGen3DModel": LoadDetailGen3DModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDetailGen3DModel": "Load DetailGen3D Model",
}
