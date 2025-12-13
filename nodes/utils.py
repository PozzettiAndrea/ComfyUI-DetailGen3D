"""
Utility functions and type definitions for ComfyUI-DetailGen3D
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional

# Add vendor directory to path for detailgen3d imports
_VENDOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)

# Custom ComfyUI types
DETAILGEN3D_MODEL = "DETAILGEN3D_MODEL"
DETAILGEN3D_LATENT = "DETAILGEN3D_LATENT"
DETAILGEN3D_SDF = "DETAILGEN3D_SDF"

# Model cache to avoid reloading
_MODEL_CACHE = {}


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    # NOTE: torch-cluster (FPS) doesn't support bfloat16, so we default to float16
    if dtype_str == "auto":
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        # bfloat16 disabled - torch-cluster FPS doesn't support it
        # "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float16)


def get_detailgen3d_models_path() -> Path:
    """Get the path to store DetailGen3D models."""
    try:
        import folder_paths
        models_dir = Path(folder_paths.models_dir)
    except (ImportError, AttributeError):
        # Fallback if folder_paths not available
        models_dir = Path(__file__).parent.parent.parent.parent / "models"

    detailgen3d_dir = models_dir / "detailgen3d"
    detailgen3d_dir.mkdir(parents=True, exist_ok=True)
    return detailgen3d_dir


def comfy_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI image tensor to PIL Image.

    ComfyUI format: [B, H, W, C] float32 [0, 1]
    PIL format: (H, W, C) uint8 [0, 255]
    """
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]  # Take first image in batch

    # Convert to numpy and scale to 0-255
    image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(image_np, mode="RGB")


def pil_to_comfy_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image to ComfyUI image tensor.

    PIL format: (H, W, C) uint8 [0, 255]
    ComfyUI format: [B, H, W, C] float32 [0, 1]
    """
    # Ensure RGB
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Convert to numpy array
    image_np = np.array(pil_image).astype(np.float32) / 255.0

    # Add batch dimension
    image_tensor = torch.from_numpy(image_np)[None, ...]

    return image_tensor


def normalize_mesh(mesh, target_scale: float = 1.9):
    """
    Center and normalize a trimesh object to fit within target bounds.

    Args:
        mesh: trimesh.Trimesh object
        target_scale: Scale to fit mesh within [-scale, scale] range

    Returns:
        Normalized copy of the mesh
    """
    import trimesh

    # Create a copy to avoid modifying the original
    normalized = mesh.copy()

    # Center at origin
    center = normalized.bounding_box.centroid
    normalized.apply_translation(-center)

    # Scale to fit in target range
    scale = max(normalized.bounding_box.extents)
    if scale > 0:
        normalized.apply_scale(target_scale / scale)

    return normalized


def sample_surface_points(mesh, num_points: int = 20480) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points and normals from mesh surface.

    Args:
        mesh: trimesh.Trimesh object
        num_points: Number of points to sample

    Returns:
        Tuple of (points [N, 3], normals [N, 3])
    """
    import trimesh

    # Sample many points to ensure we can subsample
    sample_count = max(num_points * 2, 1000000)
    surface, face_indices = trimesh.sample.sample_surface(mesh, sample_count)
    normals = mesh.face_normals[face_indices]

    # Randomly subsample to exact count
    rng = np.random.default_rng()
    indices = rng.choice(surface.shape[0], num_points, replace=False)

    return surface[indices], normals[indices]


def points_to_tensor(points: np.ndarray, normals: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert points and normals to tensor format expected by DetailGen3D.

    Args:
        points: [N, 3] numpy array
        normals: [N, 3] numpy array
        device: torch device
        dtype: torch dtype

    Returns:
        Tensor of shape [1, N, 6] (xyz + normal)
    """
    # Concatenate points and normals
    surface = np.concatenate([points, normals], axis=-1)

    # Convert to tensor with batch dimension
    tensor = torch.FloatTensor(surface).unsqueeze(0)

    return tensor.to(device=device, dtype=dtype)
