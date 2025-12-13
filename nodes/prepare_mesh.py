"""
DetailGen3D_PrepareMesh node for normalizing mesh and encoding to latent space.
"""

import torch
import numpy as np
from typing import Any

from .utils import (
    get_device,
    get_dtype,
    normalize_mesh,
    sample_surface_points,
    points_to_tensor,
    DETAILGEN3D_MODEL,
    DETAILGEN3D_LATENT,
)


class DetailGen3D_PrepareMesh:
    """
    Prepare mesh for DetailGen3D processing.

    Takes a TRIMESH input (compatible with GeometryPack), normalizes it,
    samples surface points, and encodes to latent space using the VAE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {
                    "tooltip": "Input mesh from GeometryPack's LoadMesh or other mesh node"
                }),
                "model": (DETAILGEN3D_MODEL, {
                    "tooltip": "DetailGen3D model from LoadDetailGen3DModel"
                }),
            },
            "optional": {
                "num_points": ("INT", {
                    "default": 20480,
                    "min": 1024,
                    "max": 100000,
                    "step": 1024,
                    "tooltip": "Number of surface points to sample (default: 20480)"
                }),
                "target_scale": ("FLOAT", {
                    "default": 1.9,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Scale mesh to fit within [-scale, scale] range"
                }),
            }
        }

    RETURN_TYPES = (DETAILGEN3D_LATENT, "TRIMESH")
    RETURN_NAMES = ("latent", "normalized_mesh")
    OUTPUT_TOOLTIPS = (
        "Encoded latent representation - pass to DetailGen3D_Generate",
        "Normalized mesh (centered and scaled) - for reference"
    )
    FUNCTION = "prepare_mesh"
    CATEGORY = "DetailGen3D"
    DESCRIPTION = "Normalize mesh and encode to latent space for DetailGen3D processing."

    def prepare_mesh(
        self,
        trimesh,
        model: Any,
        num_points: int = 20480,
        target_scale: float = 1.9,
    ):
        """
        Prepare mesh for DetailGen3D processing.

        Args:
            trimesh: Input mesh (trimesh.Trimesh object)
            model: DetailGen3D pipeline
            num_points: Number of surface points to sample
            target_scale: Scale to normalize mesh to

        Returns:
            Tuple of (latent_dict, normalized_mesh)
        """
        print(f"[DetailGen3D] PrepareMesh: Processing mesh...")
        print(f"[DetailGen3D] Input mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Get device and dtype from model
        device = next(model.vae.parameters()).device
        dtype = next(model.vae.parameters()).dtype

        # Step 1: Normalize mesh (center and scale)
        print(f"[DetailGen3D] Normalizing mesh to scale {target_scale}...")
        normalized = normalize_mesh(trimesh, target_scale)
        print(f"[DetailGen3D] Normalized mesh bounds: {normalized.bounds}")

        # Step 2: Sample surface points with normals
        print(f"[DetailGen3D] Sampling {num_points} surface points...")
        points, normals = sample_surface_points(normalized, num_points)
        print(f"[DetailGen3D] Sampled points shape: {points.shape}, normals shape: {normals.shape}")

        # Step 3: Convert to tensor
        surface_tensor = points_to_tensor(points, normals, device, dtype)
        print(f"[DetailGen3D] Surface tensor shape: {surface_tensor.shape}")

        # Step 4: Encode with VAE
        print(f"[DetailGen3D] Encoding with VAE...")
        with torch.no_grad():
            latent_dist = model.vae.encode(surface_tensor)
            latent = latent_dist.latent_dist.sample()

        print(f"[DetailGen3D] Latent shape: {latent.shape}")

        # Create latent dict with metadata
        latent_dict = {
            "latent": latent,
            "device": device,
            "dtype": dtype,
            "num_points": num_points,
            "target_scale": target_scale,
        }

        print(f"[DetailGen3D] PrepareMesh complete")

        return (latent_dict, normalized)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DetailGen3D_PrepareMesh": DetailGen3D_PrepareMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailGen3D_PrepareMesh": "DetailGen3D Prepare Mesh",
}
