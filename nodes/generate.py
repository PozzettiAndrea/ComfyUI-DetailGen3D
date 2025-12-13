"""
DetailGen3D_Generate node for running the diffusion pipeline and extracting mesh.
"""

import torch
import numpy as np
import trimesh
from typing import Any

from .utils import (
    comfy_image_to_pil,
    DETAILGEN3D_MODEL,
    DETAILGEN3D_LATENT,
)


class DetailGen3D_Generate:
    """
    Run DetailGen3D diffusion pipeline with image conditioning.

    Takes encoded latent from PrepareMesh and a reference image,
    generates refined geometry and outputs a TRIMESH.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (DETAILGEN3D_MODEL, {
                    "tooltip": "DetailGen3D model from LoadDetailGen3DModel"
                }),
                "latent": (DETAILGEN3D_LATENT, {
                    "tooltip": "Encoded latent from DetailGen3D_PrepareMesh"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Reference image for conditioning"
                }),
                "mask": ("MASK", {
                    "tooltip": "Mask for the object (white=object, black=background). Composites object onto white background."
                }),
            },
            "optional": {
                "guidance_scale": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Classifier-free guidance scale (higher = more faithful to image)"
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 10,
                    "max": 100,
                    "step": 5,
                    "tooltip": "Number of denoising steps (more = higher quality, slower)"
                }),
                "noise_aug_level": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Noise augmentation level (0 = no noise, adds variation)"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "control_after_generate": "fixed",
                    "tooltip": "Random seed for reproducibility"
                }),
                "octree_depth": ("INT", {
                    "default": 9,
                    "min": 7,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Octree depth for query points (9 = 513^3 grid, higher = more detail)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    OUTPUT_TOOLTIPS = (
        "Refined mesh - compatible with GeometryPack nodes",
    )
    FUNCTION = "generate"
    CATEGORY = "DetailGen3D"
    DESCRIPTION = "Run DetailGen3D diffusion to generate refined geometry. Outputs TRIMESH."

    def generate(
        self,
        model: Any,
        latent: dict,
        image: torch.Tensor,
        mask: torch.Tensor,
        guidance_scale: float = 10.0,
        num_inference_steps: int = 50,
        noise_aug_level: float = 0.0,
        seed: int = 42,
        octree_depth: int = 9,
    ):
        """
        Run DetailGen3D diffusion pipeline and extract mesh.

        Args:
            model: DetailGen3D pipeline
            latent: Encoded latent dict from PrepareMesh
            image: Reference image tensor [B, H, W, C]
            mask: Optional mask tensor [B, H, W] or [H, W] (white=object, black=background)
            guidance_scale: CFG scale
            num_inference_steps: Denoising steps
            noise_aug_level: Noise augmentation
            seed: Random seed
            octree_depth: Query point grid resolution

        Returns:
            Tuple of (trimesh.Trimesh,)
        """
        print(f"[DetailGen3D] Generate: Starting diffusion...")
        print(f"[DetailGen3D] Parameters: guidance={guidance_scale}, steps={num_inference_steps}, noise_aug={noise_aug_level}, seed={seed}")

        # Get device and dtype from latent
        device = latent["device"]
        dtype = latent["dtype"]
        encoded_latent = latent["latent"]

        # Apply mask - composite object onto white background
        print(f"[DetailGen3D] Applying mask to composite onto white background...")
        # Handle mask dimensions - ComfyUI masks are [B, H, W] or [H, W]
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # Add batch dim
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(-1)  # Add channel dim for broadcasting [B, H, W, 1]

        # Ensure mask matches image dimensions
        if mask.shape[1:3] != image.shape[1:3]:
            # Resize mask to match image
            import torch.nn.functional as F
            mask = mask.permute(0, 3, 1, 2)  # [B, 1, H, W]
            mask = F.interpolate(mask, size=(image.shape[1], image.shape[2]), mode='bilinear', align_corners=False)
            mask = mask.permute(0, 2, 3, 1)  # [B, H, W, 1]

        # Create white background
        white_bg = torch.ones_like(image)

        # Composite: object * mask + white * (1 - mask)
        image = image * mask + white_bg * (1 - mask)
        print(f"[DetailGen3D] Image composited onto white background")

        # Convert image to PIL
        pil_image = comfy_image_to_pil(image)
        print(f"[DetailGen3D] Image size: {pil_image.size}")

        # Generate query points for decoding
        print(f"[DetailGen3D] Generating query points (octree depth {octree_depth})...")
        from detailgen3d.inference_utils import generate_dense_grid_points

        box_min = np.array([-1.005, -1.005, -1.005])
        box_max = np.array([1.005, 1.005, 1.005])

        sampled_points, grid_size, bbox_size = generate_dense_grid_points(
            bbox_min=box_min,
            bbox_max=box_max,
            octree_depth=octree_depth,
            indexing="ij"
        )

        # Convert to tensor
        sampled_points = torch.FloatTensor(sampled_points).to(device, dtype=dtype)
        sampled_points = sampled_points.unsqueeze(0)  # Add batch dimension

        print(f"[DetailGen3D] Query points shape: {sampled_points.shape}")
        print(f"[DetailGen3D] Grid size: {grid_size}")

        # Create generator for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)

        # Run diffusion pipeline
        import time
        print(f"[DetailGen3D] Running diffusion pipeline ({num_inference_steps} steps)...")
        t0 = time.time()
        with torch.no_grad():
            output = model(
                pil_image,
                latents=encoded_latent,
                sampled_points=sampled_points,
                noise_aug_level=noise_aug_level,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
        print(f"[DetailGen3D] Diffusion complete in {time.time() - t0:.2f}s")

        # Extract SDF
        t0 = time.time()
        print(f"[DetailGen3D] Extracting SDF from output...")
        sdf = output.samples[0]
        print(f"[DetailGen3D] SDF shape: {sdf.shape}, extracted in {time.time() - t0:.2f}s")

        # Transfer to CPU and reshape
        t0 = time.time()
        print(f"[DetailGen3D] Transferring SDF to CPU...")
        sdf_np = sdf.cpu().numpy()
        print(f"[DetailGen3D] CPU transfer done in {time.time() - t0:.2f}s")

        t0 = time.time()
        print(f"[DetailGen3D] Reshaping SDF to grid {grid_size}...")
        grid_logits = sdf_np.reshape(grid_size)
        print(f"[DetailGen3D] Reshape done in {time.time() - t0:.2f}s")

        # Run marching cubes to extract mesh
        t0 = time.time()
        print(f"[DetailGen3D] Running marching cubes (this may take a while for 513³ grid)...")
        from skimage import measure

        vertices, faces, normals, _ = measure.marching_cubes(
            grid_logits,
            level=0.0,  # Always use 0 for true surface
            method="lewiner"
        )
        print(f"[DetailGen3D] Marching cubes done in {time.time() - t0:.2f}s")

        # Rescale vertices from grid coordinates to world coordinates
        t0 = time.time()
        print(f"[DetailGen3D] Rescaling vertices to world coordinates...")
        vertices = vertices / np.array(grid_size) * bbox_size + box_min
        print(f"[DetailGen3D] Rescale done in {time.time() - t0:.2f}s")

        # Create trimesh object
        t0 = time.time()
        print(f"[DetailGen3D] Creating trimesh object...")
        mesh = trimesh.Trimesh(
            vertices=vertices.astype(np.float32),
            faces=np.ascontiguousarray(faces),
            process=True
        )
        print(f"[DetailGen3D] Trimesh created in {time.time() - t0:.2f}s")

        mesh.metadata['source'] = 'DetailGen3D'
        mesh.metadata['guidance_scale'] = guidance_scale
        mesh.metadata['num_inference_steps'] = num_inference_steps

        print(f"[DetailGen3D] Output mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[DetailGen3D] Generate complete")

        return (mesh,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DetailGen3D_Generate": DetailGen3D_Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailGen3D_Generate": "DetailGen3D Generate",
}
