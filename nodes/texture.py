"""
DetailGen3D Texture nodes for generating and baking textures.

Split into modular nodes:
- DetailGen3D_GenerateMultiView: Render mesh + run MV-Adapter → multi-view images
- DetailGen3D_BakeTexture: Bake multi-view images onto mesh UV
"""

import os
import sys
import tempfile
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional
import trimesh

# Add vendor to path
_VENDOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)

from .utils import comfy_image_to_pil
from .load_texture_model import DETAILGEN3D_TEXTURE_MODEL

# Import vendor utilities for proper UV baking
from mvadapter.utils.mesh_utils import (
    TexturedMesh,
    get_orthogonal_camera,
    NVDiffRastContextWrapper,
)
from mvadapter.utils.mesh_utils.projection import CameraProjection
from mvadapter.utils.mesh_utils.utils import tensor_to_image


def trimesh_to_textured_mesh(
    mesh: trimesh.Trimesh,
    uv_size: int = 2048,
    device: str = "cuda",
    rescale: bool = True,
    scale: float = 0.5,
) -> TexturedMesh:
    """Convert a trimesh.Trimesh to TexturedMesh for nvdiffrast processing.

    Args:
        mesh: Input trimesh
        uv_size: UV texture resolution
        device: Target device
        rescale: If True, center and rescale mesh to fit in [-scale, scale]
                 This MUST match the rescaling done in GenerateMultiView's load_mesh()
        scale: Target scale (default 0.5 matches load_mesh default)
    """

    # Get vertices and faces
    vertices = mesh.vertices.copy()

    if rescale:
        # Center mesh (same as load_mesh)
        centroid = vertices.mean(axis=0)
        vertices = vertices - centroid
        # Scale to fit in [-scale, scale] (same as load_mesh)
        max_extent = np.abs(vertices).max()
        if max_extent > 0:
            vertices = vertices / max_extent * scale

    v_pos = torch.tensor(vertices, dtype=torch.float32)
    t_pos_idx = torch.tensor(mesh.faces, dtype=torch.int64)

    # Get UVs if available
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        v_tex = torch.tensor(mesh.visual.uv, dtype=torch.float32)
        # Flip V coordinate (trimesh uses bottom-left origin, nvdiffrast uses top-left)
        v_tex[:, 1] = 1.0 - v_tex[:, 1]
        t_tex_idx = t_pos_idx.clone()
    else:
        raise ValueError("Mesh must have UV coordinates for texture baking")

    # Create empty texture
    texture = torch.zeros((uv_size, uv_size, 3), dtype=torch.float32)

    # Create TexturedMesh
    textured_mesh = TexturedMesh(
        v_pos=v_pos,
        t_pos_idx=t_pos_idx,
        v_tex=v_tex,
        t_tex_idx=t_tex_idx,
        texture=texture,
    )

    # Set stitched mesh (for proper normal computation)
    textured_mesh.set_stitched_mesh(v_pos, t_pos_idx)

    # Move to device
    textured_mesh.to(device)

    return textured_mesh


class DetailGen3D_GenerateMultiView:
    """
    Generate multi-view images from mesh using MV-Adapter.

    Takes a mesh and reference image, renders the mesh from 6 views,
    and runs MV-Adapter diffusion to generate textured view images.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texture_model": (DETAILGEN3D_TEXTURE_MODEL, {
                    "tooltip": "Texture model from LoadDetailGen3D_TextureModel"
                }),
                "mesh": ("TRIMESH", {
                    "tooltip": "Input mesh"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Reference image for texture generation"
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "default": "high quality, detailed texture",
                    "tooltip": "Text prompt for texture generation"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Guidance scale for diffusion"
                }),
                "num_inference_steps": ("INT", {
                    "default": 30,
                    "min": 10,
                    "max": 50,
                    "step": 5,
                    "tooltip": "Number of diffusion steps"
                }),
                "resolution": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Resolution for multi-view rendering"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Random seed for reproducibility"
                }),
                "reference_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Reference image conditioning scale"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mv_images",)
    OUTPUT_TOOLTIPS = ("6 multi-view images as batch [6, H, W, C]",)
    FUNCTION = "generate_multiview"
    CATEGORY = "DetailGen3D"
    DESCRIPTION = "Generate 6 multi-view textured images from mesh using MV-Adapter."

    def generate_multiview(
        self,
        texture_model: dict,
        mesh: trimesh.Trimesh,
        image: torch.Tensor,
        prompt: str = "high quality, detailed texture",
        guidance_scale: float = 3.0,
        num_inference_steps: int = 30,
        resolution: int = 512,
        seed: int = 42,
        reference_scale: float = 1.0,
    ):
        print(f"[DetailGen3D] Generating multi-view images...")
        print(f"[DetailGen3D] Resolution: {resolution}")

        pipe = texture_model["pipeline"]
        device = texture_model["device"]
        num_views = texture_model["num_views"]

        # Convert image to PIL
        pil_image = comfy_image_to_pil(image)
        print(f"[DetailGen3D] Reference image size: {pil_image.size}")

        # Save mesh to temp file
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            temp_mesh_path = f.name
            mesh.export(temp_mesh_path)

        try:
            from mvadapter.utils.mesh_utils import (
                NVDiffRastContextWrapper,
                get_orthogonal_camera,
                load_mesh,
                render,
            )

            # Setup cameras for 6 views
            cameras = get_orthogonal_camera(
                elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
                distance=[1.8] * num_views,
                left=-0.55,
                right=0.55,
                bottom=-0.55,
                top=0.55,
                azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
                device=device,
            )

            # Create nvdiffrast context
            ctx = NVDiffRastContextWrapper(device=device)

            # Load and render mesh
            print(f"[DetailGen3D] Rendering mesh from {num_views} views...")
            loaded_mesh = load_mesh(temp_mesh_path, rescale=True, device=device)
            render_out = render(
                ctx,
                loaded_mesh,
                cameras,
                height=resolution,
                width=resolution,
                render_attr=False,
                normal_background=0.0,
            )

            # Prepare control images (position + normal maps)
            control_images = (
                torch.cat(
                    [
                        (render_out.pos + 0.5).clamp(0, 1),
                        (render_out.normal / 2 + 0.5).clamp(0, 1),
                    ],
                    dim=-1,
                )
                .permute(0, 3, 1, 2)
                .to(device)
            )

            # Preprocess reference image
            reference_image = self._preprocess_image(pil_image, resolution, resolution)

            # Generate multi-view images
            print(f"[DetailGen3D] Running MV-Adapter diffusion ({num_inference_steps} steps)...")
            generator = torch.Generator(device=device).manual_seed(seed)

            mv_images = pipe(
                prompt,
                height=resolution,
                width=resolution,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_views,
                control_image=control_images,
                control_conditioning_scale=1.0,
                reference_image=reference_image,
                reference_conditioning_scale=reference_scale,
                negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
                generator=generator,
            ).images

            print(f"[DetailGen3D] Generated {len(mv_images)} view images")

            # Convert PIL images to ComfyUI tensor [B, H, W, C]
            mv_arrays = [np.array(img).astype(np.float32) / 255.0 for img in mv_images]
            mv_tensor = torch.from_numpy(np.stack(mv_arrays, axis=0))

            return (mv_tensor,)

        finally:
            if os.path.exists(temp_mesh_path):
                os.unlink(temp_mesh_path)

    def _preprocess_image(self, image: Image.Image, height: int, width: int) -> Image.Image:
        """Preprocess reference image for MV-Adapter"""
        image = np.array(image)

        if image.shape[-1] == 4:
            alpha = image[..., 3] > 0
            H, W = alpha.shape
            y, x = np.where(alpha)
            if len(y) > 0 and len(x) > 0:
                y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
                x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
                image_center = image[y0:y1, x0:x1]

                H_c, W_c, _ = image_center.shape
                if H_c > W_c:
                    W_new = int(W_c * (height * 0.9) / H_c)
                    H_new = int(height * 0.9)
                else:
                    H_new = int(H_c * (width * 0.9) / W_c)
                    W_new = int(width * 0.9)

                image_center = np.array(Image.fromarray(image_center).resize((W_new, H_new)))

                start_h = (height - H_new) // 2
                start_w = (width - W_new) // 2
                padded = np.zeros((height, width, 4), dtype=np.uint8)
                padded[start_h:start_h + H_new, start_w:start_w + W_new] = image_center

                image = padded.astype(np.float32) / 255.0
                image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
                image = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                image = image[:, :, :3]

        return Image.fromarray(image).resize((width, height))


class DetailGen3D_BakeTexture:
    """
    Bake multi-view images onto mesh UV texture using proper nvdiffrast projection.

    Takes a mesh with UV coordinates and multi-view images,
    projects the images onto the UV space to create a texture using
    proper camera projection, depth testing, and visibility blending.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH", {
                    "tooltip": "Mesh with UV coordinates (use GeomPack UV Unwrap if needed)"
                }),
                "mv_images": ("IMAGE", {
                    "tooltip": "Multi-view images from DetailGen3D_GenerateMultiView (6 views)"
                }),
            },
            "optional": {
                "uv_size": ("INT", {
                    "default": 2048,
                    "min": 512,
                    "max": 4096,
                    "step": 256,
                    "tooltip": "UV texture resolution"
                }),
                "blend_alpha": ("FLOAT", {
                    "default": 3.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Exponential blending alpha (higher = sharper transitions)"
                }),
                "aoi_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.05,
                    "tooltip": "Angle of incidence threshold for validity"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE")
    RETURN_NAMES = ("textured_mesh", "texture_map")
    OUTPUT_TOOLTIPS = (
        "Mesh with baked texture",
        "The UV texture map image",
    )
    FUNCTION = "bake_texture"
    CATEGORY = "DetailGen3D"
    DESCRIPTION = "Bake multi-view images onto mesh UV using proper nvdiffrast projection."

    def bake_texture(
        self,
        mesh: trimesh.Trimesh,
        mv_images: torch.Tensor,
        uv_size: int = 2048,
        blend_alpha: float = 3.0,
        aoi_threshold: float = 0.2,
    ):
        print(f"[DetailGen3D] Baking texture with nvdiffrast (UV size: {uv_size})...")

        # Check for UVs
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            raise ValueError(
                "Mesh has no UV coordinates! Use GeomPack UV Unwrap node first."
            )

        print(f"[DetailGen3D] Mesh has {len(mesh.visual.uv)} UV coordinates")
        print(f"[DetailGen3D] Mesh has {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Validate input
        if mv_images.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,H,W,C], got shape {mv_images.shape}")

        num_views = mv_images.shape[0]
        if num_views != 6:
            print(f"[DetailGen3D] Warning: Expected 6 views, got {num_views}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert multi-view images to tensor format for projection
        # mv_images is [B, H, W, C] in range [0, 1]
        mv_tensor = mv_images.to(device)
        print(f"[DetailGen3D] Multi-view images shape: {mv_tensor.shape}")

        # Bake texture using proper projection
        texture_tensor = self._bake_with_projection(
            mesh=mesh,
            mv_images=mv_tensor,
            uv_size=uv_size,
            blend_alpha=blend_alpha,
            aoi_threshold=aoi_threshold,
            device=device,
        )

        # Convert texture to PIL for trimesh
        texture_np = (texture_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        texture_image = Image.fromarray(texture_np)

        # Create textured mesh
        textured_mesh = mesh.copy()
        textured_mesh.visual = trimesh.visual.TextureVisuals(
            uv=mesh.visual.uv,
            image=texture_image
        )

        # Convert texture to ComfyUI format [B, H, W, C]
        texture_out = texture_tensor.unsqueeze(0).cpu()

        print(f"[DetailGen3D] Texture baking complete")
        return (textured_mesh, texture_out)

    def _bake_with_projection(
        self,
        mesh: trimesh.Trimesh,
        mv_images: torch.Tensor,
        uv_size: int,
        blend_alpha: float,
        aoi_threshold: float,
        device: str,
    ) -> torch.Tensor:
        """Bake multi-view images using proper nvdiffrast camera projection."""

        print(f"[DetailGen3D] Converting mesh to TexturedMesh format...")
        print(f"[DetailGen3D] Original mesh bounds: {mesh.bounds}")

        # Convert trimesh to TexturedMesh (with rescaling to match GenerateMultiView)
        textured_mesh = trimesh_to_textured_mesh(mesh, uv_size=uv_size, device=device, rescale=True, scale=0.5)
        print(f"[DetailGen3D] Rescaled mesh bounds: [{textured_mesh.v_pos.min().item():.3f}, {textured_mesh.v_pos.max().item():.3f}]")

        # Setup cameras - MUST match GenerateMultiView exactly
        # GenerateMultiView uses: azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]]
        # Which equals: [-90, 0, 90, 180, 90, 90]
        num_views = mv_images.shape[0]
        base_azimuths = [0, 90, 180, 270, 180, 180][:num_views]
        elevations = [0, 0, 0, 0, 89.99, -89.99][:num_views]

        print(f"[DetailGen3D] Setting up {num_views} cameras...")
        cameras = get_orthogonal_camera(
            elevation_deg=elevations,
            distance=[1.8] * num_views,
            left=-0.55,
            right=0.55,
            bottom=-0.55,
            top=0.55,
            azimuth_deg=[x - 90 for x in base_azimuths],  # Match GenerateMultiView
            device=device,
        )

        # Create CameraProjection for proper UV baking
        print(f"[DetailGen3D] Initializing CameraProjection...")
        cam_proj = CameraProjection(
            pb_backend="torch-native",  # Use native PyTorch (no CUDA kernel compilation)
            bg_remover=None,
            device=device,
        )

        # Project multi-view images onto UV space
        print(f"[DetailGen3D] Projecting views onto UV space...")
        uv_texture = cam_proj(
            images=mv_images,
            mesh=textured_mesh,
            cam=cameras,
            uv_size=uv_size,
            aoi_cos_valid_threshold=aoi_threshold,
            depth_grad_dilation=5,
            depth_grad_threshold=0.1,
            uv_exp_blend_alpha=blend_alpha,
            uv_exp_blend_view_weight=torch.ones(num_views),
            poisson_blending=False,  # Faster without poisson
            from_scratch=True,
            uv_padding=True,
            return_dict=False,
        )

        if uv_texture is None:
            raise RuntimeError("UV projection failed - check mesh alignment with views")

        print(f"[DetailGen3D] UV texture shape: {uv_texture.shape}")

        # Clamp and return
        uv_texture = uv_texture.clamp(0.0, 1.0)

        return uv_texture


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DetailGen3D_GenerateMultiView": DetailGen3D_GenerateMultiView,
    "DetailGen3D_BakeTexture": DetailGen3D_BakeTexture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailGen3D_GenerateMultiView": "DetailGen3D Generate MultiView",
    "DetailGen3D_BakeTexture": "DetailGen3D Bake Texture",
}
