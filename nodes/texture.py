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
import numpy as np
from PIL import Image
from typing import List
import trimesh

# Add vendor to path
_VENDOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vendor")
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)

from .utils import comfy_image_to_pil
from .load_texture_model import DETAILGEN3D_TEXTURE_MODEL


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
    Bake multi-view images onto mesh UV texture.

    Takes a mesh with UV coordinates and multi-view images,
    projects the images onto the UV space to create a texture.
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
    DESCRIPTION = "Bake multi-view images onto mesh UV to create textured mesh."

    def bake_texture(
        self,
        mesh: trimesh.Trimesh,
        mv_images: torch.Tensor,
        uv_size: int = 2048,
    ):
        print(f"[DetailGen3D] Baking texture (UV size: {uv_size})...")

        # Check for UVs
        if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
            raise ValueError(
                "Mesh has no UV coordinates! Use GeomPack UV Unwrap node first."
            )

        uv = mesh.visual.uv
        print(f"[DetailGen3D] Mesh has {len(uv)} UV coordinates")

        # Convert tensor to list of PIL images
        if mv_images.dim() == 4:
            mv_list = [
                Image.fromarray((mv_images[i].cpu().numpy() * 255).astype(np.uint8))
                for i in range(mv_images.shape[0])
            ]
        else:
            raise ValueError(f"Expected 4D tensor [B,H,W,C], got shape {mv_images.shape}")

        if len(mv_list) != 6:
            print(f"[DetailGen3D] Warning: Expected 6 views, got {len(mv_list)}")

        # Bake texture
        textured_mesh, texture_image = self._bake_texture_impl(
            mesh=mesh,
            mv_images=mv_list,
            uv=uv,
            uv_size=uv_size,
        )

        # Convert texture to ComfyUI format [B, H, W, C]
        texture_np = np.array(texture_image).astype(np.float32) / 255.0
        texture_tensor = torch.from_numpy(texture_np).unsqueeze(0)

        print(f"[DetailGen3D] Texture baking complete")
        return (textured_mesh, texture_tensor)

    def _bake_texture_impl(
        self,
        mesh: trimesh.Trimesh,
        mv_images: List[Image.Image],
        uv: np.ndarray,
        uv_size: int,
    ) -> tuple:
        """Bake multi-view images onto mesh UV texture."""

        # Create texture by blending views
        texture = np.zeros((uv_size, uv_size, 3), dtype=np.float32)
        weight_sum = np.zeros((uv_size, uv_size, 1), dtype=np.float32)

        # Camera azimuths for view selection
        azimuths = [0, 90, 180, 270, 180, 180]
        elevations = [0, 0, 0, 0, 90, -90]

        vertices = mesh.vertices
        faces = mesh.faces
        vertex_normals = mesh.vertex_normals

        for view_idx, mv_img in enumerate(mv_images):
            mv_array = np.array(mv_img).astype(np.float32) / 255.0

            azim = np.radians(azimuths[view_idx])
            elev = np.radians(elevations[view_idx])
            cam_dir = np.array([
                np.cos(elev) * np.sin(azim),
                np.sin(elev),
                np.cos(elev) * np.cos(azim)
            ])

            for face in faces:
                face_uv = uv[face]
                face_normals = vertex_normals[face]
                face_verts = vertices[face]

                avg_normal = face_normals.mean(axis=0)
                avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-8)

                weight = max(0, np.dot(avg_normal, cam_dir))
                if weight < 0.1:
                    continue

                proj_x = (face_verts[:, 0] + 1) / 2
                proj_y = (face_verts[:, 1] + 1) / 2

                cos_a, sin_a = np.cos(-azim), np.sin(-azim)
                rotated_x = proj_x * cos_a - (face_verts[:, 2] + 1) / 2 * sin_a

                img_x = ((rotated_x + 0.5) * mv_array.shape[1]).astype(int).clip(0, mv_array.shape[1] - 1)
                img_y = ((1 - proj_y) * mv_array.shape[0]).astype(int).clip(0, mv_array.shape[0] - 1)

                colors = mv_array[img_y, img_x]

                uv_x = (face_uv[:, 0] * uv_size).astype(int).clip(0, uv_size - 1)
                uv_y = ((1 - face_uv[:, 1]) * uv_size).astype(int).clip(0, uv_size - 1)

                for i in range(3):
                    texture[uv_y[i], uv_x[i]] += colors[i] * weight
                    weight_sum[uv_y[i], uv_x[i]] += weight

        # Normalize by weights (fixed broadcasting)
        mask = weight_sum[:, :, 0] > 0
        texture[mask] /= weight_sum[mask][:, np.newaxis]

        # Fill holes with average color
        if not mask.all():
            avg_color = texture[mask].mean(axis=0) if mask.any() else np.array([0.5, 0.5, 0.5])
            texture[~mask] = avg_color

        # Convert to uint8
        texture = (texture * 255).clip(0, 255).astype(np.uint8)
        texture_image = Image.fromarray(texture)

        # Create textured mesh
        textured_mesh = mesh.copy()
        textured_mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv,
            image=texture_image
        )

        return textured_mesh, texture_image


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DetailGen3D_GenerateMultiView": DetailGen3D_GenerateMultiView,
    "DetailGen3D_BakeTexture": DetailGen3D_BakeTexture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailGen3D_GenerateMultiView": "DetailGen3D Generate MultiView",
    "DetailGen3D_BakeTexture": "DetailGen3D Bake Texture",
}
