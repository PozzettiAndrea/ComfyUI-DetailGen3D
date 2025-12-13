"""
DetailGen3D_Decode node for converting SDF to mesh via marching cubes.
"""

import numpy as np
import trimesh
from typing import Any

from .utils import DETAILGEN3D_SDF


class DetailGen3D_Decode:
    """
    Convert SDF (Signed Distance Field) to mesh using marching cubes.

    Outputs a TRIMESH compatible with GeometryPack nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sdf": (DETAILGEN3D_SDF, {
                    "tooltip": "SDF from DetailGen3D_Generate"
                }),
            },
            "optional": {
                "iso_level": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Isosurface level for marching cubes (0 = surface)"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    OUTPUT_TOOLTIPS = (
        "Refined mesh - compatible with GeometryPack nodes (SaveMesh, PreviewMesh, etc.)",
    )
    FUNCTION = "decode"
    CATEGORY = "DetailGen3D"
    DESCRIPTION = "Convert SDF to mesh using marching cubes. Outputs TRIMESH for GeometryPack compatibility."

    def decode(
        self,
        sdf: dict,
        iso_level: float = 0.0,
    ):
        """
        Convert SDF to mesh using marching cubes.

        Args:
            sdf: SDF dict from Generate node
            iso_level: Isosurface level for extraction

        Returns:
            Tuple of (trimesh.Trimesh,)
        """
        print(f"[DetailGen3D] Decode: Extracting mesh from SDF...")

        # Extract data from SDF dict
        sdf_tensor = sdf["sdf"]
        grid_size = sdf["grid_size"]
        bbox_size = sdf["bbox_size"]
        box_min = sdf["box_min"]

        # Convert to numpy and reshape to 3D grid
        sdf_np = sdf_tensor.cpu().numpy()
        grid_logits = sdf_np.reshape(grid_size)

        print(f"[DetailGen3D] SDF grid shape: {grid_logits.shape}")
        print(f"[DetailGen3D] SDF range: [{grid_logits.min():.4f}, {grid_logits.max():.4f}]")

        # Run marching cubes
        print(f"[DetailGen3D] Running marching cubes (iso_level={iso_level})...")
        try:
            from skimage import measure

            vertices, faces, normals, _ = measure.marching_cubes(
                grid_logits,
                level=iso_level,
                method="lewiner"
            )

            print(f"[DetailGen3D] Marching cubes output: {len(vertices)} vertices, {len(faces)} faces")

        except Exception as e:
            raise RuntimeError(f"Marching cubes failed: {e}") from e

        # Rescale vertices from grid coordinates to world coordinates
        # vertices are in [0, grid_size] space, need to map to [box_min, box_max]
        vertices = vertices / np.array(grid_size) * bbox_size + box_min

        print(f"[DetailGen3D] Rescaled vertices bounds: [{vertices.min(axis=0)}, {vertices.max(axis=0)}]")

        # Create trimesh object
        mesh = trimesh.Trimesh(
            vertices=vertices.astype(np.float32),
            faces=np.ascontiguousarray(faces),
            process=True  # Clean up the mesh
        )

        # Store metadata
        mesh.metadata['source'] = 'DetailGen3D'
        mesh.metadata['iso_level'] = iso_level

        print(f"[DetailGen3D] Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"[DetailGen3D] Mesh bounds: {mesh.bounds}")
        print(f"[DetailGen3D] Decode complete")

        return (mesh,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "DetailGen3D_Decode": DetailGen3D_Decode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DetailGen3D_Decode": "DetailGen3D Decode",
}
