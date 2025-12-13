"""
ComfyUI-DetailGen3D: ComfyUI custom nodes for DetailGen3D

Enhance coarse 3D meshes with geometric detail using DetailGen3D.
Compatible with ComfyUI-GeometryPack's TRIMESH type.

Nodes:
- LoadDetailGen3DModel: Load DetailGen3D inference pipeline
- DetailGen3D_PrepareMesh: Normalize mesh and encode to latent space
- DetailGen3D_Generate: Run diffusion pipeline with image conditioning
- DetailGen3D_Decode: Convert SDF to mesh via marching cubes
"""

import os
import sys

# Add vendor directory to path FIRST, before any other imports
_VENDOR_PATH = os.path.join(os.path.dirname(__file__), "vendor")
if _VENDOR_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_PATH)

# Import all node classes
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "0.0.1"
__author__ = "ComfyUI-DetailGen3D Contributors"

# Print info when loaded
print("[DetailGen3D] Loading ComfyUI-DetailGen3D extension")
print(f"[DetailGen3D] Version: {__version__}")
print("[DetailGen3D] ")
print("[DetailGen3D] Available nodes:")
print("[DetailGen3D]   1. LoadDetailGen3DModel - Load model from HuggingFace")
print("[DetailGen3D]   2. DetailGen3D_PrepareMesh - Normalize and encode mesh")
print("[DetailGen3D]   3. DetailGen3D_Generate - Run diffusion pipeline")
print("[DetailGen3D]   4. DetailGen3D_Decode - Extract mesh from SDF")
print("[DetailGen3D] ")
print("[DetailGen3D] Compatible with GeometryPack TRIMESH type")
print("[DetailGen3D] ")

# Export required attributes for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
