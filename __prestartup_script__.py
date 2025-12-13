"""
ComfyUI-DetailGen3D Prestartup Script

Copies example assets to ComfyUI input folders on startup.
"""

import os
import shutil
from pathlib import Path


def copy_example_assets():
    """Copy example assets to ComfyUI input folders if they don't exist."""

    # Get paths
    node_dir = Path(__file__).parent
    comfyui_dir = node_dir.parent.parent.parent  # ComfyUI root

    # Source assets directory (in DetailGen3D repo)
    assets_dir = comfyui_dir.parent / "DetailGen3D" / "assets"

    # Destination directories
    input_dir = comfyui_dir / "input"
    input_3d_dir = input_dir / "3d"

    # Create directories if needed
    input_dir.mkdir(parents=True, exist_ok=True)
    input_3d_dir.mkdir(parents=True, exist_ok=True)

    # Copy example files
    assets_to_copy = [
        # (source, destination)
        (assets_dir / "elephant.png", input_dir / "elephant.png"),
        (assets_dir / "mesh.glb", input_3d_dir / "elephant.glb"),
        # Also copy paired examples from subfolders
        (assets_dir / "image" / "503d193a-1b9b-4685-b05f-00ac82f93d7b.png", input_dir / "example_chair.png"),
        (assets_dir / "model" / "503d193a-1b9b-4685-b05f-00ac82f93d7b.glb", input_3d_dir / "example_chair.glb"),
    ]

    copied_count = 0
    for src, dst in assets_to_copy:
        if src.exists() and not dst.exists():
            try:
                shutil.copy2(src, dst)
                print(f"[DetailGen3D] Copied example asset: {dst.name}")
                copied_count += 1
            except Exception as e:
                print(f"[DetailGen3D] Warning: Failed to copy {src.name}: {e}")

    if copied_count > 0:
        print(f"[DetailGen3D] Copied {copied_count} example assets to input folders")


# Run on import (prestartup)
try:
    copy_example_assets()
except Exception as e:
    print(f"[DetailGen3D] Warning: Prestartup asset copy failed: {e}")
