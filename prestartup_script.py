"""
ComfyUI-DetailGen3D Prestartup Script

Copies example assets to ComfyUI input folders on startup.
"""

import shutil
from pathlib import Path


def copy_example_assets():
    """Copy example assets to ComfyUI input folders if they don't exist."""

    # Get paths
    node_dir = Path(__file__).parent
    comfyui_dir = node_dir.parent.parent  # ComfyUI root (custom_nodes -> ComfyUI)

    # Source assets directory (in this node's folder)
    assets_dir = node_dir / "assets"

    if not assets_dir.exists():
        return

    # Destination directories
    input_dir = comfyui_dir / "input"
    input_3d_dir = input_dir / "3d"

    # Create directories if needed
    input_dir.mkdir(parents=True, exist_ok=True)
    input_3d_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0

    # Copy all GLB files to input/3d
    for glb_file in assets_dir.rglob("*.glb"):
        dst = input_3d_dir / glb_file.name
        if not dst.exists():
            try:
                shutil.copy2(glb_file, dst)
                print(f"[DetailGen3D] Copied: {glb_file.name} -> input/3d/")
                copied_count += 1
            except Exception as e:
                print(f"[DetailGen3D] Warning: Failed to copy {glb_file.name}: {e}")

    # Copy all image files to input/
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for img_file in assets_dir.rglob(ext):
            dst = input_dir / img_file.name
            if not dst.exists():
                try:
                    shutil.copy2(img_file, dst)
                    print(f"[DetailGen3D] Copied: {img_file.name} -> input/")
                    copied_count += 1
                except Exception as e:
                    print(f"[DetailGen3D] Warning: Failed to copy {img_file.name}: {e}")

    if copied_count > 0:
        print(f"[DetailGen3D] Copied {copied_count} example assets to input folders")


# Run on import (prestartup)
try:
    copy_example_assets()
except Exception as e:
    print(f"[DetailGen3D] Warning: Prestartup asset copy failed: {e}")
