----------
Work in Progress! This node is not finished.
----------

# ComfyUI-DetailGen3D

ComfyUI custom nodes for [DetailGen3D](https://github.com/VAST-AI-Research/DetailGen3D) - enhance coarse 3D meshes with geometric detail.

## Features

- **TRIMESH compatible**: Works with ComfyUI-GeometryPack's mesh nodes
- **Modular pipeline**: Separate nodes for each stage of processing
- **HuggingFace integration**: Automatic model downloading from VAST-AI/DetailGen3D

## Nodes

| Node | Description |
|------|-------------|
| **Load DetailGen3D Model** | Load the DetailGen3D pipeline from HuggingFace |
| **DetailGen3D Prepare Mesh** | Normalize mesh and encode to latent space |
| **DetailGen3D Generate** | Run diffusion pipeline with image conditioning |
| **DetailGen3D Decode** | Convert SDF to mesh via marching cubes |

## Workflow

```
[GeomPack LoadMesh] → TRIMESH
                          ↓
[LoadDetailGen3DModel] → MODEL
                          ↓
[DetailGen3D_PrepareMesh] → LATENT + normalized TRIMESH
                          ↓
[LoadImage] → IMAGE ──────┘
                          ↓
[DetailGen3D_Generate] → SDF
                          ↓
[DetailGen3D_Decode] → TRIMESH
                          ↓
[GeomPack SaveMesh] or [GeomPack PreviewMesh]
```

## Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "DetailGen3D"
3. Click Install

### Manual Installation
1. Clone this repository into `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/PozzettiAndrea/ComfyUI-DetailGen3D
   ```
2. Install dependencies:
   ```bash
   pip install -r ComfyUI-DetailGen3D/requirements.txt
   ```
3. Restart ComfyUI

## Requirements

- ComfyUI
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Python 3.10+

## Model

The DetailGen3D model is automatically downloaded from HuggingFace on first use:
- Repository: [VAST-AI/DetailGen3D](https://huggingface.co/VAST-AI/DetailGen3D)
- Storage location: `ComfyUI/models/detailgen3d/`

## Parameters

### DetailGen3D Generate

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| guidance_scale | 10.0 | 1-20 | Classifier-free guidance strength |
| num_inference_steps | 50 | 10-100 | Denoising steps |
| noise_aug_level | 0.0 | 0-1 | Noise augmentation |
| seed | 42 | - | Random seed for reproducibility |
| octree_depth | 9 | 7-10 | Grid resolution (9 = 513³) |

## Compatibility

- **ComfyUI-GeometryPack**: Full compatibility with TRIMESH type
- Input: Use GeomPack's LoadMesh node to load coarse meshes
- Output: Use GeomPack's SaveMesh or PreviewMesh nodes

## Credits

- [DetailGen3D](https://github.com/VAST-AI-Research/DetailGen3D) by VAST-AI Research
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

MIT License - see [LICENSE](LICENSE) file
