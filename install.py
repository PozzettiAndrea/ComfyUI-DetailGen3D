# This script is called by ComfyUI-Manager after requirements.txt is installed
# It handles CUDA-specific dependencies like nvdiffrast

import subprocess
import sys
import os
import platform
import tempfile
import urllib.request

# Wheel URLs from https://github.com/PozzettiAndrea/sam3dobjects-wheels
NVDIFFRAST_WHEELS = {
    # CUDA 12.8 wheels
    ("linux", "3.10", "cu128"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu128/nvdiffrast-0.4.0%2Bcu128-cp310-cp310-linux_x86_64.whl",
    ("linux", "3.11", "cu128"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu128/nvdiffrast-0.4.0%2Bcu128-cp311-cp311-linux_x86_64.whl",
    ("linux", "3.12", "cu128"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu128/nvdiffrast-0.4.0%2Bcu128-cp312-cp312-linux_x86_64.whl",
    ("windows", "3.10", "cu128"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu128/nvdiffrast-0.4.0%2Bcu128-cp310-cp310-win_amd64.whl",
    ("windows", "3.11", "cu128"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu128/nvdiffrast-0.4.0%2Bcu128-cp311-cp311-win_amd64.whl",
    ("windows", "3.12", "cu128"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu128/nvdiffrast-0.4.0%2Bcu128-cp312-cp312-win_amd64.whl",
    # CUDA 12.4 wheels (fallback)
    ("linux", "3.10", "cu124"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu124/nvdiffrast-0.3.3%2Bcu124-cp310-cp310-linux_x86_64.whl",
    ("linux", "3.11", "cu124"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu124/nvdiffrast-0.3.3%2Bcu124-cp311-cp311-linux_x86_64.whl",
    ("linux", "3.12", "cu124"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu124/nvdiffrast-0.3.3%2Bcu124-cp312-cp312-linux_x86_64.whl",
    ("windows", "3.10", "cu124"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu124/nvdiffrast-0.3.3%2Bcu124-cp310-cp310-win_amd64.whl",
    ("windows", "3.11", "cu124"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu124/nvdiffrast-0.3.3%2Bcu124-cp311-cp311-win_amd64.whl",
    ("windows", "3.12", "cu124"): "https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu124/nvdiffrast-0.3.3%2Bcu124-cp312-cp312-win_amd64.whl",
}


def get_platform():
    """Get platform string"""
    if platform.system() == "Windows":
        return "windows"
    return "linux"


def get_python_version():
    """Get Python major.minor version"""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_cuda_version():
    """Detect CUDA version - prefer cu128, fallback to cu124"""
    # Try to detect from nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if "CUDA Version: 12" in result.stdout:
            return "cu128"  # Use cu128 for CUDA 12.x
        elif "CUDA Version: 11" in result.stdout:
            return "cu124"  # cu124 might work for CUDA 11.x
    except:
        pass

    return "cu128"  # Default


def get_torch_cuda_version():
    """Get PyTorch version and its CUDA version for PyG wheel index"""
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # e.g., "2.8.0"
        # Get major.minor only
        parts = torch_version.split('.')
        torch_short = f"{parts[0]}.{parts[1]}.0"  # e.g., "2.8.0"

        # Get CUDA version from torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda  # e.g., "12.8"
            if cuda_version:
                cuda_major_minor = cuda_version.replace('.', '')[:3]  # "128" -> "cu128"
                return torch_short, f"cu{cuda_major_minor}"

        return torch_short, "cu121"  # Default fallback
    except Exception as e:
        print(f"[DetailGen3D] Error detecting torch/cuda version: {e}")
        return "2.1.0", "cu121"


def check_torch_cluster_installed():
    """Check if torch_cluster is already installed"""
    try:
        import torch_cluster
        print(f"[DetailGen3D] torch_cluster already installed")
        return True
    except ImportError:
        return False


def install_torch_cluster():
    """Install torch_cluster from PyG wheels"""
    if check_torch_cluster_installed():
        return True

    torch_ver, cuda_ver = get_torch_cuda_version()
    print(f"[DetailGen3D] Installing torch_cluster for PyTorch {torch_ver} + {cuda_ver}")

    # PyG wheel index URL
    wheel_index = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_ver}.html"

    # Try installing from PyG wheels
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch_cluster",
            "-f", wheel_index
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("[DetailGen3D] torch_cluster installed successfully")
            return True
        else:
            print(f"[DetailGen3D] PyG wheel install failed, trying fallback...")
    except Exception as e:
        print(f"[DetailGen3D] Error: {e}")

    # Fallback: try without specific index (will build from source)
    try:
        print("[DetailGen3D] Attempting to build torch_cluster from source...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch_cluster", "--no-cache-dir"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("[DetailGen3D] torch_cluster built from source successfully")
            return True
        else:
            print(f"[DetailGen3D] Failed to install torch_cluster: {result.stderr}")
            return False
    except Exception as e:
        print(f"[DetailGen3D] Error building torch_cluster: {e}")
        return False


def check_nvdiffrast_installed():
    """Check if nvdiffrast is already installed and working"""
    try:
        import nvdiffrast
        print(f"[DetailGen3D] nvdiffrast already installed: {nvdiffrast.__file__}")
        return True
    except ImportError:
        return False


def install_nvdiffrast():
    """Install nvdiffrast from pre-built wheels"""
    if check_nvdiffrast_installed():
        return True

    plat = get_platform()
    py_ver = get_python_version()
    cuda_ver = get_cuda_version()

    print(f"[DetailGen3D] Platform: {plat}, Python: {py_ver}, CUDA: {cuda_ver}")

    # Find matching wheel
    wheel_url = NVDIFFRAST_WHEELS.get((plat, py_ver, cuda_ver))

    # Try fallback CUDA version
    if not wheel_url and cuda_ver == "cu128":
        wheel_url = NVDIFFRAST_WHEELS.get((plat, py_ver, "cu124"))
        if wheel_url:
            print("[DetailGen3D] Using CUDA 12.4 wheel as fallback")

    if not wheel_url:
        print(f"[DetailGen3D] No pre-built wheel for {plat}/{py_ver}/{cuda_ver}")
        return False

    print(f"[DetailGen3D] Downloading nvdiffrast wheel...")

    # Download wheel to temp directory
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            wheel_name = wheel_url.split("/")[-1].replace("%2B", "+")
            wheel_path = os.path.join(tmpdir, wheel_name)

            urllib.request.urlretrieve(wheel_url, wheel_path)
            print(f"[DetailGen3D] Downloaded: {wheel_name}")

            # Install wheel
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", wheel_path
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("[DetailGen3D] nvdiffrast installed successfully")
                return True
            else:
                print(f"[DetailGen3D] Failed to install wheel: {result.stderr}")
                return False

    except Exception as e:
        print(f"[DetailGen3D] Error downloading wheel: {e}")
        return False


if __name__ == "__main__":
    print("[DetailGen3D] Running install script...")

    # Install torch_cluster (required for DetailGen3D model)
    if not install_torch_cluster():
        print("[DetailGen3D] WARNING: torch_cluster installation failed!")
        print("[DetailGen3D] DetailGen3D model will not work without torch_cluster.")
        print("[DetailGen3D] Manual install:")
        print("[DetailGen3D]   pip install torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html")

    # Install nvdiffrast (required for texturing)
    if not install_nvdiffrast():
        print("[DetailGen3D] WARNING: nvdiffrast installation failed!")
        print("[DetailGen3D] Texturing nodes may not work without nvdiffrast.")
        print("[DetailGen3D] Manual install:")
        print("[DetailGen3D]   wget https://github.com/PozzettiAndrea/sam3dobjects-wheels/releases/download/nvdiffrast-cu128/nvdiffrast-0.4.0%2Bcu128-cp310-cp310-linux_x86_64.whl")
        print("[DetailGen3D]   pip install nvdiffrast-0.4.0+cu128-cp310-cp310-linux_x86_64.whl")

    print("[DetailGen3D] Install script complete")
