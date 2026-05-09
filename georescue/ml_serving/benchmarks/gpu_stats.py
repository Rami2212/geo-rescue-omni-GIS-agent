"""GPU monitoring utilities for AMD MI300X via ROCm."""

import subprocess
import torch


def get_rocm_stats():
    """Get GPU stats via rocm-smi (AMD equivalent of nvidia-smi)."""
    try:
        result = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=10)
        return result.stdout
    except FileNotFoundError:
        return "rocm-smi not found — are you running on an AMD GPU instance?"
    except subprocess.TimeoutExpired:
        return "rocm-smi timed out"


def get_torch_gpu_stats():
    """Get GPU memory stats from PyTorch."""
    if not torch.cuda.is_available():
        return {"error": "No GPU available via PyTorch"}

    return {
        "device": torch.cuda.get_device_name(0),
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
    }


if __name__ == "__main__":
    print("=== ROCm SMI ===")
    print(get_rocm_stats())
    print("\n=== PyTorch GPU Stats ===")
    stats = get_torch_gpu_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
