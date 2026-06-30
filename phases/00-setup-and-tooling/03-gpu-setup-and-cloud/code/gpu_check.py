import time
import sys


def benchmark(device, size=4000):
    """在指定设备上执行矩阵乘法基准测试，返回耗时（秒）。"""
    import torch

    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # CPU 基准
    start = time.time()
    _ = a @ b
    cpu_time = time.time() - start
    print(f"CPU matrix multiply ({size}x{size}): {cpu_time:.3f}s")

    if device == "cpu":
        return

    a_dev = a.to(device)
    b_dev = b.to(device)

    # 同步函数按设备区分
    sync = torch.cuda.synchronize if device == "cuda" else torch.mps.synchronize

    # 预热一次，避免首次内核编译/搬运影响计时
    _ = a_dev @ b_dev
    sync()

    start = time.time()
    _ = a_dev @ b_dev
    sync()
    gpu_time = time.time() - start
    print(f"GPU matrix multiply ({size}x{size}): {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")


def check_gpu():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        return

    print("=== GPU Check ===\n")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # ---------- 分支 1：NVIDIA CUDA ----------
    if torch.cuda.is_available():
        print(f"\nCUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")

        print("\n=== CPU vs GPU Benchmark ===\n")
        benchmark("cuda")

        vram_gb = props.total_memory / 1e9
        params_billions = vram_gb * 1e9 / 2 / 1e9
        print(f"\nEstimated max model size (fp16): ~{params_billions:.0f}B parameters")
        return

    # ---------- 分支 2：Apple Silicon MPS ----------
    if torch.backends.mps.is_available():
        print(f"MPS built: {torch.backends.mps.is_built()}")
        print("GPU: Apple Silicon (Metal / MPS)")

        try:
            import psutil
            total_gb = psutil.virtual_memory().total / 1e9
        except ImportError:
            print("(提示：安装 psutil 可显示统一内存总量 —— pip install psutil)")
            total_gb = None

        if total_gb is not None:
            print(f"Unified memory: {total_gb:.1f} GB（CPU 与 GPU 共享）")

        print("\n=== CPU vs GPU Benchmark ===\n")
        benchmark("mps")

        print(f"\nPyTorch 已分配 MPS 内存: "
              f"{torch.mps.current_allocated_memory() / 1e9:.2f} GB")

        if total_gb is not None:
            params_billions = total_gb * 1e9 / 2 / 1e9
            print(f"Estimated max model size (fp16): "
                  f"~{params_billions:.0f}B parameters（受统一内存限制）")
        return

    # ---------- 分支 3：仅 CPU ----------
    print("\nNo GPU detected. That's fine for most lessons.")
    print("For GPU-heavy lessons, use Google Colab (free).")
    benchmark("cpu")


if __name__ == "__main__":
    check_gpu()
