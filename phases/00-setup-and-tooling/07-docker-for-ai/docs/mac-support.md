# Mac 平台 Dockerfile 修改指南

## 为什么需要修改？

### CUDA 问题
- **CUDA 是 NVIDIA 专有技术**，只能在 NVIDIA GPU 上运行
- Mac（包括 Apple Silicon M1/M2/M3）**完全不支持 CUDA**
- 使用 CUDA 基础镜像在 Mac 上会导致构建失败

### Mac 的 GPU 加速选项

1. **CPU 模式**（最简单，推荐用于入门）
   - 完全移除 CUDA 依赖
   - 使用标准 Python 镜像
   - 适合学习和小型模型

2. **Metal 加速**（Apple Silicon 独有）
   - 使用 PyTorch 的 Metal 后端
   - 比 CPU 快 5-10 倍
   - 需要特殊配置

3. **Docker Desktop GPU 支持**（实验性）
   - Docker Desktop for Mac 提供有限的 GPU 加速
   - 使用 `--gpus all` 但性能有限

## 方案 1：CPU 版本（推荐入门）

创建一个新的 Dockerfile 文件：`Dockerfile.mac`

```dockerfile
# 使用标准 Python 镜像，无 CUDA
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装 CPU 版本的 PyTorch
# 注意：移除了 --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1

# 安装 AI 库
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    transformers \
    datasets \
    accelerate \
    safetensors

WORKDIR /workspace

VOLUME ["/workspace", "/models"]

EXPOSE 8888

CMD ["python"]
```

**构建和使用：**
```bash
# 构建
docker build -t ai-dev-mac -f Dockerfile.mac .

# 运行
docker run --rm -it \
    -v $(pwd):/workspace \
    -v ~/models:/models \
    ai-dev-mac python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**关键区别：**
- ✅ 使用 `python:3.12-slim` 而非 `nvidia/cuda` 镜像
- ✅ 安装 CPU 版本的 PyTorch（移除了 CUDA 索引）
- ✅ 移除了 `--gpus all` 参数

## 方案 2：Metal 加速版（Apple Silicon M1/M2/M3）

如果你有 Apple Silicon Mac 并想利用 GPU 加速：

```dockerfile
# 使用 Python 基础镜像
FROM python:3.12

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装支持 Metal 的 PyTorch
# Apple Silicon 需要从 PyTorch 官方源安装
RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 启用 Metal 后端（在代码中设置）
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# 安装 AI 库
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    jupyter \
    transformers \
    datasets \
    accelerate \
    safetensors

WORKDIR /workspace

VOLUME ["/workspace", "/models"]

EXPOSE 8888

CMD ["python"]
```

**在 Python 代码中使用 Metal：**
```python
import torch

# 检查 Metal 是否可用
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using Metal GPU acceleration")
else:
    device = torch.device("cpu")
    print(f"Using CPU")

# 你的代码
model = model.to(device)
```

**限制：**
- Metal 对 PyTorch 的支持仍在改进中
- 某些 CUDA 特有的操作可能不兼容
- 性能不如原生 CUDA

## 原 Dockerfile 在 Mac 上的问题

### ❌ 这行会失败：
```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
```
**原因：** 这个镜像是为 Linux + NVIDIA GPU 设计的

### ❌ 这行也会失败：
```dockerfile
--index-url https://download.pytorch.org/whl/cu124
```
**原因：** 下载 CUDA 版本的 PyTorch

### ✅ 即使移除这些：
```dockerfile
RUN python -m pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    torchaudio==2.3.1
```
PyTorch 会**自动检测**无 CUDA 并安装 CPU 版本

## 快速验证

构建完成后，验证 PyTorch 是否正确安装：

```bash
docker run --rm -it \
    -v $(pwd):/workspace \
    ai-dev-mac python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'Metal (MPS) 可用: {torch.backends.mps.is_available()}')
print(f'设备: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")}')
"
```
