# ComfyUI DAT Attention Cache

加速 Spandrel DAT 超分模型的窗口注意力蒙版计算。插件随 ComfyUI 启动自动生效，工作流无需增加节点。

## 优化内容

DAT 会为偏移窗口生成注意力遮罩。反复在 CPU 构建相同遮罩并传到 GPU，会拖慢部分超分工作流。
本插件在模型参数所在设备上生成遮罩，并按设备、尺寸、窗口、偏移和 dtype 复用结果。

- 每个设备最多缓存 8 组变体；尺寸改变时丢弃旧尺寸缓存。
- ComfyUI 的 `ImageUpscaleWithModel` 正常结束或抛出异常后清空缓存。
- 保留 FP32、FP16、BF16 的遮罩语义；GPU 取决于模型实际所在设备。
- 运行时打补丁，不改 ComfyUI 核心或 `site-packages` 文件。

它可以用于调用 ComfyUI 标准模型超分节点的工作流，包括 Ultimate SD Upscale。
`4x-UltraSharpV2` 是已验证的 DAT 模型示例；文件名相似不代表底层架构相同。
这项优化针对 DAT 内部注意力遮罩，与图像编辑蒙版、羽化拼接、扩散采样器和 SageAttention 是不同的路径。

## 安装

使用 ComfyUI 自己的 Python 环境和已有的 PyTorch、Spandrel 依赖。进入 `ComfyUI/custom_nodes` 后执行：

```console
git clone https://github.com/landian01/ComfyUI-DAT-Attention-Cache.git
```

重启 ComfyUI。日志应包含：

```text
DAT attention-mask cache patch 1.1.0 enabled (DAT=True, cleanup=True).
```

节点列表里不会新增节点。卸载时移除插件目录并重启即可。

## 可选 C++ / CUDA 蒙版内核

1.1.0 增加专用蒙版生成算子。它直接写出两个方向的 `0/-100` 注意力蒙版，减少中间张量和 kernel launch。
**现有缓存键、LRU 容量、尺寸失效和任务结束清理逻辑保留。** 算子只处理缓存未命中的情况。

普通安装继续使用已有 PyTorch 实现。插件不会在 ComfyUI 启动时下载工具或自动编译。
需要启用时，使用 ComfyUI 自己的 Python，在插件目录运行：

```powershell
python build_native.py --device cuda:0 --cuda-root 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0'
```

将 `python` 换成 ComfyUI 的 Python 路径，将 CUDA 路径换成与 `torch.version.cuda` 一致的 Toolkit。
当前构建脚本支持 Windows x64，需要 MSVC x64 C++ Build Tools 和匹配的 CUDA Toolkit；不需要 Python 开发头文件或 Ninja。
编译器环境仅在构建子进程中设置，不改系统 PATH。成功后重启 ComfyUI，首次生成蒙版时日志会显示：

```text
DAT native CUDA mask backend enabled for SM 120.
```

本地编译产物放在 `_native/`，不进入 Git。加载前核对 PyTorch/CUDA 版本、平台、GPU 架构和内核源码哈希。
没有产物、版本不匹配、加载失败、CPU、其他 GPU 架构或不支持的窗口配置均回退 PyTorch。
更新 PyTorch、CUDA 或内核源码后需重新构建；构建只针对所选 GPU 的架构。
原生算子支持 FP32、FP16、BF16，窗口面积不超过 1024；已实测 Windows RTX 5080 / SM 120，未验证 RTX 2070 或 Linux 原生构建。
运行中的 CUDA 错误会抛出，避免在设备状态异常后继续计算。

临时关闭原生后端、保留缓存，可在启动 ComfyUI 前设置：

```powershell
$env:DAT_CACHE_NATIVE = '0'
```

已有缓存使实际超分只剩少量蒙版构建，**专用内核对整段超分的收益很小**，通常没有必要为此单独安装完整编译工具链。
本次基准与数值/内存检查记录见 [`benchmarks/cuda-rtx5080-20261017.md`](benchmarks/cuda-rtx5080-20261017.md)。

## 兼容性

已验证：ComfyUI 0.41.1、Spandrel 0.4.2、Python 3.13.16、PyTorch 2.14.1+cu130、Windows、RTX 5080。
测试也覆盖 CPU 上的遮罩构建。其他组合需要自行验证。

当前接入点要求 `ImageUpscaleWithModel.execute` 是 classmethod，且 DAT 提供
`calculate_mask(self, H, W[, dtype])`。接入不兼容时会记录错误，并将 `PATCH_ACTIVE` 设为 `False`。
1.0.2 先检查并安装清理钩子，避免旧版 ComfyUI 缺少接入点时留下未清理的蒙版缓存。
如果检测到 DAT 提供 `get_runtime_mask`，插件优先使用该原生实现；这一分支有接口模拟测试，尚无真实版本的兼容性结论。

## 性能记录

2026-10-16 的本地对照：RTX 5080、Spandrel 0.4.2、PyTorch 2.14.1+cu130，
输入 1920×960，`4x-UltraSharpV2.safetensors`，输出 7680×3840，仅测模型超分阶段。

| 指标 | 原实现 | 缓存优化 |
|---|---:|---:|
| 总耗时 | 46.97 秒 | 12.46 秒 |
| 蒙版构建累计耗时 | 34.36 秒 | 0.008 秒 |
| PyTorch 峰值已分配显存 | 4.22 GiB | 4.47 GiB |

两次输出逐像素完全相同，最大绝对误差为 0。该次测量使用插件 1.0.1，先运行原实现再运行优化实现，
不是冷启动条件完全相同的多轮基准；1.0.2 的正常蒙版计算路径相同，调整了失败时的安装顺序。
记录中的 `output_device=cpu` 是 ComfyUI 最终输出张量的设备，不表示 DAT 模型推理在 CPU 上运行。
此记录不代表完整工作流或所有 DAT 模型都有 3.77 倍加速。原始数值见 [`benchmarks/rtx5080-20261016.json`](benchmarks/rtx5080-20261016.json)。

## 验证

在已安装 PyTorch、torchvision、Spandrel 的环境中：

```console
python -m unittest discover -s tests -v
```

CPU 测试覆盖数值一致性、缓存键、容量和失效、成功/异常清理、重复安装及不兼容接口。
CUDA 一致性和真实 ComfyUI 接入测试可按需开启（PowerShell 示例）：

```powershell
$env:DAT_CACHE_TEST_CUDA = '1'
$env:COMFYUI_ROOT = 'D:/ComfyUI/ComfyUI'
python -m unittest discover -s tests -v
```

真实接入测试在独立子进程中运行，使用合成数据和轻量模型替身验证 ComfyUI 节点的调用与清理，不加载超分权重。
构建原生后端后，另设 `$env:DAT_CACHE_TEST_NATIVE = '1'` 可运行 33 组原生蒙版对照、非默认 CUDA 流、缓存复用及参数校验。
GitHub 的 CPU CI 只验证缓存和后端选择/回退，不能证明 CUDA 内核或真实模型性能。

## 来源与许可证

Apache-2.0。蒙版构造适配自 [Spandrel](https://github.com/chaiNNer-org/spandrel) 的
[DAT](https://github.com/zhengchen1999/DAT) 实现，其窗口蒙版逻辑源自
[Swin Transformer](https://github.com/microsoft/Swin-Transformer)。详细归属见 [`NOTICE`](NOTICE) 和 `licenses/`。

## English summary

A runtime extension that caches DAT shifted-window attention masks on the model's device.
It clears cached masks when ComfyUI's `ImageUpscaleWithModel` returns or raises.
Install by cloning into `custom_nodes` and restart ComfyUI; no workflow nodes are added.
The measured speedup above applies to one learned super-resolution stage, not to a complete diffusion workflow.
An optional, explicitly built Windows C++/CUDA operator accelerates cache misses while retaining the same cache.
It falls back to PyTorch when absent or incompatible. The additional measured full-stage speedup is small;
see the separate CUDA benchmark report rather than attributing the earlier cache speedup to the new kernel.
