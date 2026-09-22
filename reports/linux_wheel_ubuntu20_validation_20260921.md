# neuronBridge Ubuntu 20.04 Linux wheel 构建与验证报告

- 日期：2026-09-21
- 目标系统：Ubuntu 20.04 x86-64（glibc 2.31）
- Python ABI：CPython 3.12
- CUDA：12.6；实际验证 GPU 为 NVIDIA GeForce GTX 1050
- Wheel 标签：`manylinux_2_31_x86_64`

## 构建产物

- Wheel：`D:\neuronBridge\artifacts\wheel-linux-ubuntu20\neuronbridge-0.1.0a0-cp312-cp312-manylinux_2_31_x86_64.whl`
- 大小：4,519,413 bytes
- SHA-256：`7E6BFD6EB91D5DCBD9B4D53291B6919422554C2F73B9D5A258F1F00B9F40B734`

该 wheel 是 CUDA-enabled 完整版本，同时包含普通 CPU 神经元、OpenMP、Dense CPU/GPU、事件队列、学习规则、通信、OuterDynamic、DebugMonitor 和流式构建能力。`libcuda.so.1` 由宿主 NVIDIA Driver 提供，未打入 wheel。

## 本轮 Linux 兼容修复

1. Pinocchio 传播 `BOOST_MPL_LIMIT_*_SIZE=30` 时，Linux Boost 的默认预处理 MPL 头仍只有 20 项，导致流式解析器使用 PropertyTree 时编译失败。Linux 构建增加 `BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS`，最小复现和完整编译均通过。
2. Linux 改用 Pinocchio 官方 `pinocchio::pinocchio_headers` target，由 NeuronBridge 本地实例化实际使用的刚体算法，不再无条件链接未使用的 parser、collision、URDF 等共享库。
3. Linux 链接启用 `--as-needed`，去除未使用的传递依赖。
4. Linux 最低 GCC 调整为 9，以支持 Ubuntu 20.04 原生 GCC 9.4；NVCC 显式使用该 host compiler。
5. ZeroMQ 使用 Ubuntu 20.04 官方 4.3.2 构建，避免 conda ZeroMQ 带入高于 Ubuntu 20.04 的 `GLIBCXX` 要求。

## 验证结果

| 验证项 | 结果 |
|---|---:|
| C++/CUDA/dense/流式解析器完整编译 | 通过 |
| auditwheel ABI | `manylinux_2_31_x86_64`，通过 |
| wheel 内容检查 | 通过；28 个条目，12 个 ELF shared objects |
| RPATH | 通过；仅使用 `$ORIGIN` 相对路径 |
| ldd | 通过；无 `not found` |
| NVIDIA driver 打包检查 | 通过；未包含 `libcuda.so.1` |
| 全新 Python 3.12 venv 安装与导入 | 通过 |
| CPU / 普通 GPU / dense GPU | 通过 |
| 核心 Python 回归 | 102 passed，18 deselected，5.99 s |
| 同步 REQ/REP + heap | 通过 |
| 同步 REQ/REP + timing wheel | 通过 |

核心回归按既定范围排除了 `handwriting` 与 `ei_connectivity` 相关用例。暂缓开放的实时实验接口未纳入发布门禁。

## 证据文件

- Wheel 内容：`D:\neuronBridge\artifacts\wheel-linux-ubuntu20\wheel-contents.json`
- ELF/运行时依赖：`D:\neuronBridge\artifacts\wheel-linux-ubuntu20\runtime-dependencies.json`
- 三后端结果：`D:\neuronBridge\artifacts\wheel-linux-ubuntu20\triple-backend.json`
- JUnit：`D:\neuronBridge\artifacts\wheel-linux-ubuntu20\pytest-linux.xml`

## 结论

Ubuntu 20.04 对应的 Linux CUDA wheel 已成功构建，并通过 ABI、ELF、干净安装、CPU/GPU/dense 和核心回归验证。该产物可作为本轮 Linux 测试 wheel。
