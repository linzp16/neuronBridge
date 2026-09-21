# NeuronBridge Linux GPU 全功能移植方案

## 1. 文档目的

本文档是 NeuronBridge Linux/Ubuntu 移植工作的长期实施依据，用于约束后续设计、代码修改、打包和测试。

核心目标不是构建一个只包含 GPU 神经元的精简版本，而是让 Linux wheel 与 Windows wheel 在功能上完全一致。Linux 采用 CUDA-enabled 完整构建，同时保留普通 CPU 神经元、OpenMP、Dense CPU/GPU、事件队列、学习规则、通信、OuterDynamic、DebugMonitor 和 Python API。

本项目不计划发布 Linux CPU-only wheel，但这不意味着删除或禁用 CPU 神经元功能。

## 当前实施状态（2026-09-21）

已完成第一批跨平台基础改造：

- 顶层 CMake 已按 Windows/Linux 分派平台配置；
- 已新增 `cmake/platforms/LinuxGNU.cmake`；
- 已启用静态库 PIC；
- Windows 专用宏已移出公共编译定义；
- ZeroMQ imported target 已区分 Windows `.lib/.dll` 与 Linux `.so`；
- 已增加 Linux Threads、`dl`、GCC/Clang 和 CUDA/OpenMP 配置；
- CUDA runtime 已支持 `AUTO/STATIC/SHARED`；
- Python CMake 已区分 Windows DLL 和 Linux ELF/RPATH；
- wheel 检查器已支持 Windows 和 manylinux 布局；
- 已新增 Linux wheel 构建、auditwheel 修复和 ELF 运行时检查脚本；
- 已新增 Linux source CI 和 Linux CUDA release workflow；
- 已新增 include 路径大小写检查；
- Windows 完整 `_core` 目标已重新配置并成功构建。
- Windows 完整 wheel 已重新生成并通过两种 Matplotlib/NeuronBridge 导入顺序；
- 当前筛选回归结果为 `55 passed, 22 deselected`。

尚未完成：

- Linux 第三方依赖 bundle 的实际构建和 SHA256 manifest；
- Ubuntu/CUDA 主机上的第一次原生编译；
- 根据 GCC/NVCC 诊断进行的源码级兼容修复；
- Linux wheel 的实际 auditwheel repair；
- Windows/Linux 完整数值一致性测试。

当前 Windows 主机的 WSL 枚举受到系统权限限制，因此 Linux 实编译必须在 Ubuntu/CUDA 主机或配置好的 Linux CI runner 上继续。

## 2. 目标产物与范围

目标产物：

```text
Windows: neuronbridge-<version>-cp312-cp312-win_amd64.whl
Linux:   neuronbridge-<version>-cp312-cp312-manylinux_<tag>_x86_64.whl
```

两个 wheel 必须具有相同的：

- Python API；
- 神经元模型 catalog；
- learning rule catalog；
- 网络定义结构；
- 权重文件格式；
- DebugMonitor 输出格式；
- ZMQ 消息协议；
- reset/control 消息；
- CPU、Dense 和 GPU 混合建模能力。

Linux 发布构建固定启用：

```text
NR_ENABLE_CUDA=ON
NR_ENABLE_PYTHON=ON
NR_ENABLE_MODEL_CODEGEN=ON
```

## 3. 顶层 CMake 修改

### 3.1 `CMakeLists.txt`

将无条件加载 Windows 平台配置：

```cmake
include(platforms/WindowsMSVC)
```

改为平台分派：

```cmake
if(WIN32)
  include(platforms/WindowsMSVC)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  include(platforms/LinuxGNU)
else()
  message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()
```

全局启用 PIC：

```cmake
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
```

CUDA 编译器错误信息不得再限定 Visual Studio。配置阶段还应打印：

- 目标平台；
- C++ 编译器及版本；
- CUDA 编译器及版本；
- CUDA architectures；
- CUDA runtime linkage；
- 依赖模式和依赖根目录。

当 `NR_CUDA_ARCHITECTURES` 非空时，将其写入 `CMAKE_CUDA_ARCHITECTURES`。

### 3.2 新增 `cmake/platforms/LinuxGNU.cmake`

该文件负责：

- 限制目标为 Linux x86-64；
- 支持 GCC 或 Clang；
- 检查最低编译器版本；
- 查找 `Threads`；
- 检查 NVCC 与 host compiler 的兼容性；
- 保证 C++17；
- 不定义任何 Windows 宏；
- 输出 glibc、libstdc++ ABI 和编译器信息。

建议框架：

```cmake
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "LinuxGNU.cmake requires Linux")
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "NeuronBridge requires a 64-bit build")
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR "Linux requires GCC or Clang")
endif()

find_package(Threads REQUIRED)
```

Windows 的 `cmake/platforms/WindowsMSVC.cmake` 保持 Windows x64/MSVC 检查职责，不再影响 Linux 配置。

## 4. 构建选项修改

### 4.1 `cmake/NeuronBridgeOptions.cmake`

新增：

```cmake
set(NR_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures included in the build")

set(NR_CUDA_RUNTIME_LINKAGE "STATIC" CACHE STRING
    "CUDA runtime linkage: STATIC or SHARED")
set_property(CACHE NR_CUDA_RUNTIME_LINKAGE PROPERTY STRINGS STATIC SHARED)
```

默认留空以保持现有 Windows 构建行为；Linux 发布脚本显式传入
`52;61;75;80;86;89`。其中 6.1 覆盖当前 GTX 1050，5.2 保持现有
Windows CUDA 构建基线。Linux 不新增功能裁剪选项。Windows 和 Linux
必须构建相同的核心 target 和功能源码。

## 5. 依赖系统修改

### 5.1 `cmake/NeuronBridgeDependencies.cmake`

将固定的 `windows-x64-msvc` 平台 key 改为平台相关默认值：

```cmake
if(WIN32)
  set(_nr_default_platform_key "windows-x64-msvc")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(_nr_default_platform_key "linux-x86_64-gcc-cuda12")
endif()
```

以下宏只能在 Windows 分支定义：

```text
_WIN32_WINNT
WINVER
WIN32
WIN32_LEAN_AND_MEAN
NOMINMAX
_CRT_SECURE_NO_WARNINGS
```

Linux 需要链接：

```cmake
Threads::Threads
${CMAKE_DL_LIBS}
```

#### ZeroMQ

Windows 保留 `.lib + .dll` imported target。Linux 优先使用标准 CMake target，否则使用 pkg-config：

```cmake
find_package(ZeroMQ CONFIG QUIET)

if(NOT TARGET libzmq)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(ZeroMQ REQUIRED IMPORTED_TARGET libzmq)
endif()

find_path(CPPZMQ_INCLUDE_DIR zmq.hpp REQUIRED)
```

Linux 不得设置 `IMPORTED_IMPLIB`，也不得引用 `.dll`。

#### Pinocchio

两平台均通过：

```cmake
find_package(pinocchio CONFIG REQUIRED)
```

并链接 `pinocchio::pinocchio`，公共逻辑不得假设依赖位于 Windows `Library/bin`。

### 5.2 `dependencies/manifest.json`

新增 Linux bundle：

```json
{
  "bundle_id": "neuronbridge-deps-linux-x86_64-gcc-cuda12-r1",
  "platform_key": "linux-x86_64-gcc-cuda12",
  "architecture": "x86_64",
  "compiler_abi": "gcc-cxx11",
  "revision": 1
}
```

依赖版本应与 Windows 对齐：

- Pinocchio 4.0.0；
- Boost 1.88.0；
- Eigen 3.4.0；
- ZeroMQ 4.3.5；
- 对应版本的 cppzmq；
- coal/hpp-fcl；
- urdfdom；
- sdformat/gz 系列。

### 5.3 新增 Linux 依赖脚本

新增：

```text
scripts/bootstrap_dependencies_linux.sh
scripts/package_dependencies_linux.sh
docker/linux-cuda-wheel/Dockerfile
```

要求：

- 在固定 Docker/manylinux 环境中从源码构建依赖；
- 全部依赖使用相同 GCC 和 C++ ABI；
- 记录源码版本、编译参数和 SHA256；
- 不直接复制开发机 `/usr/lib`；
- 在公开分发前完成第三方许可证审核。

## 6. 原生核心构建修改

### 6.1 `cmake/NeuronBridgeNativeCore.cmake`

Windows 和 Linux 必须共同构建：

```text
neuronbridge_core_cpp
neuronbridge_core_cuda
neuronbridge_dense_runtime
neuronbridge_runtime_support
neuronbridge_runtime
```

不能根据平台删除普通 CPU 神经元、Dense、OuterDynamic、通信、事件队列或 DebugMonitor 源码。

平台宏改为条件定义：

```cmake
if(WIN32)
  target_compile_definitions(neuronbridge_native_settings INTERFACE
      _CRT_SECURE_NO_WARNINGS
      NOMINMAX
      WIN32=1
      _WIN32_WINNT=0x0A00
      WIN32_LEAN_AND_MEAN)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_compile_definitions(neuronbridge_native_settings INTERFACE
      NR_PLATFORM_LINUX=1)
endif()
```

以下功能宏在两个平台保持一致：

```text
SNN_WITH_ZMQ=1
SNN_WITH_PINOCCHIO=1
NPGR_ENABLE_CUDA=1
NR_ENABLE_MODEL_CODEGEN=1
```

Linux 编译参数：

```cmake
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(neuronbridge_native_settings INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
endif()
```

OpenMP 的主要编译和链接参数继续由 `OpenMP::OpenMP_CXX` 提供。

### 6.2 CUDA runtime

新增可配置 target：

```cmake
if(NR_CUDA_RUNTIME_LINKAGE STREQUAL "STATIC")
  set(NR_CUDART_TARGET CUDA::cudart_static)
else()
  set(NR_CUDART_TARGET CUDA::cudart)
endif()
```

所有 CUDA target 统一链接：

```cmake
${NR_CUDART_TARGET}
CUDA::cuda_driver
```

`libcuda.so.1` 必须由宿主 NVIDIA Driver 提供，不能打包进入 wheel。

为 `neuronbridge_core_cuda` 和 `neuronbridge_dense_runtime` 设置相同的 `CUDA_ARCHITECTURES`。

## 7. Python 扩展构建

### 7.1 `python/CMakeLists.txt`

所有以下 Windows 行为必须置于 `if(WIN32)`：

- 复制 `.dll`；
- 查找 `cudart64_*.dll`；
- 安装 MSVC/OpenMP runtime；
- 安装 Pinocchio、Boost、ZeroMQ DLL。

Linux 仅安装扩展和 Python 包：

```cmake
install(TARGETS neuronbridge_python_core
    LIBRARY DESTINATION neuronbridge)
```

Linux target 设置：

```cmake
set_target_properties(neuronbridge_python_core PROPERTIES
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH OFF)
```

Linux 生成文件应为：

```text
neuronbridge/_core.cpython-312-x86_64-linux-gnu.so
```

Linux 共享依赖由 `auditwheel` 分析和收集，CMake 不手工猜测 `.so` 文件名。

## 8. Python 元数据

### 8.1 `python/pyproject.toml`

增加：

```toml
"Operating System :: POSIX :: Linux",
```

开发依赖改为条件依赖：

```toml
"delvewheel>=1.9; sys_platform == 'win32'",
"auditwheel>=6; sys_platform == 'linux'",
```

可增加 cibuildwheel 配置：

```toml
[tool.cibuildwheel]
build = "cp312-*"
skip = "*-musllinux_*"

[tool.cibuildwheel.linux]
archs = ["x86_64"]
```

两个平台使用相同包名、版本和 Python API。

## 9. wheel 构建与修复

### 9.1 Windows

继续使用 `scripts/build_neuronbridge_wheel.ps1`：

- MSVC 构建；
- delvewheel 修复；
- MSVC/OpenMP runtime 私有命名；
- PE 依赖表修补。

`scripts/private_runtime_wheel.py` 增加平台保护：

```python
if sys.platform != "win32":
    raise RuntimeError("private_runtime_wheel.py is Windows-only")
```

### 9.2 Linux

新增 `scripts/build_neuronbridge_wheel.sh`，主要流程：

```bash
python -m build python \
  --wheel \
  --no-isolation \
  --outdir artifacts/wheel-linux/raw

auditwheel show artifacts/wheel-linux/raw/*.whl

auditwheel repair \
  artifacts/wheel-linux/raw/*.whl \
  --plat manylinux_2_28_x86_64 \
  --wheel-dir artifacts/wheel-linux
```

构建参数至少包含：

```text
-DNR_ENABLE_CUDA=ON
-DNR_ENABLE_PYTHON=ON
-DNR_ENABLE_MODEL_CODEGEN=ON
-DNR_BUILD_TESTS=OFF
-DNR_DEPENDENCY_MODE=BUNDLED
-DNR_CUDA_RUNTIME_LINKAGE=STATIC
-DNR_CUDA_ARCHITECTURES=52;61;75;80;86;89
```

manylinux tag 最终以实际构建容器和 `auditwheel show` 结果为准，不允许手工伪造平台 tag。

## 10. wheel 检查工具

### 10.1 `scripts/inspect_neuronbridge_wheel.py`

根据 wheel tag 分平台检查。

Windows 检查：

```text
_core*.pyd
cudart64*.dll
pinocchio*.dll
libzmq*.dll
```

Linux 检查：

```text
_core*.so
neuronbridge.libs/*.so*
```

Linux wheel 禁止：

- `.a`、`.o` 等构建产物；
- 构建目录绝对路径；
- 绝对 RPATH；
- 将 `libcuda.so.1` 打包进 wheel；
- 未解析的动态库依赖。

静态链接 cudart 时，不要求 wheel 内存在 `libcudart.so`。

### 10.2 新增 `scripts/check_linux_wheel_runtime.py`

执行并解析：

```text
auditwheel show
readelf -d
ldd
```

检查：

- `ldd` 不出现 `not found`；
- RPATH 不指向源码或构建目录；
- `libcuda.so.1` 保持外部依赖；
- Pinocchio、Boost、ZeroMQ 依赖已经正确处理；
- `_core.so` 能在干净 Ubuntu 环境导入。

## 11. 发布测试修改

### 11.1 `tests/release/test_release_tools.py`

将 Windows wheel 名称、`.pyd` 和 `.dll` 断言参数化。

Windows fixture：

```text
_core.cp312-win_amd64.pyd
cudart64_12.dll
pinocchio_default.dll
libzmq*.dll
```

Linux fixture：

```text
_core.cpython-312-x86_64-linux-gnu.so
neuronbridge.libs/libpinocchio*.so
neuronbridge.libs/libzmq*.so
```

增加以下测试：

- Linux wheel inspector 接受合法 `.so` 布局；
- 拒绝绝对 RPATH；
- 拒绝打包 `libcuda.so.1`；
- Windows runtime 私有化检查只作用于 Windows wheel；
- 两个平台都包含相同 Python 模块和元数据。

## 12. 跨平台源码检查

新增 `scripts/check_include_case.py`，验证所有本地 `#include` 的路径和文件名大小写与仓库完全一致。Linux 文件系统区分大小写，而 Windows 通常不会暴露此问题。

首轮 GCC/NVCC 编译后，只根据实际错误修复：

- 缺少标准头文件；
- MSVC 接受但 GCC 拒绝的非标准语法；
- 隐式类型转换；
- signed/unsigned 比较；
- 模板实例化差异；
- CUDA host/device 限定；
- 静态库链接顺序；
- OpenMP data-sharing 声明；
- 文件名和 include 大小写。

所有源码修复必须同时回归 Windows，不允许形成 Linux 独立的神经动力学或学习算法实现。

## 13. CI 与发布流程

新增 `.github/workflows/linux-cuda-release.yml`，使用带 NVIDIA GPU 的自托管 Linux runner：

```yaml
runs-on: [self-hosted, Linux, X64, neuronbridge-cuda]
```

流水线顺序：

1. 检查 NVIDIA Driver；
2. 检查 CUDA Toolkit；
3. 恢复或构建 Linux 依赖；
4. CMake configure；
5. 完整编译；
6. CTest；
7. 构建 raw wheel；
8. auditwheel repair；
9. 在干净 Ubuntu 容器安装；
10. 执行完整功能测试；
11. 上传 wheel、测试报告和 SHA256。

现有 `.github/workflows/ci.yml` 增加无 GPU 的 Linux 静态检查，但不构建 CPU-only wheel：

- Python API 测试；
- model codegen 测试；
- include 大小写检查；
- Linux 构建脚本语法检查；
- 平台无关发布工具测试。

完整 native build 只在 CUDA runner 上运行。

## 14. 跨平台功能一致性测试

新增：

```text
tests/platform_parity/cases.json
tests/platform_parity/run_platform_matrix.py
tests/platform_parity/compare_platform_results.py
```

Windows 和 Linux 使用相同网络、随机种子、初始权重、输入和仿真步数，覆盖：

| 功能 | Windows | Linux | 主要比较项 |
|---|---:|---:|---|
| 普通 CPU 神经元 | 必测 | 必测 | state、spike |
| OpenMP 1/2/4 线程 | 必测 | 必测 | 准确性和事件顺序 |
| Heap | 必测 | 必测 | 事件顺序 |
| TimeWheel | 必测 | 必测 | 事件顺序 |
| Dense CPU | 必测 | 必测 | state、spike |
| Dense GPU | 必测 | 必测 | state、spike |
| 普通 GPU | 必测 | 必测 | 数值误差 |
| CPU/Dense/GPU 混合网络 | 必测 | 必测 | 传播和学习 |
| STDP/R-STDP | 必测 | 必测 | 权重轨迹 |
| DebugMonitor | 必测 | 必测 | state/spike 文件 |
| OuterDynamic | 必测 | 必测 | 状态和反馈 |
| ZMQ 同步/异步 | 必测 | 必测 | 协议和时序 |
| reset/control 消息 | 必测 | 必测 | 生命周期 |
| 权重跨平台互读 | 必测 | 必测 | 格式和数值 |

建议容差：

- 整数、事件 ID、消息结构：完全一致；
- 单线程 CPU state：`rtol <= 1e-6`；
- GPU state：`rtol <= 1e-5`；
- 学习权重：`rtol <= 1e-5`；
- 阈值附近 spike：最多允许一个仿真步差异；
- 非阈值边界案例：spike 序列应完全一致；
- 权重文件和通信帧结构：格式一致并支持跨平台互读。

## 15. 实施阶段

### 阶段 A：Linux 完整源码构建

- 完成平台 CMake 分支；
- 修正 Windows 宏；
- 建立 Linux 依赖；
- 完整编译 CPU、Dense、GPU、通信和 OuterDynamic；
- 完成首轮 GCC/NVCC 兼容性修复。

### 阶段 B：Linux wheel

- 构建 raw wheel；
- auditwheel 修复；
- 检查 ELF、RPATH 和动态依赖；
- 在干净 Ubuntu 容器安装和导入。

### 阶段 C：完整功能回归

- 执行当前 Windows 测试矩阵的 Linux 对等版本；
- 完成 OpenMP、Heap、TimeWheel、Dense、GPU、DebugMonitor、通信和 OuterDynamic 测试；
- 修复平台差异。

### 阶段 D：跨平台数值一致性

- 固定测试输入和随机种子；
- Windows/Linux 分别生成结果；
- 自动比较 state、spike、权重和通信输出；
- 形成最终兼容性报告。

## 16. 完成标准

Linux wheel 只有同时满足以下条件才算完成：

- 在目标 Ubuntu 干净环境中可以安装；
- 无需用户设置 `LD_LIBRARY_PATH`；
- 普通 CPU、Dense CPU/GPU、普通 GPU 功能与 Windows 一致；
- OpenMP、Heap、TimeWheel、学习规则全部通过；
- ZMQ、OuterDynamic、DebugMonitor 全部通过；
- Linux 能读取 Windows 保存的权重，Windows 也能读取 Linux 权重；
- `ldd` 不存在 `not found`；
- `auditwheel show` 通过；
- wheel 不包含 `libcuda.so.1`；
- Windows 现有回归测试继续通过；
- 跨平台数值结果满足本文定义的容差；
- 没有为了 Linux 引入功能裁剪或独立算法分支。

## 17. 长期约束

后续任何 Linux 相关修改必须遵守：

1. Linux 是 CUDA-enabled 完整版本，不是 GPU-only 功能版本；
2. 不发布 Linux CPU-only wheel，但必须保留 CPU 神经元和 CPU 事件调度；
3. 平台差异仅限编译、链接、依赖查找和二进制打包；
4. 网络模型、学习算法、通信协议和文件格式必须跨平台共享；
5. 新功能必须同时加入 Windows/Linux 测试矩阵；
6. 任何跨平台数值差异都必须有明确容差、原因和测试报告。
