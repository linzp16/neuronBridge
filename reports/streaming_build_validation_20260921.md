# `.nbnet` 低内存构建验证报告

日期：2026-09-21  
平台：Windows 64-bit，Python 3.12.4  
验证对象：当前源码、C++ `NbnetReader` 与完整 `streaming_direct` 构建路径

## 阶段二更新：C++ 解析与完整直接流式构建

本轮已把 header、SHA-256 checksum、metadata 和连接记录解析从 Python 搬入 C++，删除路径构造流程中的 Python `_native_description()`。pybind 的路径构造重载直接创建 `NbnetReader`。对于不含 dense、OuterDynamic 和 InputConv 的主网，运行时执行两遍扫描：第一遍统计学习规则状态，第二遍直接写入最终 `Interconnections[]`，不再生成完整 `ConnectionDescription` 中间副本。

所有 `.nbnet` 构建现在均报告 `build_stats["runtime_build_path"] == "streaming_direct"`。纯主网直接两遍扫描；混合网络通过受预算限制的 dense 分区批次、过滤/重映射主网连接源和少量合成接口连接完成，不再恢复完整连接描述。

新基准使用 4096 神经元、16 MiB staging、mmap 和 checksum：

| 规模 | fast 峰值工作集 | C++ 兼容路径峰值 | 主网两遍扫描峰值 | 相对 fast 降幅 | 两遍扫描耗时 |
|---:|---:|---:|---:|---:|---:|
| 500,000 | 304.6 MB | 174.0 MB | 133.6 MB | 56.1% | 0.184 s |
| 1,000,000 | 565.2 MB | 304.3 MB | 223.8 MB | 60.4% | 0.354 s |

dense/GPU 直接流式基准：

| 规模 | fast 峰值工作集 | `streaming_direct` 峰值 | 降幅 | 构建耗时 |
|---:|---:|---:|---:|---:|
| 500,000 | 357.7 MB | 233.7 MB | 34.7% | 0.337 s |
| 1,000,000 | 597.4 MB | 350.8 MB | 41.3% | 0.509 s |

对应原始数据：

- `artifacts/streaming_build/cpp_memory_comparison_500k.json`
- `artifacts/streaming_build/cpp_memory_comparison_1m.json`
- `artifacts/streaming_build/direct_memory_comparison_500k.json`
- `artifacts/streaming_build/direct_memory_comparison_1m.json`
- `artifacts/streaming_build/full_direct_comparison_500k.json`
- `artifacts/streaming_build/full_direct_comparison_1m.json`
- `artifacts/streaming_build/full_direct_dense_comparison_500k.json`
- `artifacts/streaming_build/full_direct_dense_comparison_1m.json`

本轮验证结果：

- `.nbnet` C++ loader、混合路由和原始索引回归：10 passed；
- Python 与现有运行时合并回归：78 passed，5 个既有环境型 skip；
- 全新 venv 安装成品 wheel 后 `.nbnet` smoke：PASS；
- wheel 内容和私有 MSVC/OpenMP runtime 检查：PASS。

成品 wheel：`artifacts/wheel_streaming_full_direct/neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`。  
SHA-256：`EA298C204105DF0890606935DE56E4A14D80F4782A70C7FCFF5EA124BAE5D814`。

dense/GPU、跨子网路由、学习规则、OuterDynamic、InputConv 和原始连接索引均已进入直接流式路径并完成成品 wheel smoke。剩余内存主要是最终运行时布局、GPU 主机镜像和用户配置的 staging 批次，而不是完整连接描述副本。

最终收口同时删除了未被运行路径调用的 `LoadNbnetDescription`/`NbnetLoadedDescription` 全量物化兼容 loader，避免以后误接回 `std::list<ConnectionDescription>`。重新构建后的核心 `.nbnet` 测试为 10 passed，合并回归为 78 passed、5 skipped、7 deselected，隔离虚拟环境中的基础与完整混合网络 wheel smoke 均为 PASS。

## 阶段一历史结果

## 结论

新增的 `.nbnet` 文件构建路径在 100 万突触测试中，将进程 Peak Working Set 从 547.5 MiB 降至 297.3 MiB，降低 45.7%；Python 跟踪分配峰值从 137.1 MiB 降至 12.7 MiB，降低 90.7%。同一小型确定性 LIF 网络分别通过内存路径与文件路径构建后，输出 spike 和连接权重一致。

当前实现是过渡阶段：文件路径已经独立于 `Network.to_native()`，但分批解析后仍进入现有 `NativeNetworkDescription` 和原生运行时构造函数。它证明了消除 Python 全量网络描述及规范化副本可以显著降低峰值内存；要达到“稳定运行内存 + staging 预算”的最终目标，还需实现 C++ 直接流式构建。

## 已实现内容

- `Simulation(Network, config)` 保持原有快速路径；
- `Simulation(path, config, build_options=StreamingBuildOptions(...))` 新增文件路径重载；
- 两条 Python 构造路径分别调用 `Network.to_native()` 和独立的 `.nbnet` loader；
- 新增 `NbnetDescriptionBuilder`，连接按批写盘，不保存在 builder 中；
- 新增 `from_network()`、`from_batches()`、`inspect()`、`validate()`、`is_valid()`；
- `.nbnet` v1 使用小端固定宽度连接记录和 SHA-256；
- 支持 `str`/`PathLike`、mmap、校验和与加载内存预算；
- 原文件默认不覆盖，生成采用同目录临时文件和原子替换；
- CMake 测试列表已加入 `neuronbridge_python_nbnet`。

## 等价性验证

测试网络包含 1 个 InputSpike 神经元、1 个 `TimeDrivenLIF_Exponential_double` 神经元和 1 条延迟突触。两种路径使用相同参数、输入 spike 和仿真配置。

检查项：

- `.nbnet` 元数据和连接数正确；
- `NativeParameter` 的 `float32`/`int32` 类型经过文件往返后保真；
- 输出 spike 列表完全相同；
- 原始连接索引 0 的权重在 `1e-7` 容差内一致；
- mmap 文件读取与 checksum 验证成功；
- 修改文件末字节后能以 checksum mismatch 安全拒绝；
- 默认拒绝覆盖已存在的 `.nbnet` 文件。

测试结果：

```text
python/tests/test_api.py
python/tests/test_catalog.py
python/tests/test_native_smoke.py
python/tests/test_nbnet.py
15 passed

tests/python/test_migrated_examples.py
64 passed, 5 skipped
```

skip 为原测试套件中依赖可选环境/功能的既有跳过项。本轮唯一 warning 是 pytest 无法写入仓库内缓存目录，不影响测试内容。

另外，完整 CUDA Windows wheel 已重新构建，并安装到不带源码 `PYTHONPATH` 的全新虚拟环境：

- wheel：`artifacts/wheel_streaming_build/neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- SHA-256：`17FBAC0D0CF10ED253F82A209FF80985BA5B877828AB764D8FC512FEC4019FBB`
- wheel 内容与私有 MSVC/OpenMP runtime 检查：PASS；
- 已安装 wheel 的 CUDA/runtime smoke：PASS；
- 已安装 wheel 的 `.nbnet` 生成、校验、双路径运行和 spike/weight 等价性 smoke：PASS；
- 已安装 wheel 的 50 万突触独立进程基准再次得到 41.5% Peak Working Set 降幅，与构建树结果一致；
- 实际导入位置：`build/wheel-streaming-validation/venv/Lib/site-packages/neuronbridge/__init__.py`。

## 峰值内存对比

基准使用两个全新 Python 进程，避免前一次构建的 allocator/cache 影响下一模式。Windows Peak Working Set 通过 `GetProcessMemoryInfo` 读取；Python 分配峰值通过 `tracemalloc` 读取。`.nbnet` 预生成不计入 Simulation 加载内存，因为它是可离线复用的静态网络描述文件。

网络由等规模 InputSpike 和 LIF 层组成，连接采用确定性映射，两个模式构建完全相同的拓扑。streaming staging 预算设为 16 MiB，实际每批 65,536 条连接。

| 规模 | 路径 | Peak Working Set | Python 分配峰值 | 构建总耗时 |
|---:|---|---:|---:|---:|
| 50 万突触 / 4,096 神经元 | 现有内存路径 | 290.5 MiB | 67.4 MiB | 1.31 s |
| 50 万突触 / 4,096 神经元 | `.nbnet` 路径 | 170.0 MiB | 12.6 MiB | 10.17 s |
| 100 万突触 / 8,192 神经元 | 现有内存路径 | 547.5 MiB | 137.1 MiB | 2.69 s |
| 100 万突触 / 8,192 神经元 | `.nbnet` 路径 | 297.3 MiB | 12.7 MiB | 20.95 s |

结果：

- 50 万突触：进程峰值下降 41.5%，Python 峰值下降 81.3%；
- 100 万突触：进程峰值下降 45.7%，Python 峰值下降 90.7%；
- 网络规模翻倍时，streaming 的 Python 峰值基本稳定，说明批次上限已生效；
- 当前 streaming 路径明显更慢，主要耗时来自 Python 对固定宽度记录的逐条拆列和 pybind11 批次转换。

原始结果：

- `artifacts/streaming_build/memory_comparison.json`
- `artifacts/streaming_build/memory_comparison_1m.json`
- `artifacts/streaming_build/installed_wheel_memory_comparison_500k.json`

复现命令：

```powershell
$env:PYTHONPATH='D:\neuronBridge\build\runtime-safety-wheel\python\Release'
python tests\streaming_build\benchmark_streaming_memory.py `
  --connections 1000000 `
  --neurons-per-layer 4096 `
  --budget-mb 16 `
  --path artifacts\streaming_build\memory_network_1m.nbnet `
  --output artifacts\streaming_build\memory_comparison_1m.json
```

## 尚未完成与下一步

当前数据只证明“首阶段文件构建路径”比现有 Python 全内存描述路径显著节省内存，不代表最终原生 streaming 架构已经完成。后续优先级如下：

1. 将固定记录批量解码移入 C++，消除当前约 7.8 倍构建耗时代价；
2. C++ 第一遍只统计和规划，第二遍直接填充最终主网结构，不再保留完整 `NativeNetworkDescription.connections`；
3. 为 dense/GPU、main/dense 边界连接、学习规则和原始连接索引增加独立 section；
4. 对 Heap/TimingWheel、OpenMP、dense/GPU、OuterDynamic、InputConv、DebugMonitor、reset 和通信逐项做双路径等价性测试；
5. 在 Ubuntu 20.04 对同一 `.nbnet` 做字节级兼容和内存基准。
