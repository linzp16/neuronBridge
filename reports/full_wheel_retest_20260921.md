# NeuronBridge Windows wheel 全量回归报告（2026-09-21）

> 状态更新：本报告最初发现的 helper 元数据缺陷和 `AdditiveKernalChange` 构造缺陷已修复。修复后 wheel 的完整信息和验证结果见 `helper_additive_fix_validation_20260921.md`；下文原始 wheel 数据保留用于追溯。

## 1. 结论

本轮使用隔离虚拟环境重新安装发布 wheel 后执行测试。核心功能、快速构建、C++ 直接流式构建、主网、legacy GPU、dense GPU、OuterDynamic、通信、OpenMP 和绘图接口均能运行；流式构建的内存优势得到复现。

当前 wheel 确认存在 2 个明确的 API/工厂缺陷：

1. `NeuronLayer.lif_double(..., output=True, monitored=True)` 等 helper 会把结构字段错误塞入 `parameters`，实际 `layer.output == False`、`layer.monitored == False`。这会造成输出神经元/DCN 无输出或监控缺失。
2. `AdditiveKernalChange` 在学习规则目录中声明为支持，但构建阶段主网工厂报 `Unknown learning rule type: AdditiveKernalChange` 并终止，dense 路径也会在全局规则实例化阶段被阻断。

dense Izhikevich、Voltage Jump 和 Poisson 相对 CPU 的时序/状态差异来自已知的 `InterfaceNeuron` 延迟，不再作为本轮缺陷。后端比较数据仍保留，作为该延迟行为的回归基线。

除以上问题外，本轮未发现新的崩溃、非法内存访问或快速/流式构建结果分歧。

## 2. 被测对象与环境

- Wheel：`neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- SHA-256：`EA298C204105DF0890606935DE56E4A14D80F4782A70C7FCFF5EA124BAE5D814`
- Wheel 大小：4,316,937 bytes
- Python：3.12.4
- NumPy：2.4.6
- Matplotlib：3.11.2
- PyZMQ：25.1.2
- GPU：NVIDIA GeForce GTX 1050，驱动 560.94，2 GB
- 隔离环境：`D:\neuronBridge\artifacts\full_retest_20260921\venv`
- 安装来源：`D:\neuronBridge\artifacts\wheel_streaming_full_direct`
- Wheel 内容：1 个 `.pyd`、13 个随包 DLL、无 PDB；CUDA runtime、Boost serialization、ZeroMQ、Pinocchio、OpenMP/VC runtime 均随包存在。

按用户此前要求，本轮继续排除 `HandwritingTimeDrivenModel` 和 `ei_connectivity`。

## 3. 测试矩阵与结果

### 3.1 核心自动化回归

- 结果：`72 passed, 18 deselected, 1 warning`
- 覆盖：Python API、模型目录、原生 smoke、`.nbnet` header/checksum/损坏检测、快速与流式等价性、dense 跨子网路由、混合路由、OuterDynamic/InputConv metadata、学习规则原始连接索引、同步/异步 InputConv、协议错误、权重读写和绘图基础功能。
- 唯一 warning：pytest 无法写源码目录下的 `.pytest_cache`，不影响测试结果。

### 3.2 神经元模型

除 Handwriting 外，目录内模型均至少完成构建、运行、状态读取、放电或通信输入验证：

| 模型 | CPU | legacy GPU | dense GPU | 结果摘要 |
|---|---:|---:|---:|---|
| `TimeDrivenLIF_Exponential_Decay` | 通过 | 通过 | 通过 | dense 对 CPU 最大误差 `3.81e-6` |
| `TimeDrivenLIF_Exponential_double` | 通过 | 通过 | 通过 | dense 对 CPU 最大误差 `3.81e-6` |
| `TimeDrivenLIF_Voltage_jump` | 通过 | 通过 | 通过但有风险 | legacy GPU 误差 `7.45e-8`；dense 误差 `4.753125` |
| `TimeDrivenIzhikevic_Exponential_Decay` | 通过 | 通过 | 运行通过但不等价 | 见 3.3 |
| `PoissonRate` | 通过 | 通过 | 通过但提前 1 步 | 三后端均 46 次放电 |
| `TimeDrivenLIF_Exponential_triple` | 通过 | 通过 | 通过 | 三后端均能运行且状态有限 |
| `InputSpikeNeuronModel` | 通过 | 通过 | 不适用 | 同步/异步通信及外部 spike 输入通过 |
| `InputCurrentNeuronModel` | 通过 | 通过 | 不适用 | 动态电流输入和状态响应通过 |
| `TriggerRelayNeuronModel` | 通过 | 通过 | 支持矩阵确认 | CPU/GPU 各产生 4 次输出放电 |
| `EDLUTLikeLIF` | 不支持（设计） | 通过 | 不支持（设计） | legacy GPU 产生 8 次放电 |
| `CustomLifConductanceV1` | 通过 | 通过 | 通过 | 三后端构建和运行通过 |

### 3.3 数值精度与动力学

- CPU 解析/参考基线：
  - LIF Decay 最大误差：`5.08e-6`（阈值 `2e-5`）
  - LIF Double 最大误差：`5.08e-6`
  - Voltage Jump 最大误差：`3.82e-7`
  - Izhikevich 电压与恢复变量：基线误差 `0`
  - Poisson 固定随机种子：10 个预期 spike 完全一致
- Izhikevich 典型动力学：tonic spiking 38、phasic spiking 1、tonic bursting 120、phasic bursting 50、mixed mode 33，五类模式均成功产生。
- legacy GPU Izhikevich 与 CPU 接近：最差电压误差约 `0.0461`，放电次数和时刻一致。
- dense Izhikevich：
  - tonic spiking：38 vs 38，但时刻和逐点状态有偏差；最大电压误差 `100.36`
  - phasic spiking：1 vs 1，提前 1 步
  - tonic bursting：118 vs CPU 120
  - phasic bursting：41 vs CPU 50
  - mixed mode：33 vs 33，但时刻逐渐漂移

这些差异符合已知的 `InterfaceNeuron` 延迟语义。本轮将其判定为预期后端时序差异，而不是神经动力学实现缺陷。

### 3.4 学习规则

以下规则均在快速构建和流式构建各运行一次，两个构建路径最终权重完全一致：

| 规则 | 初始权重 | 最终权重 | 快速/流式 |
|---|---:|---:|---|
| `STDP` | 8 | 17.3583908 | 一致 |
| `R_STDP` | 8 | 14.1349716 | 一致 |
| `CerebullarLearningRule` | 8 | 7.1326189 | 一致 |
| `CustomPairStdpV1` | 8 | 17.3583908 | 一致 |
| `CustomRStdpV1` | 8 | 14.1349716 | 一致 |
| `CustomRStdpPersistentV1` | 8 | 20.0 | 一致 |

`AdditiveKernalChange` 未通过：无论尝试主网连接还是 dense 内部连接，全局学习规则实例化都会报 `Unknown learning rule type`。这是源码/工厂注册问题，不是测试配置问题。

### 3.5 OuterDynamic

以下模型均完成快速/流式两路径运行，状态完全一致：

- `OuterDynamicSpikeCounter`：观察到 6 个事件，计数 `[3, 3]`，加权和 `[0.3, 0.15]`
- `PlanarArm2DOFPinocchio`：2 关节状态有限，快速/流式一致
- `StrictMatlabPlanarArm2DOFOuterDynamic`：2 关节状态有限，快速/流式一致

旧 `complex_network_tests.py` 中的 OuterDynamic 子用例虽然标记 PASS，但计数为 0，原因是测试使用了有缺陷的 layer helper。报告结论以本轮新增的显式 `NeuronLayer(...)` 严格用例为准。

### 3.6 快速构建与直接流式构建

| 场景 | 快速峰值工作集 | 流式峰值工作集 | 降低 | 快速耗时 | 流式耗时 |
|---|---:|---:|---:|---:|---:|
| 主网 50 万突触 | 304.3 MB | 133.7 MB | 56.08% | 1.332 s | 0.178 s |
| 主网 100 万突触 | 565.1 MB | 223.8 MB | 60.39% | 2.727 s | 0.344 s |
| dense 50 万突触 | 356.9 MB | 233.9 MB | 34.48% | 1.456 s | 0.340 s |

三组均走 `runtime_build_path=streaming_direct`，checksum 验证成功。代表性小网络的状态、路由和权重通过 pytest 逐项对比，快速/流式一致。

### 3.7 OpenMP 与事件队列

- 单线程与 2/4 队列对比：全部一致。
- 4 个 dense 子网、共 40 个神经元、240 步：snapshot 和 113 条 spike 记录完全一致。
- 2 个 OuterDynamic counter：每槽 5 个事件，1/2/4 队列完全一致。
- 混合主网 + 2 个 dense 子网 + 2 个 OuterDynamic：heap 与 TimeWheel、1/2/4 队列均一致。
- 注意：旧脚本中的 output layer 使用 helper，因此 `output_spike_count=0` 这一字段是空验证；dense snapshot、DebugMonitor spike 与 OuterDynamic 计数仍是有效验证。

### 3.8 复杂网络、压力与通信

- 复杂混合网络：4 个场景执行通过。
- 大型主网混合案例：2,432 神经元、200 步、3,114 输出 spike。
- 大型 dense 级联：2,560 神经元、300 步，三个 dense 子网 snapshot 分别为 1,024/1,024/512。
- 压力：20 次重复构建运行、512 dense 神经元、1,024 dense 神经元 × 1,000 步、TimeWheel 1,200 步均通过。
- 同步 REQ/REP：40 个请求，回复完整、序号唯一且单调。
- 异步通信：queues=1/2/4 均接收 40 条消息，序号唯一且单调。
- 流式 REQ/REP roundtrip：heap 和 TimeWheel 均通过，请求步 `[5,10,15,20,25]`，返回计数 `[0,1,1,1,1]`。

### 3.9 绘图

以下 10 个公开绘图功能均成功生成非空 PNG，并对 raster、OuterDynamic trace、InputConv heatmap 做了目视验收：

- spike raster
- neuron state trace
- weight trace
- weight summary
- weight distribution
- OuterDynamic trace
- tracking error
- phase plane
- torque
- InputConv rate heatmap

绘图模块本身通过。测试环境中的系统级 pandas 与 NumPy 2.4.6 ABI 不兼容，因此 `to_pandas()` 未纳入判定；该问题来自复用系统 site-packages 的测试环境，不是 wheel 原生仿真或 matplotlib 绘图缺陷。

## 4. 缺陷与风险分级

### 高：helper 丢失 `output` / `monitored`

- 复现：`nb.NeuronLayer.lif_double(1, output=True, monitored=True)`
- 实际：`output=False`、`monitored=False`，而 `parameters` 中出现字符串键 `output`、`monitored`。
- 影响：DCN/输出层无输出、DebugMonitor 选择失效，并会让部分旧测试产生假阳性。
- 根因范围：Python helper 参数分流，不是 C++ 仿真核心。

### 高：`AdditiveKernalChange` 目录与工厂不一致

- 目录和 dense learning model 声明该规则受支持。
- Simulation 构建时主学习规则工厂无法实例化，直接报告 unknown type。
- 影响：该学习规则实际上不可从当前 Python API 使用，也阻断 dense 专项路径。

### 已知行为：dense Izhikevich 时序差异

- 部分模式 spike count 和逐点电压不同。
- 已确认来源为 `InterfaceNeuron` 延迟；保留现有数据作为预期行为基线，不计入缺陷。

### 已知行为：dense Voltage Jump / Poisson 时序

- Voltage Jump 最大电压差 `4.753125`。
- Poisson spike count 一致，但 dense 全部提前 1 步。
- 上述差异已确认由 `InterfaceNeuron` 延迟造成，不计入缺陷。

### 环境警告

测试 venv 使用 `--system-site-packages`，继承环境存在 PyWavelets、Streamlit 与 NumPy/Packaging/Pillow/Protobuf 的版本冲突。核心测试不依赖这些包，但正式发布验证应使用真正无继承的干净 Python 3.12 venv，并单独安装 wheel 声明的依赖。

## 5. 发布建议

建议先修复 helper 元数据分流和 `AdditiveKernalChange` 工厂注册，然后复跑本报告中的同一套矩阵。dense Izhikevich、Voltage Jump、Poisson 应按 `InterfaceNeuron` 延迟语义建立对齐后的基线，而不是直接用同一步 CPU 状态作强制阈值比较。

## 6. 主要产物

- 核心 JUnit：`D:\neuronBridge\artifacts\full_retest_20260921\pytest_core.xml`
- 学习/OuterDynamic/绘图：`D:\neuronBridge\artifacts\full_retest_20260921\feature_matrix\learning_outer_plot_report.json`
- 绘图目录：`D:\neuronBridge\artifacts\full_retest_20260921\feature_matrix\plots`
- 内存基准：`D:\neuronBridge\artifacts\full_retest_20260921\streaming_500k`、`streaming_1m`、`streaming_dense_500k`
- OpenMP：`D:\neuronBridge\artifacts\full_retest_20260921\openmp_mixed`、`openmp_dense`、`openmp_outer`
- 通信：`D:\testfile\full_retest_20260921\realtime_sync`、`realtime_async`
- 神经元、数值和压力结果：`D:\testfile\special_neuron_tests`、`numerical_accuracy`、`izhikevich_dynamics`、`izhikevich_backend_comparison`、`other_neuron_backend_comparison`、`complex_networks`、`11_regression`、`stress`
