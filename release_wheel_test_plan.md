# NeuronBridge 发布 wheel 测试计划

## 1. 测试目标

验证发布 wheel：

`D:\neuronBridge\artifacts\wheel\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`

在 Windows x64、Python 3.12 环境下能够：

- 正常安装、导入和卸载；
- 覆盖各类神经元、稠密子网络、输入卷积和外部动力学模型；
- 覆盖数组、队列、文件、ZeroMQ 同步/异步通信；
- 验证网络定义、参数定义、连接定义、学习规则定义；
- 验证 heap/timing wheel 等仿真和通信模式；
- 验证运行结果、权重读写、监控输出和错误处理；
- 出现失败时定位到 Python API、wheel 打包、原生绑定、CUDA、通信协议或运行时依赖。

所有测试计划、测试日志、结果、诊断信息和复现脚本统一放置在：

`D:\testfile`

## 2. 测试环境

- Windows 10/11 x64
- Python 3.12
- 目标 wheel：`cp312-cp312-win_amd64`
- CUDA Toolkit/驱动可用
- GPU 可用性单独记录
- ZeroMQ/PyZMQ 可用
- 使用隔离虚拟环境，不使用源码目录中的 Python 包

开始测试前记录：

- Python、pip、操作系统版本；
- wheel 文件 SHA-256；
- NVIDIA 驱动、CUDA 版本、GPU 型号和显存；
- PyZMQ 版本；
- 测试环境变量和安装依赖。

## 3. 测试阶段

### 3.1 wheel 静态检查

检查 wheel 文件名标签、元数据和内部布局，重点确认：

- 存在 `neuronbridge/_core.cp312-win_amd64.pyd`；
- 包含 CUDA runtime、ZeroMQ、Pinocchio、MSVC/OpenMP 运行库；
- 不包含 `.pdb`、中间编译文件、源码泄漏或调试产物；
- 依赖声明与实际运行需求一致。

输出：`wheel_inspection.json`、`wheel_contents.txt`、`wheel_sha256.txt`。

### 3.2 安装、导入和卸载

在全新虚拟环境中执行安装、导入、版本查询、`backend_info()`、模型目录查询和卸载测试。重点关注 DLL 加载失败、`_core` 导入失败、依赖缺失以及源码/build 目录污染。

### 3.3 模型和网络定义

覆盖以下模型：

| 类型 | 模型 |
|---|---|
| 输入脉冲 | `InputSpikeNeuronModel` |
| 输入电流 | `InputCurrentNeuronModel` |
| LIF | `TimeDrivenLIF_Exponential_Decay`、`TimeDrivenLIF_Exponential_double`、`TimeDrivenLIF_Voltage_jump` |
| Izhikevich | `TimeDrivenIzhikevic_Exponential_Decay` |
| Poisson | `PoissonRate` |
| Dense CUDA | Dense LIF、Dense current、Dense mixed |
| InputConv | `InputConvV1` |

覆盖单层、多层、脉冲输入、电流输入、普通连接、多源多目标连接、权重、延迟、突触类型、学习规则、Dense 子网络、InputConv、OuterDynamic 及其连接。

验证 `to_dict()`、`to_native()`、参数类型、数组维度、参数广播、模型目录和 backend 选择，以及非法定义是否产生明确异常。

### 3.4 仿真和事件队列模式

覆盖：

- `event_queue="heap"`；
- `event_queue="timing_wheel"`；
- 单队列和多队列；
- 不同 timestep 和仿真步数；
- 单次运行、分段运行、重复运行、`reset()`、`flush()` 和 `publish_output()`。

验证分段运行与一次性运行结果一致，非法配置被正确拒绝。

### 3.5 通信方式

#### 数组和队列

覆盖 `add_external_spikes()`、`add_external_currents()`、InputConv frame queue、单帧/多帧、多 source、缺帧、滞后帧和队列清理。

#### ZeroMQ 同步通信

覆盖 REQ/REP、`InputConvFrameServer`、`InputConvFrameClient`、单请求、连续请求、错误请求、超时、服务端关闭和连接失败。

#### ZeroMQ 异步通信

覆盖 PUB/SUB、`InputConvFramePublisher`、异步 InputConv source、异步输入输出 spike driver、topic、多帧发布、重连、丢帧和关闭清理。

验证消息数量、frame 内容、shape、时间步、source/camera index、topic 隔离、线程和 socket 生命周期。

### 3.6 通信协议和数据定义

覆盖同步/异步 InputConv frame、frame request、spike batch、header、payload、长度、shape、时间步、source/camera/index、endian 和版本字段。

测试空 payload、最小/大 payload、截断数据、错误 header、错误版本、错误长度、错误 shape、非法时间步、重复消息和乱序消息。

预期合法数据能够完整 round-trip，非法数据产生明确异常，不发生崩溃、死锁或静默丢失。

### 3.7 学习规则和外部动力学

覆盖 `LearningRule`、STDP/R-STDP、触发规则、后突触规则、权重更新、OuterDynamic Spike Counter、Planar Arm 2DOF、Strict Matlab Planar Arm 2DOF、state/error feedback 和外部动力学连接。

验证规则参数、权重变化、外部动力学状态、反馈数据和通信周期传递正确。

### 3.8 结果、监控和权重 I/O

覆盖 spike 输出、neuron state、Dense snapshot、InputConv output、debug monitor、state/weight/outer-dynamic trace、CSV/JSON、权重保存和加载、reset 后权重行为以及绘图接口。

验证输出文件完整、字段正确、数量与网络定义一致、权重保存/加载结果一致，且输出不写入源码目录。

### 3.9 示例和回归

执行基础 API、Dense CUDA、InputConv、ZeroMQ、闭环仿真等公开示例。外部数据依赖示例分为无数据导入测试、最小数据 smoke test 和完整数据测试。

### 3.10 复杂混合网络测试

在现有单一模型和小型组合测试之外，增加复杂混合网络专项测试。该专项不执行 handwriting 和 `ei_connectivity` 外部数据流程，但覆盖其余公开 API 和运行时能力。

#### 混合网络案例

1. **脉冲输入 + LIF + Izhikevich + STDP**
   - InputSpike 输入层；
   - LIF 和 Izhikevich 并行层；
   - 交叉连接、不同 delay 和 synapse type；
   - 同时启用普通 STDP 和 R-STDP；
   - 验证脉冲传播、权重变化和输出一致性。

2. **电流输入 + Poisson + Dense LIF**
   - InputCurrent 输入层；
   - PoissonRate 层；
   - Dense LIF 子网络；
   - 普通网络到 Dense 子网络的连接；
   - 验证电流输入、发放率、Dense snapshot 和权重状态。

3. **多 Dense 子网络级联**
   - 两个或多个 Dense 子网络；
   - Dense LIF、Dense current 或 CustomLifConductanceV1 混合；
   - Dense-to-Dense 连接；
   - 不同 queue、update timestep 和 firing export 配置；
   - 验证子网络名称、局部/原始 neuron id 映射和跨子网络数据传递。

4. **InputConv + 主网络 + Dense 子网络**
   - InputConvV1 使用 bar、grating、plaid 和 file 定义；
   - 一路输出到主网络，另一路输出到 Dense 子网络；
   - 同时使用 frame queue 和 ZeroMQ frame source；
   - 验证 source/camera index、帧时间步、输出映射和帧缺失策略。

5. **OuterDynamic 闭环网络**
   - 主神经网络产生 spike 或 state；
   - OuterDynamic SpikeCounter 或 PlanarArm2DOF 接收反馈；
   - 外部动力学输出再次反馈到神经网络；
   - 混合 state feedback、error feedback 和外部动力学连接；
   - 验证通信周期、状态更新、反馈延迟和监控结果。

6. **完整闭环混合网络**
   - InputConv 动态输入；
   - 主网络包含 InputSpike、InputCurrent、LIF、Poisson 和 Izhikevich；
   - 至少一个 Dense 子网络；
   - 至少一个学习规则；
   - 至少一个 OuterDynamic；
   - 同时启用 debug monitor、weight I/O 和输出发布；
   - 分别使用 heap/timing wheel、单队列/多队列、同步/异步通信运行。

#### 复杂网络测试层级

| 层级 | 网络规模 | 目的 |
|---|---:|---|
| M1 组合冒烟 | 4–32 neurons | 验证定义和连接拓扑 |
| M2 中型混合 | 64–512 neurons | 验证跨模型、跨 backend 和数据流 |
| M3 大型混合 | 1,000–10,000 neurons | 验证内存、队列、Dense runtime 和稳定性 |
| M4 长时闭环 | 1,000+ steps | 验证状态漂移、学习规则和资源生命周期 |

#### 复杂网络验收点

- 网络定义能够完整转换为 native description；
- 所有层、连接、学习规则、InputConv 和 OuterDynamic 数量与定义一致；
- 不同 backend 对明确支持/不支持的组合返回正确结果；
- heap 与 timing wheel 在相同输入下结果满足预设容差；
- 同步与异步通信在相同输入下输出事件顺序和关键统计一致；
- 分段 run 与一次性 run 的状态和权重结果一致；
- reset、save/load weights 后网络状态符合预期；
- monitor、snapshot、spike、state 和 weight 文件字段完整；
- 运行结束后 ZeroMQ socket、线程、CUDA 资源和临时文件正确释放；
- 发生失败时能够缩减到单个模型、单条连接或单种通信模式。

复杂混合网络的测试脚本、网络定义、输入数据、输出快照和对比报告统一放在 `D:\testfile\complex_networks`，失败复现脚本放在 `D:\testfile\repro`。

## 4. 测试矩阵

| 维度 | 覆盖项 |
|---|---|
| 后端 | legacy CPU、legacy GPU、dense GPU |
| 神经元 | InputSpike、InputCurrent、LIF、Izhikevich、Poisson |
| 连接 | 单连接、多连接、延迟、权重、学习规则 |
| 输入 | spike、current、InputConv queue、file、ZeroMQ |
| ZeroMQ | REQ/REP、PUB/SUB、同步、异步 |
| 仿真队列 | heap、timing wheel |
| 执行模式 | 单次、分段、reset、重复运行 |
| 输出 | spike、state、weight、monitor、snapshot |
| 外部动力学 | spike counter、planar arm、strict Matlab |
| 数据规模 | 最小、普通、边界、大规模 |
| 错误场景 | 类型、范围、长度、协议、资源、生命周期 |

测试分为代表性冒烟组合和重点模型 × 通信方式 × 仿真模式的全量组合。

## 5. 结果目录

```text
D:\testfile\
├─ 00_environment\
├─ 01_wheel_inspection\
├─ 02_install_import\
├─ 03_catalog_model\
├─ 04_network_definition\
├─ 05_simulation_modes\
├─ 06_communication\
│  ├─ array_queue\
│  ├─ zmq_sync\
│  └─ zmq_async\
├─ 07_protocol\
├─ 08_learning_outer_dynamic\
├─ 09_result_weight_io\
├─ 10_examples\
├─ 11_regression\
├─ logs\
├─ failures\
├─ repro\
└─ summary\
```

每个用例至少保存用例 ID、参数、环境、stdout/stderr、返回码、预期结果、实际结果、PASS/FAIL 和复现脚本。

## 6. Bug 定位流程

1. 在全新 wheel 环境确认复现；
2. 排除源码目录、旧 build 目录和旧 DLL 污染；
3. 对比源码构建版本和发布 wheel 版本；
4. 判断属于 wheel 打包、Python API、pybind11 转换、C++ runtime、CUDA、ZeroMQ、协议、模型或线程生命周期问题；
5. 缩减为最小模型、输入和步数；
6. 保存完整参数、输入数据、环境和错误日志；
7. 对比 CPU/GPU、同步/异步、heap/timing wheel；
8. 生成缺陷报告和可重复运行脚本；
9. 修复后执行相关回归用例。

缺陷报告至少包含编号、严重程度、失败用例、复现命令、预期/实际行为、错误信息、最小复现脚本、初步根因、影响范围和回归结果。

## 7. 通过标准

以下情况判定 wheel 不通过：无法安装或导入、核心 DLL 缺失、关键模型无法初始化、正常输入导致崩溃/死锁/数据破坏、同步或异步通信无法闭环、仿真结果与定义明显不一致，或运行依赖源码/build 目录。

非核心外部数据缺失、可选绘图问题、性能退化和非公开模型差异单独记录并评估。

最终输出测试汇总、用例通过率、失败清单、Bug 清单、风险评估、发布建议和修复后的回归范围。
