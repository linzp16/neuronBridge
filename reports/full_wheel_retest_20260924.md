# NeuronBridge 0.1.0a1 Windows wheel 全量回归报告（2026-09-24）

> 修复更新：根目录 `VERSION`、release 版本断言、dense 旧用例和混合 OpenMP spike 顺序断言均已修正。精简发布门禁复跑结果为全绿：31 个 pytest、7 个 extended regression 用例以及 Heap/TimeWheel × 1/2/4 队列混合 OpenMP 矩阵全部通过。

## 1. 结论

本轮从发布产物 `neuronbridge-0.1.0a1-cp312-cp312-win_amd64.whl` 创建独立虚拟环境并重新执行既定全量测试。继续按既定范围排除 Handwriting 和 `ei_connectivity`。

核心仿真结构未发现被本次 ROKAE/Pinocchio 更新破坏：快速构建、C++ 直接流式构建、CPU/legacy GPU/dense GPU、学习规则、OuterDynamic、DebugMonitor、权重接口、OpenMP、Heap/TimeWheel、同步/异步通信、绘图和大规模网络均能运行。未出现内存非法访问、崩溃或 checksum/依赖加载错误。

初次执行发现 1 个发布元数据问题和 2 个旧测试基线问题，现均已修复：

1. 根目录 `VERSION` 仍为 `0.1.0a0`，而 CMake、Python 包和 wheel 为 `0.1.0a1`，导致发布元数据测试失败。
2. `wheel_extended_tests.py` 的 dense 用例仍把 dense 层标记为 `output=True`；现行接口要求通过 dense→main 或 dense→dense 路由声明输出。合法路由下的 dense 专项、压力和多线程测试均通过。
3. `mixed_openmp_eventqueue_test.py` 对 `output_spikes()` 使用列表逐项相等。在多线程下，同一时间步的两个 spike 返回顺序可能互换；事件数量及排序后的 `(time, neuron_id)` 集合完全一致，dense 状态和 OuterDynamic 计数也完全一致。

因此，本轮没有证据表明 wheel 更新破坏了已有运行时结构；修复后精简发布门禁已全部通过。

## 2. 被测对象

- Wheel：`D:\neuronBridge\artifacts\wheel\neuronbridge-0.1.0a1-cp312-cp312-win_amd64.whl`
- Wheel 大小：9,841,054 bytes
- SHA-256：`5BA77E14FBFB6AFA057BBB71B5518AE52CDC8A187C800A9D3EC22120404C7E6F`
- Python：3.12.4
- 测试环境：`D:\neuronBridge\artifacts\full_retest_20260924\venv`
- 实际导入：上述 venv 的 `site-packages\neuronbridge\__init__.py`
- CUDA：启用
- dense runtime：启用

## 3. 自动化回归

- 核心集合：153 passed，2 skipped，1 failed，14 deselected，32 subtests passed。
- ROKAE 测试在显式提供 `D:\Pinocchio\xMateSR3C.urdf` 后单独复跑：1 passed。
- `ei_connectivity` skip 属于明确排除范围。
- 唯一核心失败：根目录 `VERSION` 为 `0.1.0a0`，其余版本位置为 `0.1.0a1`。

JUnit：`D:\neuronBridge\artifacts\full_retest_20260924\pytest_core.xml`

## 4. 功能矩阵

### 4.1 神经元与数值准确性

- LIF Decay 最大绝对误差：`5.08034e-6`，阈值 `2e-5`。
- LIF Double 最大绝对误差：`5.08034e-6`。
- Voltage Jump 最大绝对误差：`3.82265e-7`。
- Izhikevich 电压与恢复变量参考误差：`0`。
- Poisson 固定序列：10 个 spike 完全一致。
- Izhikevich 五类动力学均复现：tonic spiking 38、phasic spiking 1、tonic bursting 120、phasic bursting 50、mixed mode 33。
- Triple LIF：CPU 与 legacy GPU spike 时刻一致，dense GPU 状态有限且 GPU backend ready。
- TriggerRelay、EDLUTLikeLIF、CustomLifConductanceV1 的支持矩阵与运行测试通过。
- dense Izhikevich、Voltage Jump、Poisson 的既有 InterfaceNeuron 延迟差异保持原基线，没有出现新的偏差类型。

### 4.2 学习规则

以下规则在 fast 与 streaming 两条路径全部通过，最终权重跨路径一致：

- `STDP`
- `R_STDP`
- `AdditiveKernalChange`
- `CerebullarLearningRule`
- `CustomPairStdpV1`
- `CustomRStdpV1`
- `CustomRStdpPersistentV1`

helper 的 `output` / `monitored` 元数据转发检查通过。

### 4.3 OuterDynamic

- `OuterDynamicSpikeCounter`：fast/streaming 均观察到 6 个事件，状态一致。
- `PlanarArm2DOFPinocchio`：fast/streaming 均返回完整有限的 2 轴状态。
- `StrictMatlabPlanarArm2DOFOuterDynamic`：fast/streaming 均返回完整有限的 2 轴状态。
- `ROKAE_Arm`：fast 路径 DebugMonitor 测试通过；streaming 路径走 `streaming_direct`，`q/qv/qdd/q_des/qv_des/tau_total` 均为 6 轴有限值。

### 4.4 权重接口

- fast/streaming 的读取、修改、保存、恢复、reset 保持权重全部通过。
- fast 保存的权重可由 streaming 网络读取，跨构建模式迁移通过。
- 非有限值被拒绝；越界索引和 I/O 错误均返回明确异常。

### 4.5 OpenMP 与事件队列

- 主网 LIF Double、Voltage Jump、Izhikevich：1/2/4 队列状态与 spike 完全一致。
- 4 个 dense 子网、40 神经元、240 步：1/2/4 队列 snapshot、113 条 DebugMonitor spike 和 10 条输出 spike 一致。
- 2 个 OuterDynamic counter：1/2/4 队列计数一致。
- 混合主网 + 2 dense + 2 OuterDynamic：Heap/TimeWheel 的数值状态、计数和 spike 事件集合一致。
- 多线程返回列表中，同一时间步 spike 的顺序不保证稳定；排序后完全一致。应修正旧测试断言，而不是判定为数值错误。

### 4.6 网络规模与压力

- 大型主网混合网络：2,432 神经元、200 步、3,114 输出 spike，通过。
- 大型 dense 级联：2,560 神经元、300 步，1024/1024/512 三个子网 snapshot 完整。
- 20 次重复构建运行通过。
- 512 dense 神经元 × 500 步通过。
- 1,024 dense 神经元 × 1,000 步通过。
- TimeWheel 1,200 步通过。

### 4.7 流式内存基准

| 场景 | fast 峰值工作集 | streaming 峰值工作集 | 降低 | streaming 构建时间 |
|---|---:|---:|---:|---:|
| 主网 50 万突触 | 292.4 MiB | 130.1 MiB | 55.51% | 0.189 s |
| 主网 100 万突触 | 550.1 MiB | 218.6 MiB | 60.27% | 0.372 s |
| dense 50 万突触 | 345.6 MiB | 225.4 MiB | 34.79% | 0.352 s |

三项均报告 `runtime_build_path=streaming_direct`，mmap 和 checksum 验证成功。

### 4.8 通信与实时接口

- Python 同步 REQ/REP：100/100 请求响应成功。
- 原生同步外部服务器：41/41 帧成功，无失败或丢帧。
- 原生异步 PUB/SUB：20 帧成功，无丢帧。
- 实时同步 REQ/REP：40 个窗口全部回复，步号单调且唯一。
- 实时异步通信：queues=1/2/4 均收到 40 条消息，无协议错误，步号单调且唯一。

### 4.9 DebugMonitor 与绘图

以下公开绘图接口均生成非空 PNG：

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

Izhikevich 五类动力学图、CPU/dense/legacy GPU 对比图和 Poisson raster 也已重新生成。

## 5. Wheel 隔离与依赖检查

- 隔离安装 CUDA/runtime smoke：PASS。
- wheel 内容检查：PASS。
- wheel 包含 `_core.cp312-win_amd64.pyd`、CUDA runtime、ZeroMQ、Pinocchio default/parser、URDFDOM、Boost、OpenMP/MSVC runtime 等 28 个 DLL。
- unsafe/forbidden entries：0。
- 基础 `.nbnet` fast/streaming 等价 smoke：PASS。
- dense、跨子网、OuterDynamic、InputConv 的完整直接流式 smoke：PASS。

## 6. 已完成的修复

1. 根目录 `VERSION` 已从 `0.1.0a0` 更新为 `0.1.0a1`；release 测试改为读取 `VERSION`，不再硬编码版本号。
2. `wheel_extended_tests.py` 已取消 dense 层的 `output=True`，并增加合法 dense→main 输出路由。
3. `mixed_openmp_eventqueue_test.py` 已按 `(time, neuron_id)` 对输出 spike 多重集合进行规范化比较，不再依赖同时间步内的并发插入顺序。
4. 精简发布门禁复跑：31 passed；extended regression 7/7 passed；混合 OpenMP 全矩阵 passed。

## 7. 主要产物

- 总测试目录：`D:\neuronBridge\artifacts\full_retest_20260924`
- 核心 JUnit：`D:\neuronBridge\artifacts\full_retest_20260924\pytest_core.xml`
- 学习/OuterDynamic/绘图：`D:\neuronBridge\artifacts\full_retest_20260924\feature_matrix`
- OpenMP：`D:\neuronBridge\artifacts\full_retest_20260924\openmp_main`、`openmp_dense`、`openmp_outer`、`openmp_mixed`
- 流式内存：`streaming_500k`、`streaming_1m`、`streaming_dense_500k`
- 权重接口：`D:\neuronBridge\artifacts\full_retest_20260924\weight_matrix`
- ROKAE 流式验证：`D:\neuronBridge\artifacts\full_retest_20260924\rokae_streaming`
- 隔离 wheel 验证：`D:\neuronBridge\artifacts\full_retest_20260924\isolated_wheel_validation`
- 历史兼容输出：`D:\testfile`
