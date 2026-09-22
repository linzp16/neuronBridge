# NeuronBridge 未测接口与测试缺口清单

日期：2026-09-21  
适用版本：`neuronbridge 0.1.0a0` Windows wheel  
最新被测 wheel SHA-256：`0F6EB0E02B65886590C860BE8D55469A5D29A53CB5AB90F5313E2C366BA9066F`

## 1. 统计口径

本清单根据当前 Python 公共 API、pybind 暴露接口、pytest 用例、专项测试脚本和既有测试报告进行静态对照。

- “未直接测试”表示测试代码中没有对该公共入口进行直接调用和结果断言。
- 某些入口可能被其他高级接口间接执行，但这不能替代独立的接口契约测试。
- “历史/脚本覆盖”表示此前实验或训练脚本曾经使用，但没有纳入最新 wheel 的固定自动化回归。
- Handwriting 和 `ei_connectivity` 按当前测试范围继续排除。
- 实时运行相关工作按当前版本决策暂缓，不作为发布阻断项。

最新 wheel 已完成：核心回归 `81 passed, 18 deselected`；权重接口快速/流式专项矩阵通过。

## 1.1 补测进度更新

本清单建立后已开始补测，扩展后的固定回归结果为 `92 passed, 18 deselected`。以下原缺口已经补齐直接测试：

- `.nbnet`：`from_batches()`、`append_connection()`、`abort()`、`is_valid()`、显式重复 `finalize()`、batch 生成器异常清理。
- 参数与格式 helper：`float64()`、`int32_list()`、`float32_list()`、`expected_payload_bytes()`、`input_conv_pixel_format()`、`native_name`。
- DebugMonitorResult：`meta()`、`meta_path`、`table_path()`、`to_pandas()` 委托契约。
- DebugMonitor 生命周期：disable、再次 enable、继续运行与 flush。
- InputConv：frame queue 清空、monitor enable/disable。
- Publisher：`publish_frames()`、`wait_for_native_receiver()` 成功和超时。
- 权重文件：错误 header、截断、网络连接数不匹配、非有限文本以及失败后不修改当前权重。
- reset 组合：学习权重更新后，在 DebugMonitor 开启状态下进行快速/流式 reset，并验证权重保持。
- OuterDynamic：运行时 reset state、desired state、错误维度和未知名称。
- 同步 spike ZMQ：heap 与 TimeWheel REQ/REP roundtrip。
- 异步 spike ZMQ：真实 PUB/SUB 输入、输出和显式 `publish_output()` 端到端测试。
- 最新 wheel：7 种学习规则、3 类 OuterDynamic、10 类绘图和 triple 神经元三后端矩阵重新通过。

静态扫描后，当前仅剩 9 个没有直接测试引用的公共入口，全部标记为 **暂缓开放（Deferred Experimental）**。它们不属于当前版本的稳定支持范围，也不作为当前发布测试的阻断项：

| 接口 | 当前状态 | 发布保证 |
|---|---|---|
| `enable_realtime()` | 暂缓开放 | 不纳入当前稳定 API |
| `run_realtime()` | 暂缓开放 | 不纳入当前稳定 API |
| `disable_realtime()` | 暂缓开放 | 不纳入当前稳定 API |
| `reset_bench_profiling()` | 暂缓开放 | 不纳入当前稳定 API |
| `bench_profiling_snapshot()` | 暂缓开放 | 不纳入当前稳定 API |
| `realtime_skip_counters()` | 暂缓开放 | 不纳入当前稳定 API |
| `reset_realtime_skip_counters()` | 暂缓开放 | 不纳入当前稳定 API |
| `realtime_restriction_counts()` | 暂缓开放 | 不纳入当前稳定 API |
| `reset_realtime_restriction_counts()` | 暂缓开放 | 不纳入当前稳定 API |

“暂缓开放”表示接口代码暂时保留，便于后续版本继续开发和验证，但当前不承诺行为稳定性、数值一致性、实时期限或向后兼容性。用户不应在生产流程中依赖这些接口。

注意：`to_pandas()` 当前使用受控 fake pandas 验证路径传递和参数委托；真实 pandas 读取仍受测试环境 NumPy/pandas ABI 冲突影响，尚不能视为完整集成覆盖。

## 2. 完全缺少直接自动化测试的接口

### 2.1 实时运行与性能计数器（暂缓开放）

- `Simulation.enable_realtime()`
- `Simulation.run_realtime()`
- `Simulation.disable_realtime()`
- `Simulation.reset_bench_profiling()`
- `Simulation.bench_profiling_snapshot()`
- `Simulation.realtime_skip_counters()`
- `Simulation.reset_realtime_skip_counters()`
- `Simulation.realtime_restriction_counts()`
- `Simulation.reset_realtime_restriction_counts()`

待验证内容：参数边界、计数器单调性、reset 行为、重复启停、普通运行与实时运行切换、不同实时限制等级下的统计一致性。

当前处理：所有上述接口均为 **暂缓开放**，不作为本版本发布阻断项，不建议用于生产任务。

### 2.2 异步 spike ZMQ 通信

- `Simulation.add_zmq_async_input_output_spike_driver()`
- `Simulation.publish_output()`

现有 `zmqcommunication_client` smoke 使用 `enable_zmq=0`，没有实际启动 native 异步 spike driver。

待验证内容：真实 PUB/SUB 端到端传输、topic 过滤、主动 publish、空 spike 窗口、断线重连、端口占用、消息损坏、慢消费者和关闭顺序。

### 2.3 DebugMonitor 生命周期

- `Simulation.disable_debug_monitor()`

待验证内容：关闭后继续运行、关闭后不再写入、重复关闭、重新开启、多个输出目录、不可写目录、写入失败、多次 `flush()` 幂等性以及 `DebugMonitorConfig.enabled=False`。

### 2.4 InputConv 管理接口

- `Simulation.clear_input_conv_frame_queue()`
- `Simulation.enable_input_conv_monitor()`
- `Simulation.disable_input_conv_monitor()`

待验证内容：队列清空后的状态和统计、运行中动态开关 monitor、多 camera 隔离、队列溢出与淘汰顺序、清空不存在的 source、同步/异步 source 的一致行为。

### 2.5 `.nbnet` helper

- `nbnet.from_batches()`
- `NbnetDescriptionBuilder.append_connection()`
- `NbnetDescriptionBuilder.abort()`
- `nbnet.is_valid()`

`NbnetDescriptionBuilder.finalize()` 已通过 context manager 间接执行，但缺少显式接口测试。

待验证内容：空 batch、多 batch、生成器异常、长度不一致、单连接追加、重复 finalize、finalize 后继续写入、abort 后临时文件清理、覆盖策略以及 `is_valid()` 对 header/checksum/truncation 的返回值。

### 2.6 DebugMonitorResult 数据访问

- `DebugMonitorResult.meta`
- `DebugMonitorResult.meta_path`
- `DebugMonitorResult.table_path()`
- `DebugMonitorResult.to_pandas()`

`to_pandas()` 此前因系统 pandas 与 NumPy ABI 不兼容而未纳入判定。需要在依赖版本受控的隔离环境中补测。

待验证内容：正常表格、表格缺失、空表、错误列、limit、dtype、超大文件及 pandas 未安装时的错误信息。

### 2.7 InputConv 通信辅助接口

- `InputConvFramePublisher.publish_frames()`
- `InputConvFramePublisher.wait_for_native_receiver()`

已覆盖 `publish_frame()`、`publish_frame_until_received()`、FrameClient/FrameServer 请求响应和基本统计。

待验证内容：批量顺序、repeat、等待超时、关闭后调用、native receiver 永不就绪、多 camera 统计。

### 2.8 参数与像素格式辅助函数

- `float64()`
- `int32_list()`
- `float32_list()`
- `expected_payload_bytes()`
- `input_conv_pixel_format()`
- `InputConvPixelFormat.native_name`

待验证内容：正常转换、空列表、错误元素类型、整数溢出、非有限浮点值、未知字符串/枚举值以及各种像素格式的 payload 大小。

## 3. 有历史或脚本覆盖，但未在最新 wheel 上重新执行

- `Simulation.reset_outer_dynamic_state()`
- `Simulation.set_outer_dynamic_desired_state()`
- `FeedbackProductEncoding`
- 同步 REQ/REP spike 通信完整闭环
- 完整小脑多轮训练
- CoppeliaSim 机械臂闭环
- 50 万和 100 万突触流式构建内存基准
- `tests/full_acceptance/triple_backend_smoke.py`
- `tests/full_acceptance/learning_outer_plot_matrix.py`
- `tests/streaming_build/reqrep_spike_roundtrip_probe.py`

这些功能不是“从未运行”，但当前最新 wheel 只完成了核心 pytest 和权重专项复验，因此仍需要重新执行后才能纳入最新 wheel 的发布结论。

## 4. 已测试但异常路径覆盖不足

### 4.1 权重接口

已覆盖：读取、修改、保存、加载、快速/流式跨模式快照、reset 保持主网和 dense 权重、索引越界、非有限值拒绝、文件打开失败传播。

仍缺少：

- 损坏或截断的权重快照。
- 权重数量与当前网络不一致。
- 快照中存在 `NaN` 或 `Inf`。
- 只读文件和磁盘写满等写入错误。
- 负权重及超过 `max_weight` 的正式接口策略。目前二者仍允许写入。

### 4.2 神经元状态接口

涉及：`neuron_state()`、`neuron_states()`。

仍缺少：负索引、越界索引、重复 ID、空列表、main/dense 混合批量读取、仿真初始化前读取、reset 前后字段一致性。

### 4.3 dense 子网接口

涉及：名称查找、权重、snapshot、reset 和 full-firing export。

仍缺少：错误名称、负索引、越界索引、同名子网、运行中 reset、学习更新后 reset、reset 后 GPU/host 缓存一致性。

### 4.4 OuterDynamic

已覆盖 spike counter、planar arm、strict Matlab arm、状态读取和绘图。

仍缺少：未知模型、重复名称、错误关节维度、非有限目标、运行中 reset、多 OuterDynamic 独立控制和异常动力学参数。

### 4.5 绘图接口

主要公开绘图方法已生成 PNG 并通过 smoke。

仍缺少：空数据、缺列、损坏 CSV、非有限值、超大文件、多个 component/OuterDynamic 选择、非法 neuron/synapse ID 以及用户传入 axes 的组合行为。

### 4.6 同步和异步通信

InputConv 同步/异步正常路径、错误响应 magic 和错误 topic 恢复已有测试。

仍缺少：超时、连接中断、server 重启、端口占用、并发 client、超大消息、版本不兼容和关闭期间仍有请求。

### 4.7 reset 综合语义

最新专项已证明 `Simulation.reset()` 保持 main 和 dense 当前权重。

仍缺少：

- 学习规则更新后多次 reset/run。
- DebugMonitor 开启状态下 reset。
- 同步或异步通信 driver 存在时 reset。
- InputConv 缓冲区和绑定状态。
- OuterDynamic、事件队列、输出 spike 缓冲和随机数状态。

## 5. 建议实施顺序

### P0：下一轮优先补齐

1. 在最新 wheel 上重跑学习规则、OuterDynamic、全部绘图和三后端综合矩阵。
2. 为 reset 增加学习规则、dense、OuterDynamic、DebugMonitor 和通信组合测试。
3. 测试损坏权重快照、网络不匹配快照和非有限文件内容。
4. 把同步 REQ/REP spike roundtrip 纳入固定 pytest 或发布验收脚本。

### P1：重要公共接口

1. 为 `from_batches()`、`append_connection()`、`abort()`、`is_valid()` 增加完整 `.nbnet` helper 测试。
2. 完成真实异步 spike ZMQ 端到端及断线恢复测试。
3. 补齐 InputConv 队列清空、monitor 动态开关和通信错误路径。
4. 在受控 pandas 环境中测试 `to_pandas()`。

### P2：便利接口和边界质量

1. 参数类型与像素格式 helper 单元测试。
2. DebugMonitorResult 路径、metadata 和缺失文件处理。
3. 绘图空数据、损坏数据和大文件测试。
4. dense、状态查询和 OuterDynamic 的错误参数矩阵。

### 暂缓开放项

实时运行、实时限制策略和相关计数器保留在源码中，但统一标记为“暂缓开放”，按当前决策推迟到后续版本，不阻塞本版本发布，也不属于当前稳定 API 保证。

## 6. 完成标准

一个接口只有同时满足以下条件才从本清单移除：

1. 至少有一个正常路径自动化测试。
2. 至少有一个关键错误路径测试。
3. 对 native 接口验证 Python 异常或返回值契约。
4. 涉及快速/流式构建时，两条路径均验证。
5. 涉及 CPU、legacy GPU 或 dense GPU 时，覆盖所有声明支持的后端。
6. 测试进入固定回归命令，而不只是一次性手工脚本。
