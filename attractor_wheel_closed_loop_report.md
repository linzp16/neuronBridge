# H:\\attractor 吸引子任务的 NeuronBridge wheel 闭环验证报告

日期：2026-09-18  
wheel：`neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`  
测试输出目录：`D:\\testfile\\attractor_wheel_closed_loop_final2`

## 1. 源工程调查结论

`H:\\attractor` 的核心任务是双环 R-STDP 吸引子网络：输入 spike group 驱动两个相互抑制的 LIF ring，奖励/惩罚 spike 通过 trigger rule 更新输入到 ring 的可塑性连接；原工程还使用端口 5565 的同步 ZMQ REQ/REP，并可连接 CoppeliaSim MTB 控制器。

当前目录不能直接作为可复现实验工程编译，原因是 CMake 和源文件引用了未提供的内容：

- `shared/generated_dual_ring_targets.h`
- `target_sequence_generation_config.h`
- `H:\\attractor\\shared` 目录
- 原测试代码引用的外部 `source_file_realtime_v1_async` 路径

因此本轮没有伪造原始目标序列，而是将缺失配置显式化为 `config.json`，保留拓扑、R-STDP 参数和 wheel 异步 spike 协议，作为可复现的移植夹具。CoppeliaSim/MTB 机器人控制本轮未启动，因为缺少其运行场景和目标序列；当前验证的是神经网络训练与通信闭环。

## 2. wheel 侧闭环实现

脚本：`D:\\testfile\\attractor_wheel_closed_loop.py`

- 输入层：4 个 `InputSpikeNeuronModel`，启用 `communication_input`。
- 环网络：两个 ring，每个 8 个 `TimeDrivenLIF_Exponential_double`，共 16 个输出 LIF。
- 可塑性：`R_STDP`，覆盖 `Max_LTP=0.08`、`Max_LTD=0.05`、`LTP_tau=20`、`LTD_tau=30`、奖励因子 `+1`、惩罚因子 `-1` 和 eligibility 清除。
- 环间连接：双向抑制连接。
- 反馈：偶数阶段发送 reward，奇数阶段发送 punishment，均通过 `trigger_rule=0` 触发对应的 R-STDP 连接。
- 通信：使用 wheel 的 `add_zmq_async_input_output_spike_driver()`，输入端口 5566，输出端口 5567，协议 magic `0x53504B31`、version `1`。

## 3. 实测结果

最终结果文件：`D:\\testfile\\attractor_wheel_closed_loop_final2\\result.json`

| 指标 | 结果 |
|---|---:|
| 仿真步数 | 800 |
| 网络神经元数 | 36 |
| 输入阶段 | 4 |
| 反馈批次 | 4 |
| 反馈 spike | 4 |
| SNN 输出批次 | 81 |
| SNN 输出 spike | 152 |
| Ring 0 输出 spike | 72 |
| Ring 1 输出 spike | 80 |
| 首个塑性连接初始权重 | 2.20000005 |
| 首个塑性连接最终权重 | 2.99729013 |
| 权重变化 | +0.79729009 |
| peer 错误 | 无 |
| 结论 | PASS |

权重发生非饱和增长，说明输入 spike、突触后 spike、reward/punishment trigger 和 R-STDP 更新已经处于同一个可运行闭环中，而不只是通信接口连通。

## 4. 期间定位到的适配问题

首轮没有输出 spike，原因是两个层构造器的 Python 参数误用：`NeuronLayer.lif_double(..., output=True, monitored=True)` 会把这些关键字段放进模型参数，而不是层属性，native 状态显示 `is_output=False`。改为显式构造 `NeuronLayer("TimeDrivenLIF_Exponential_double", ..., output=True, parameters={...})` 后输出恢复。

随后发现桌面上 native 仿真完成速度快于 PUB/SUB 建链。闭环脚本在 native `init()` 完成后显式放行 peer，并等待输入批次排队，再运行带时间戳的输入；这不是修改 wheel，而是测试夹具的握手同步，避免把网络启动竞态误判为模型故障。

## 5. 尚未覆盖的部分

本轮尚未宣称与原工程完全数值等价，因为原始目标序列和 CoppeliaSim 控制依赖不在 `H:\\attractor` 中。下一步应补齐缺失生成配置后：

1. 用原始 phase steps、目标 ring 和 target selection 替换夹具配置。
2. 将 wheel 的异步 PUB/SUB peer 替换为实际控制服务器或增加同步 REQ/REP 适配层。
3. 接入 CoppeliaSim 后验证“神经输出 → 机器人动作 → reward/punishment → 权重更新”的真实控制闭环。
4. 增加多轮训练、权重持久化/恢复和 CPU/Dense/GPU 后端对照。
