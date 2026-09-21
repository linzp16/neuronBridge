# CoppeliaSim 机械臂闭环通信测试方案

## 1. 测试目标

将之前的吸引子模型机械臂控制任务整理为通信专项测试，验证以下闭环是否完整、同步且无丢包：

```text
NeuronBridge wheel
    --同步 ZMQ REQ/REP--> Python Peer
    --ZMQ Remote API--> CoppeliaSim MTB 场景
    <--关节/末端反馈--
    <--reward/punishment spike-- Python Peer
```

本测试同时观察控制结果，但主要验收对象是通信闭环，而不是训练准确率。

## 2. 组件职责

- NeuronBridge wheel：运行吸引子网络，按通信间隔发送 ring 输出 spike，并阻塞等待 Python Peer 的反馈。
- Python Peer：运行 REP 服务端，接收 wheel 输出；解码两个 ring 的 spike winner；根据 winner 生成关节控制目标；读取 CoppeliaSim 的关节/末端状态；返回 reward 或 punishment spike。
- CoppeliaSim：运行 MTB 场景和场景脚本，执行关节目标，提供关节与末端位置反馈，并绘制末端红色轨迹。
- CoppeliaSim Remote API：默认连接 `localhost:23000`，只负责 Python Peer 与仿真场景之间的控制和状态读取。

## 3. 协议检查

NeuronBridge 与 Python Peer 使用同步 REQ/REP：

- 请求头：`<II>`，分别为 `time_step` 和 `output_spike_count`。
- 每个输出 spike：`<iff>`，分别为 `neuron_id`、`spike_time`、`base_timestep`。
- 回复头：`<I>`，表示反馈 spike 数量。
- 每个反馈 spike：`<iff>`。
- 每个 REQ 必须对应一个 REP；Peer 未回复时 wheel 的仿真推进应暂停。

## 4. 测试指标

必须记录：

1. REQ/REP 请求数和反馈批次数是否一致。
2. wheel 输出 spike 批次数、总数及 Peer 接收数。
3. reward/punishment 反馈 spike 数。
4. Remote API 连接错误、协议解析错误和 Peer 异常。
5. 阶段控制次数、轨迹采样点数及末端轨迹范围。
6. Peer 等待、协议解析、CoppeliaSim 控制和回复发送耗时。

## 5. 运行命令

先启动 CoppeliaSim，加载：

`H:\attractor\MTB_scene_wheel_closed_loop.ttt`

确认 Remote API 服务端口为 `23000`，再执行：

```powershell
python D:\neuronBridge\coppeliasim_communication_test.py `
  --output-dir D:\testfile\coppeliasim_communication_run `
  --port 23000 `
  --phases 64 `
  --communication-interval 1000 `
  --epochs 1 `
  --joint-path /MTB/axis `
  --joint-path /MTB/link/axis
```

专项入口会复用 `coppeliasim_attractor_closed_loop.py`，并生成：

- `result.json`：原始闭环结果。
- `communication_report.json`：结构化通信指标。
- `communication_report.md`：人工可读报告。
- `closed_loop_stdout.txt`：完整控制台日志。
- `joint_trajectory.csv`：关节和末端轨迹。
- `ring_spikes_from_zmq.csv`：从 wheel 接收的 ring spike。
- `debug_monitor_ring`：ring 监控数据。

## 6. 通过标准

满足以下条件才判定通信测试通过：

- REQ/REP 请求数大于 0。
- 收到反馈批次大于 0。
- 无 Peer 或 Remote API 错误。
- 阶段控制次数大于 0。
- 轨迹采样点大于 0。
- wheel 输出、Peer 接收和反馈数据均能在报告中对应起来。

如果轨迹为零但通信指标全部通过，应判定为“通信通过、机械控制失败”，继续检查场景脚本或关节模式。
