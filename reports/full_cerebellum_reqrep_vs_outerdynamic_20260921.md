# 完整小脑模型：OuterDynamic 与同步 REQ/REP 对比

## 结论

新的外部机械臂方案已经完成 100 个 epoch 的训练。机械臂动力学、目标轨迹、GC/CF 编码和奖励式学习反馈均由 Python REP server 负责；NeuronBridge 只运行神经网络，并通过阻塞式 REQ/REP 每 200 个仿真步交换一次 DCN 输出与下一窗口的 GC/CF 输入。

REQ/REP 方案成功收敛，最终精度与内部 OuterDynamic 基线接近。100 轮期间共完成 4,800 次同步请求，无协议错误。

## 网络和训练配置

| 项目 | 数值 |
|---|---:|
| 神经元 | 47,400 |
| 突触 | 36,001,600 |
| 可塑 GC→PC 突触 | 18,000,000 |
| 每个关节 GC | 22,500 |
| PC / CF / DCN（每群） | 200 / 200 / 200 |
| 基础时间步 | 0.1 ms |
| 控制/通信间隔 | 200 步（20 ms） |
| 每个 epoch 控制窗口 | 48 |
| epoch | 100 |

## 发现并修复的问题

DCN 层最初使用 `NeuronLayer.lif_double(..., output=True)` 创建。该 helper 会把未声明的 `output` 收入模型参数字典，而不会设置 `NeuronLayer.output`，因此原生层的 `Neuron::IsOutput` 实际为 `false`。

这在内部 OuterDynamic 路径中不明显，因为 DCN 仍可通过隐藏的 OuterDynamic 连接传播控制 spike；但外部 REQ/REP server 只能接收标记为输出的神经元，所以最初 4,800 次通信中收到的 DCN spike 为 0。

修复方式是显式构造 `NeuronLayer("TimeDrivenLIF_Exponential_double", ..., output=True, parameters=...)`。训练脚本还增加了原生 DCN `is_output` 启动前检查，并要求验收时 DCN 输出数大于 0、学习权重发生变化。

独立的最小回环测试证明，REQ/REP 回复中的 spike 在 heap 与 TimeWheel 中均能正确重新进入原生事件队列；因此问题不是 ZMQ 解码或事件调度缺陷。

## 100 轮结果

| 指标 | 内部 OuterDynamic | 外部同步 REQ/REP |
|---|---:|---:|
| 第 1 轮平均误差 | 0.145190 m | 0.204942 m |
| 第 100 轮平均误差 | 0.010639 m | 0.011675 m |
| 最终/初始误差比 | 7.33% | 5.70% |
| 最佳轮 | 98 | 93 |
| 最佳平均误差 | 0.005766 m | 0.005837 m |
| 平均每轮耗时 | 0.1526 s | 0.1825 s |
| 100 轮训练段耗时 | 26.99 s | 29.91 s |

REQ/REP 的平均每轮耗时增加约 19.6%，但 Python server 单次请求处理平均仅 0.564 ms。100 轮共接收 10,851,253 个 DCN spike，返回 65,162 个 GC/CF spike。

## 文件

- 外部机械臂训练脚本：`tests/streaming_build/train_full_cerebellum_reqrep.py`
- 外部网络构建脚本：`tests/streaming_build/build_full_cerebellum_streaming.py --external-arm-server`
- REQ/REP 回环探针：`tests/streaming_build/reqrep_spike_roundtrip_probe.py`
- 对比绘图脚本：`tests/streaming_build/plot_cerebellum_reqrep_comparison.py`
- 外部网络：`artifacts/full_cerebellum_reqrep_network/full_cerebellum.nbnet`
- REQ/REP 训练结果：`artifacts/full_cerebellum_reqrep_training_100/training_report.json`
- 最终权重：`artifacts/full_cerebellum_reqrep_training_100/weights_final.bin`
- 对比图：`artifacts/full_cerebellum_reqrep_training_100/outerdynamic_vs_reqrep_comparison.png`

## 运行命令

```powershell
$env:PYTHONPATH='D:\neuronBridge\build\wheel_streaming_full_direct\python\Release;D:\neuronBridge\python\src;D:\neuronBridge\tests\streaming_build'
& 'D:\anaconda\python.exe' 'tests\streaming_build\train_full_cerebellum_reqrep.py' `
  --epochs 100 `
  --output-dir 'artifacts\full_cerebellum_reqrep_training_100'
```

脚本在同一 Python 进程中启动 REP server 线程，然后构造一次流式网络并连续训练 100 轮。每轮调用 `reset(preserve_weights=True)`，不会重复构建网络。
