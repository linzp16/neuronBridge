# 修复后 wheel 验证报告

日期：2026-09-17

## 修复内容

1. `TimeDrivenLIF_Exponential_Decay_GPU_Interface.cu` 与 `TimeDrivenLIF_Voltage_jump_GPU_Interface.cu` 的 `UpdateState()` 增加 GPU 到 CPU 的状态变量、`LastUpdate`、`LastSpike` 异步回读。此前 GPU 内核已经处理外部电流并产生事件，但 Python 侧监控状态没有同步，导致看起来“没有响应电流”。
2. 修正 `TimeDrivenLIF_Voltage_jump_GPU_Interface.cu::InitStateVector()`，初始电压改为 `V_reset + init[0]`，与 CPU 实现一致；此前错误使用 `V_rest`。

## wheel

- 文件：`D:\neuronBridge\artifacts\wheel\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- SHA-256：`6576480158496402350303005DECF2F9327D569EE70C532AF765B08840A6C952`
- wheel 检查：PASS；24 个条目、13 个 native DLL、无缺失依赖、无 forbidden/unsafe entry。
- 测试环境：Python 3.12，独立环境 `D:\testfile\venv_bugfix`。

## 定向回归结果

- Exponential Decay legacy GPU：外部电流 5.0 下电压 `-65.0 -> -61.1051483`，`last_update=0 -> 120`，通过。
- Voltage Jump legacy GPU：初始电压 `-5.0`，与 CPU 一致；外部电流 5.0 下电压 `-5.0 -> 4.5267506`，`last_update=0 -> 120`，通过。
- 定向检查 JSON：`legacy_gpu_lif_bugfix_validation.json`，两个检查项均为 `true`。

## 回归结果

- ordinary CPU / Dense GPU / legacy GPU 其他神经元模型曲线：脚本退出码 0。
- Exponential Decay：legacy GPU 与 CPU 最大电压误差 0；Dense GPU 为约 `3.81e-6`。
- Exponential double：legacy GPU 与 CPU 最大电压误差 0；Dense GPU 为约 `3.81e-6`。
- Voltage Jump：legacy GPU 与 CPU 最大电压误差约 `7.45e-8`。
- PoissonRate：CPU、Dense GPU、legacy GPU 均产生 46 个 spike；legacy GPU 与 CPU 的序列一致。
- 数值准确性测试：5/5 通过，最大误差不超过 `5.09e-6`，容差 `2e-5`。

## 产物

- [legacy_gpu_lif_bugfix_validation.json](D:\testfile\other_neuron_backend_comparison\legacy_gpu_lif_bugfix_validation.json)
- [other_neuron_backend_comparison.json](D:\testfile\other_neuron_backend_comparison\other_neuron_backend_comparison.json)
- [other_neuron_voltage_curves.png](D:\testfile\other_neuron_backend_comparison\other_neuron_voltage_curves.png)
- [wheel-contents.json](D:\neuronBridge\artifacts\wheel\wheel-contents.json)
