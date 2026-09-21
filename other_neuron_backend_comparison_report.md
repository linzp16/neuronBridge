# 其它神经元模型三后端测试报告

## 测试范围

本轮对 Izhikevich 之外、同时注册在普通 CPU、Dense GPU 和 legacy GPU 的模型进行了同样测试：

- `TimeDrivenLIF_Exponential_Decay`
- `TimeDrivenLIF_Exponential_double`
- `TimeDrivenLIF_Voltage_jump`
- `PoissonRate`

测试使用 `dt=0.25`、4000 步和 `synapse_type=3`、`delay=1` 外部输入。Dense PoissonRate 使用其专用参数键 `poisson_rate_bias_hz` 和 `poisson_rate_gain_hz_per_current`。

## 结果

| 模型 | 普通 CPU | Dense GPU | legacy GPU | 结论 |
|---|---:|---:|---:|---|
| LIF Exponential Decay | 正常积分 | 正常，最大差异 `3.81e-6` | 未响应外部电流 | legacy GPU 存在输入路径问题 |
| LIF Exponential Double | 正常积分 | 正常，最大差异 `3.81e-6` | 与 CPU 一致 | 通过 |
| LIF Voltage Jump | 正常积分 | 有明显初始/相位差异 | 未响应外部电流且初始状态不同 | 需定位后端差异 |
| PoissonRate | 46 spikes | 46 spikes，时间整体提前 1 步 | 46 spikes，时间一致 | Dense 为调度偏移 |

## 主要问题

### 1. legacy GPU 的 LIF 外部电流未生效

在 `LIF Exponential Decay` 中，CPU 首步电压从 `-65.0` 变为 `-64.9375`，而 legacy GPU 始终保持 `-65.0`，状态中的外部电流也保持 `0.0`。Voltage Jump 中 legacy GPU 也保持 `0.0`，没有消费传入的电流。

这与 Izhikevich legacy GPU 的外部电流测试不同：Izhikevich GPU 能正确消费同类 `synapse_type=3` 输入。因此问题集中在两个 LIF GPU 实现的 current staging 或同步路径，建议重点检查：

- `TimeDrivenLIF_Exponential_Decay_GPU_Interface::ProcessCurrent`
- `TimeDrivenLIF_Voltage_jump_GPU_Interface::ProcessCurrent`
- `State_GPU->AuxStateCPU` 到 `AuxStateGPU` 的输入槽布局和复制长度
- GPU kernel 中 `I_EXT_index` 对应的状态槽

### 2. GPU Voltage Jump 初始状态与 CPU 不一致

CPU Voltage Jump 使用 `V_reset=-5` 初始化；legacy GPU 实现初始化时使用 `V_rest`，本次参数下为 `0`。因此即使电流路径修复，两个后端仍需统一初始化语义。

### 3. Dense 后端差异

Dense LIF Exponential Decay/Double 与 CPU 曲线基本重合。Dense Voltage Jump 的初始状态和输入更新顺序与 CPU 不完全一致，最大曲线差异约 `4.75`。Dense PoissonRate 产生相同数量的 spike，但 spike 时间整体比 CPU/legacy GPU 提前 1 个仿真步。

## 图表和数据

- 电压曲线：[other_neuron_voltage_curves.png](D:/testfile/other_neuron_backend_comparison/other_neuron_voltage_curves.png)
- 电压矢量图：[other_neuron_voltage_curves.svg](D:/testfile/other_neuron_backend_comparison/other_neuron_voltage_curves.svg)
- Poisson raster：[poisson_backend_raster.png](D:/testfile/other_neuron_backend_comparison/poisson_backend_raster.png)
- 原始结果：[other_neuron_backend_comparison.json](D:/testfile/other_neuron_backend_comparison/other_neuron_backend_comparison.json)
- 测试脚本：[other_neuron_backend_comparison.py](D:/testfile/repro/other_neuron_backend_comparison.py)
