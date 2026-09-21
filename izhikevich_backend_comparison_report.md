# Izhikevich CPU / Dense / GPU 后端对比报告

## 测试设置

- 模型：`TimeDrivenIzhikevic_Exponential_Decay`
- 后端：普通 CPU (`legacy_cpu`)、Dense GPU (`dense_gpu`)、legacy GPU (`legacy_gpu`)
- 时间步：`dt=0.25`
- 仿真步数：4000
- 外部电流：通过 `synapse_type=3`、`delay=1` 输入
- 为保证比较公平，三个后端显式统一 `V_th=30`

## 结果摘要

| 动力学过程 | CPU spikes | Dense spikes | GPU spikes | Dense 与 CPU 的现象 | GPU 与 CPU |
|---|---:|---:|---:|---|---|
| Tonic spiking | 38 | 38 | 38 | 波形相位逐渐偏移 | 基本重合 |
| Phasic spiking | 1 | 1 | 1 | 首次 spike 约提前 1 步 | 基本重合 |
| Tonic bursting | 120 | 118 | 120 | burst 内节奏和相位不同 | 基本重合 |
| Phasic bursting | 50 | 41 | 50 | burst 数量及相位存在差异 | 接近，峰值附近有舍入误差 |
| Mixed mode | 33 | 33 | 33 | 初始 burst 和后续相位略有差异 | 基本重合 |

普通 CPU 和 legacy GPU 的 spike 数量及主要 spike 时间完全一致；电压轨迹最大差异约 `4.61e-2`，恢复变量最大差异约 `1.61e-4`，属于 GPU/CPU 浮点与事件调度差异范围内，但在峰值附近会出现可见的小相位差。

Dense GPU 能够产生相同的定性动力学过程，但与普通 CPU 不是逐点数值等价：周期、burst 数量和峰值相位会发生变化。Dense 后端的曲线仍然表现出 tonic spiking、phasic spiking、tonic bursting、phasic bursting 和 mixed mode 五种动力学类别。

## Dense 数值差异的源码原因

本次显式统一了 `V_th=30`。这是必要的，因为 Dense Izhikevich 模型的默认阈值为 `-50`，而 legacy CPU/GPU 的默认阈值为 `30`；若不覆盖该参数，比较会被默认配置差异主导。

此外，Dense CUDA kernel 的恢复变量更新顺序为：先用更新后的 `v` 计算 `u`；legacy CPU 的 `CaculateDifferentialEquation` 使用更新前的 `V` 计算 `du/dt`。两者分别相当于：

```text
legacy CPU: u_next = u + dt * a * (b * v_old - u)
Dense GPU:  u_next = u + dt * a * (b * v_new - u)
```

这是 Dense 与 CPU 波形相位和 burst 结构产生差异的主要实现原因。该差异不是绘图误差，而是两个后端的实际数值更新公式不同。

## 判定

- 后端可用性：三种后端均成功运行。
- 动力学能力：三种后端均能产生不同 Izhikevich 动力学类型。
- CPU/GPU 一致性：legacy GPU 与普通 CPU 高度一致。
- Dense 数值一致性：Dense GPU 目前应判定为“定性一致、逐点数值不一致”，不应使用 CPU 轨迹的严格逐点误差阈值验收。

## 文件

- 对比图：[izhikevich_backend_curves.png](D:/testfile/izhikevich_backend_comparison/izhikevich_backend_curves.png)
- 矢量图：[izhikevich_backend_curves.svg](D:/testfile/izhikevich_backend_comparison/izhikevich_backend_curves.svg)
- 原始数据：[izhikevich_backend_comparison.json](D:/testfile/izhikevich_backend_comparison/izhikevich_backend_comparison.json)
- 测试脚本：[izhikevich_backend_comparison.py](D:/testfile/repro/izhikevich_backend_comparison.py)
