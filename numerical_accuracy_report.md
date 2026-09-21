# 神经元模型数值准确性复核报告

## 结论

Voltage Jump 和 Izhikevich 的首轮误差不是 native 方程错误，而是测试参考模型没有复现运行时的事件调度语义。修正参考模型后，5 个模型全部通过：

| 模型 | 最大绝对误差 | 结果 |
|---|---:|---|
| `TimeDrivenLIF_Exponential_Decay` | `5.08e-6` | PASS |
| `TimeDrivenLIF_Exponential_double` | `5.08e-6` | PASS |
| `TimeDrivenLIF_Voltage_jump` | `3.82e-7` | PASS |
| `TimeDrivenIzhikevic_Exponential_Decay` | `0`（float32 对照） | PASS |
| `PoissonRate` | spike 时间和 ID 完全一致 | PASS |

## Voltage Jump 误差来源

模型源码的方程为：

```text
dV/dt = R * I_EXT / tau + (V_rest - V) / tau
```

初始状态为 `V=-5`，输入电流为 `5`，`tau=10`。第一次调用 `sim.run(1)` 时，事件调度会先执行一个初始时间点更新，再处理 `delay=1` 的外部电流事件。因此首轮实际过程是：

```text
V=-5.0 --无输入子步--> -4.5 --输入 I=5 子步--> -3.55
```

之前的参考脚本只执行了一个“有输入”积分步，预测为 `-4.5`，由此产生约 `4.64` 的累计误差。修正为“首次 run 包含初始化更新，后续 run 各包含一个模型更新”后，最大误差降为 `3.82e-7`。

## Izhikevich 误差来源

源码方程为：

```text
dv/dt = 0.04 V^2 + 5 V + 140 - U + R * I
du/dt = a * (b V - U)
```

源码还将 `LastSpike` 初始化为 `10000`。第一次运行的实际事件顺序为：首个初始化更新不包含 `delay=1` 的输入电流；外部电流到达后执行包含 `I=5` 的更新；后续每次 `sim.run(1)` 执行一个模型更新。

实际首轮状态 `V=-65.03999328613281`、`U=-13.01200008392334` 与上述两阶段过程一致。之前参考实现只模拟了一个阶段，导致阈值附近轨迹整体错位，并把事件调度误差误判成模型数值误差。

此外，Izhikevich 使用非线性方程，必须采用与 C++ 相同的 `float32` 运算顺序进行严格比较。改用 float32 参考后，电压和恢复变量最大误差均为 `0`。

## 源码依据

- `NeuralModel/src/TimeDriven/TimeDrivenLIF_Voltage_jump.cpp`：Voltage Jump 微分方程和外部电流读取。
- `NeuralModel/src/TimeDriven/TimeDrivenIzhikevic_Exponential_Decay.cpp`：Izhikevich 方程、阈值复位和 `LastSpike` 判断。
- `Intergration/inc/FixStep/ForwardEulerMethod.h`：积分更新顺序。
- `NeuralModel/src/NeuralStateVector/Neuron_State_Vector.cpp`：`LastSpike=10000` 初始化。

## 文件

- 结果：[numerical_accuracy_results.json](D:/testfile/numerical_accuracy/numerical_accuracy_results.json)
- 测试脚本：[numerical_accuracy_tests.py](D:/testfile/repro/numerical_accuracy_tests.py)
- 最终日志：[numerical_accuracy_tests_final2.log](D:/testfile/logs/numerical_accuracy_tests_final2.log)
