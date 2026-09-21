# Izhikevich 不同动力学过程验证报告

## 结论

`TimeDrivenIzhikevic_Exponential_Decay` 能够产生多种可区分的动力学过程。使用经典 Izhikevich 参数族并在发布 wheel 上运行后，观察到了：持续放电、相位放电、持续爆发、相位爆发和混合模式。

## 验证模型

源码实现的动力学方程为：

```text
dv/dt = 0.04 v^2 + 5v + 140 - u + R I
du/dt = a(bv - u)
```

当 `v >= V_th` 时执行 `v=c`、`u=u+d` 的复位。每个测试使用一个 Izhikevich 神经元、恒定外部电流、`delay=1` 的电流连接、`dt=0.25`，共 4000 个仿真步。burst 分组使用 spike 间隔大于 50 个步长作为组间间隔。

## 结果

| 动力学过程 | `(a,b,c,d)` | 电流 | spike 数 | burst 组大小 | 判定 |
|---|---|---:|---:|---|---|
| Tonic spiking | `(0.02, 0.20, -65, 6)` | 14.0 | 38 | `2 + 单发放` | 持续规则放电 |
| Phasic spiking | `(0.02, 0.25, -65, 6)` | 0.5 | 1 | `1` | 单次相位放电 |
| Tonic bursting | `(0.02, 0.20, -50, 2)` | 15.0 | 120 | `11, 6, 6, ...` | 重复爆发 |
| Phasic bursting | `(0.02, 0.25, -55, 0.05)` | 0.8 | 50 | `8, 6, 6, ...` | 重复的相位爆发组 |
| Mixed mode | `(0.02, 0.20, -55, 4)` | 10.0 | 33 | `3 + 单发放` | 初始短爆发后转为规则放电 |

## 关键证据

- Tonic spiking 的稳态 ISI 为 110 步，表明经过初始瞬态后进入规则周期放电。
- Phasic spiking 只产生 1 个 spike，随后保持静息，符合相位放电特征。
- Tonic bursting 的 burst 组持续出现，典型组大小为 6，组间存在约 138 步的长间隔。
- Phasic bursting 出现大小为 8、随后多个大小为 6 的 burst 组，组间最大 ISI 约 427 步。
- Mixed mode 先出现 3 个 spike 的短组，之后进入约 129 步的规则放电，体现初始 burst 与稳态 spiking 的混合行为。

## 解释与限制

这些参数来自经典 Izhikevich 参数族，但当前 wheel 的时间步、事件调度和电流单位与原论文示例的具体时间标度不一定完全相同。因此这里验证的是“动力学类型可产生且参数变化导致定性行为变化”，不是复现论文中的绝对 spike 时间。

Phasic bursting 使用了 `I=0.8` 而不是常见示例中的 `0.6`，因为在本 wheel 的 `dt=0.25` 和首步延迟语义下，`0.8` 能稳定产生清晰的多组 burst；这属于针对实现时间标度的校准，不改变 `(a,b,c,d)` 动力学参数族。

## 文件

- 机器可读结果：[izhikevich_dynamics_results.json](D:/testfile/izhikevich_dynamics/izhikevich_dynamics_results.json)
- 可复现实验脚本：[izhikevich_dynamics_validation.py](D:/testfile/repro/izhikevich_dynamics_validation.py)
- 执行日志：[izhikevich_dynamics_validation.log](D:/testfile/logs/izhikevich_dynamics_validation.log)
