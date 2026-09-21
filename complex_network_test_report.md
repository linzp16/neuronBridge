# 复杂混合网络测试报告

测试目录：`D:\testfile\complex_networks`

## 结果概览

| 案例 | 结果 |
|---|---|
| 混合神经元 + InputCurrent + LIF + Izhikevich + Poisson + TriggerRelay + 学习规则 | FAIL，原生访问冲突 |
| 多 Dense 子网络级联 | PASS |
| InputConv → Dense/主网络混合 | PASS |
| OuterDynamic SpikeCounter 闭环 | PASS |

## 已定位问题

复杂混合案例触发 Windows 原生访问冲突：

```text
process exit code: -1073741819 (0xC0000005)
```

最小稳定复现网络：

```text
InputCurrentNeuronModel(2)
        ↓ Connection(weight=0.5, delay=1)
TimeDrivenLIF_Exponential_Decay(2)
```

随后调用：

```python
simulation.add_external_currents([0, 2, 6], [0, 1, 0], [5.0, 8.0, 5.0]).run()
```

表现：网络定义和 `Simulation.init()` 成功，进入 `run()` 后进程直接退出，没有 Python 异常栈。

对比结果：

- 仅使用 InputCurrent 层但不建立 InputCurrent→LIF 连接：通过；
- InputCurrent→LIF 连接并使用外部电流：稳定复现访问冲突；
- 增加 Izhikevich、Poisson、TriggerRelay 或学习规则不是必要条件；
- 因此当前初步根因范围为 native runtime 的 InputCurrent 输入驱动与连接传播路径。

复现脚本：`D:\testfile\repro\reproduce_input_current_connection_crash.py`

证据日志：`D:\testfile\complex_networks\minimal_input_current_connection.log`

## 其他复杂案例

- Dense 级联包含 3 个 Dense 子网络，64/64/16 规模，timing wheel、多队列和学习规则运行通过；无状态变量的 TriggerRelay Dense snapshot 返回 0 个膜电位，符合模型特征。
- InputConv 混合案例包含 grating frame、Dense 目标和主网络连接，运行通过。
- OuterDynamic 闭环包含 SpikeCounter、外部连接、timing wheel 和多队列，运行通过。

## 测试脚本和日志

- `D:\testfile\repro\complex_network_tests.py`
- `D:\testfile\repro\diagnose_mixed_crash.py`
- `D:\testfile\repro\diagnose_current_mixed.py`
- `D:\testfile\complex_networks\complex_test_results.json`
- `D:\testfile\complex_networks\case_results.json`
- `D:\testfile\complex_networks\current_mixed_diagnosis.json`
- `D:\testfile\logs\complex_network_tests.log`

## 结论

复杂混合网络测试发现一个需要修复的高优先级 native runtime 问题：InputCurrent→LIF 连接与外部电流输入组合会导致进程级访问冲突。该问题已完成最小化复现和初步边界定位；Dense、InputConv 和 OuterDynamic 复杂案例目前通过。
