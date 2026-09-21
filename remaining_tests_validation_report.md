# 修复后 wheel 剩余测试汇总

日期：2026-09-17

测试环境：`D:\testfile\venv_bugfix`

## 已完成

| 测试类别 | 结果 | 规模/覆盖 |
|---|---|---|
| 仿真模式矩阵 | PASS | heap/timing-wheel，单队列/多队列；InputConv bar/grating/plaid/file；OuterDynamic 3 类 |
| 扩展 API 回归 | PASS | 200 步长时运行、reset、Dense snapshot、权重读写、非法定义、畸形协议 |
| 压力测试 | PASS | 20 次重复运行、256+256 Dense、1024 Dense、1200 步 timing-wheel |
| 复杂混合网络 | PASS | 6 层、4 类连接、2 个学习规则；3 Dense 级联；InputConv+Dense；OuterDynamic 闭环 |
| 大规模 legacy 混合网络 | PASS | 2432 神经元、200 步、3114 个输出 spike |
| 大规模 Dense 级联 | PASS | 2560 神经元、3 个 Dense 子网络、300 步 |

## 测试脚本调整说明

复杂混合网络原始用例将 `InputCurrent` 连接错误地使用了默认 synapse type 0，已改为合法的 type 3；同时状态检查只查询有 `Neuron_State_Vector` 的 LIF/Izhikevich 神经元，避免把 Poisson/Relay 的无状态接口误当作可监控状态。修正后完整测试 4/4 PASS。

## 不执行项目

handwriting 和 `ei_connectivity` 按用户此前要求排除；当前环境也没有这些外部数据集。除此之外，本阶段可执行的剩余测试均已完成。

结果文件：

- `D:\testfile\05_simulation_modes\wheel_matrix_smoke.json`
- `D:\testfile\11_regression\extended_test_results.json`
- `D:\testfile\stress\stress_results.json`
- `D:\testfile\complex_networks\complex_test_results.json`
- `D:\testfile\complex_networks\large\large_mixed_results.json`
