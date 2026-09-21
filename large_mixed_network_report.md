# 大规模混合网络测试报告

测试目录：`D:\testfile\complex_networks\large`

## 结果

| 案例 | 规模 | 配置 | 结果 |
|---|---:|---|---|
| large legacy mixed | 2432 neurons | InputSpike + InputCurrent(type=3) + LIF + Izhikevich + Poisson + TriggerRelay，timing wheel，2 queues，200 steps | PASS |
| large Dense cascade | 2560 neurons | 3 个 Dense 子网络：1024 + 1024 + 512，timing wheel，2 queues，300 steps | PASS |

## 关键结果

### large legacy mixed

- 6 个网络层；
- 5 组混合连接；
- InputCurrent 连接显式使用 `synapse_type=3`；
- 初始化：约 0.031 s；
- 运行：约 0.060 s；
- 输出 spike：3114；
- LIF 状态查询成功。

### large Dense cascade

- 3 个 Dense 子网络；
- snapshot 神经元数量：1024、1024、512；
- 初始化：约 0.005 s；
- 运行：约 0.082 s；
- Dense snapshot 全部成功。

## 记录

首次执行时状态探针错误地查询了 TriggerRelay 和 Dense 神经元的 main-network state，产生的是测试脚本查询错误；修正探针后两个大规模案例均通过。

## 文件

- `large\large_mixed_results.json`
- `logs\large_mixed_network_tests_retry.log`
- `repro\large_mixed_network_tests.py`

## 结论

在正确声明 `synapse_type=3` 的前提下，大规模 legacy 混合网络和多 Dense 级联网络均未发现新的崩溃、死锁或状态访问问题。
