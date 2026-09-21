# 发布 wheel 压力与稳定性测试报告

测试目录：`D:\testfile`  
测试 wheel：`neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`

## 测试结果

| 测试 | 结果 | 结果摘要 |
|---|---|---|
| 重复仿真创建/运行/销毁 | PASS | 20/20，状态数量稳定为 8 |
| 中等 Dense 网络 | PASS | 2 个 Dense 子网络，每个 256 个神经元，500 步 |
| 大规模 Dense 网络 | PASS | 1024 个神经元，1000 步 |
| 长时 timing wheel | PASS | 1200 步，状态查询正常 |

## 观察

- 所有用例均正常完成，无崩溃、死锁、异常退出或状态查询失败。
- 重复创建和销毁仿真实例未出现状态数量漂移。
- Dense 子网络 snapshot 能够正常返回膜电位数据和网络规模信息。
- 测试期间出现多次 `OpenMP support is enabled` 信息，为运行时提示，不影响测试结果。

## 产物

- `stress\stress_results.json`
- `logs\stress_tests.log`
- `repro\wheel_stress_tests.py`

## 结论

当前规模和时长下未发现稳定性或压力相关 wheel Bug。后续剩余重点是补充外部 handwriting/EI 数据后的完整业务流程，以及更长时间的持续运行和实际 GPU 负载监控。
