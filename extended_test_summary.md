# 发布 wheel 扩展阶段测试报告

测试目录：`D:\testfile`  
测试 wheel：`neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`

## 结果

| 项目 | 结果 |
|---|---|
| 200 步 heap 长时运行 | PASS |
| 200 步 timing wheel 长时运行 | PASS |
| 重复 run 与 reset | PASS |
| Dense 子网络运行和 snapshot | PASS |
| 权重保存、修改、加载回读 | PASS |
| 非法模型/配置定义 | PASS，4/4 正确拒绝 |
| 畸形同步协议输入 | PASS，2/2 正确拒绝 |
| ZeroMQ publisher 生命周期 | PASS，10/10 |
| ZeroMQ server/client 生命周期 | PASS，10/10 |

## 关键观察

- heap 和 timing wheel 在 200 步运行中均完成，无崩溃、死锁或状态访问异常。
- 重复 `run()` 能推进状态，`reset()` 后动态状态恢复到初始值且当前权重保持不变，未见状态残留。
- Dense snapshot 包含膜电位、发放历史、权重、局部到原始神经元映射等关键字段。
- 权重由 0.25 修改为 0.15 后加载保存值 0.75，读写闭环正确。
- 非法配置、非法层大小、错误 frame payload 和畸形协议请求均被拒绝。
- 10 轮同步/异步 ZeroMQ 资源均在上下文退出后正常关闭，无残留连接迹象。

## 文件

- `logs\extended_tests.log`
- `logs\communication_lifecycle.log`
- `11_regression\extended_test_results.json`
- `11_regression\extended_weights.bin`
- `repro\wheel_extended_tests.py`
- `repro\wheel_communication_lifecycle.py`

## 阶段结论

扩展阶段未发现 wheel 功能性 bug。剩余未覆盖内容主要是依赖外部数据的 handwriting/EI 完整运行，以及更大规模 GPU 压力和长时间稳定性测试。
