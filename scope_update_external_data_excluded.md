# 测试范围最终更新

日期：2026-09-17

根据用户要求，本次发布 wheel 验收排除以下外部数据流程：

- handwriting_stage1
- handwriting_stage2
- handwriting_stage3
- handwriting_stage3_framework
- handwriting_stage4_different_position
- ei_connectivity

以上数据集相关测试不执行、不计入未完成项，也不作为 wheel 发布阻断条件。

本次验收范围以已完成的 wheel 静态检查、安装导入、Python API、模型/网络定义、仿真模式、数组/队列通信、同步/异步 ZeroMQ、InputConv、Dense、OuterDynamic、学习规则、压力稳定性、公开示例、CUDA runtime 和交付完整性测试为准。

当前验收结论：在最终确认的测试范围内，已执行测试均通过，未发现 wheel 功能性 Bug。
