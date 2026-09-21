# 测试范围更新

日期：2026-09-17

根据用户要求，本次发布 wheel 验收正式排除所有 handwriting 相关测试，包括：

- handwriting_stage1
- handwriting_stage2
- handwriting_stage3
- handwriting_stage3_framework
- handwriting_stage4_different_position

这些测试不再计入未完成项，也不作为 wheel 发布阻断条件。

`ei_connectivity` 流程仍单独保留为未执行项，原因是对应外部数据集未提供。

当前验收结论：在排除 handwriting 测试后的范围内，已执行测试均通过，未发现 wheel 功能性 Bug。
