# 权重读取与修改接口专项测试（2026-09-21）

> 状态更新：原报告中的问题 1 已通过统一 `reset()` 语义消除；问题 2 和问题 3 已修复。新 wheel 与复验结果见 `weight_api_fix_validation_20260921.md`。下文保留为旧 wheel 的缺陷记录。

## 被测 wheel

- 路径：`D:\neuronBridge\artifacts\wheel_issue12_fix\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- SHA-256：`84B7342F08262031418118C62BF52AE64D372EA7576BE4297B4B04445B6009EC`
- 测试环境：`D:\neuronBridge\artifacts\wheel_issue12_fix_testvenv`

## 网络与路径

同一混合网络分别通过快速构建和 C++ 直接流式构建运行。原始连接顺序为：

1. main -> main，初始权重 1.0
2. main -> dense，初始权重 2.0
3. dense 内部，初始权重 3.0
4. dense -> dense，初始权重 4.0
5. dense -> main，初始权重 5.0

## 正常功能结果

- `get_connection_weight`：快速/流式均正确返回 `[1, 2, 3, 4, 5]`。
- `set_connection_weight`：main 内部连接 `1.0 -> 1.5` 成功。
- `set_connection_weight`：dense 内部连接 `3.0 -> 3.5` 成功。
- `dense_subnetwork_weights` 能立即观察到 dense 修改值 `3.5`。
- main/dense 修改后再次读取结果正确。
- `save_weights` 后修改权重，再 `load_weights` 能恢复 main 和 dense 权重。
- 快速构建保存的快照可加载到流式构建，结果为 `[6.25, 2, 7.5, 4, 5]`。
- `reset(preserve_weights=True)` 同时保留 main 和 dense 修改值。
- 负索引和越界索引会抛出明确 `RuntimeError`。
- main -> dense、dense -> dense、dense -> main 路由是 `InitialOnly`，读取可用，但修改会明确报告 `connection has no mutable runtime weight`，符合当前运行时所有权设计。

## 发现的问题

### 1. `reset(preserve_weights=False)` 没有恢复 dense 权重

- 修改前：`[1, 2, 3, 4, 5]`
- 修改后：`[2.25, 2, 4.25, 4, 5]`
- reset 后实际：`[1, 2, 4.25, 4, 5]`
- 预期：`[1, 2, 3, 4, 5]`

主网恢复到初始值，dense 内部连接仍为 4.25。快速和流式路径均复现。

原因是 `ResetForNextRound` 把 `preserve_weights` 传给主网 `ResetDynamicState`，但 dense 只调用无该参数的 `Reset()`；dense reset 清理状态和学习缓存，却没有把 `d_syn_weight_` 恢复为构建时权重。

### 2. 非有限权重没有输入校验

`NaN`、`+Inf`、`-Inf` 均能通过 `set_connection_weight` 写入并原样读回。这可能让传播和学习结果永久变成非有限值。

负权重和超过 `max_weight` 的值也会被接受；当前接口没有明确边界策略，因此先记录为语义待明确项，不直接判定为实现错误。

### 3. 保存/加载失败没有传播到 Python

- 保存到不存在的父目录：C++ 向 stderr 输出失败，但 Python `save_weights()` 正常返回，文件没有生成。
- 加载不存在的文件：C++ 向 stderr 输出失败，但 Python `load_weights()` 正常返回。

当前 native wrapper 调用返回 `void` 的 `SaveWeightToFile`/`LoadWeight`，无法把底层 `reason` 转换为 Python 异常。

## 总体结论

正常路径中的权重读取、主网/dense 内部修改、统一快照保存恢复、快速/流式跨路径加载均正常。接口尚不能判定为完全通过，因为 dense 非保留 reset、非有限输入校验和文件 I/O 错误传播存在缺陷。

## 产物

- JSON：`D:\neuronBridge\artifacts\wheel_issue12_fix_validation\weight_interface\weight_interface_report.json`
- 快速构建权重快照：`fast_weights.txt`
- 流式构建权重快照：`streaming_weights.txt`
- 跨模式快照：`cross_mode_weights.txt`
