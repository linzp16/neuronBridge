# 权重 API 修复与验证报告（2026-09-21）

## 修复结论

本轮完成以下接口收敛和缺陷修复：

1. 删除 `preserve_weights` 开关。Python、pybind、`Simulation`、共享 simulation helper 和主网 `Network` 的 reset 调用链均改为无参数；`reset()` 只重置动态状态，始终保留当前权重。
2. `set_connection_weight()` 拒绝 `NaN`、`+Inf` 和 `-Inf`，并通过 `RuntimeError` 返回明确原因 `connection weight must be finite`。
3. `SaveWeightToFile` 和 `LoadWeight` 改为返回底层成功状态与失败原因；pybind 将失败转换为 Python `RuntimeError`，不再仅向 stderr 打印后正常返回。

负权重和超过连接声明 `max_weight` 的有限值仍按现有语义允许写入，本轮未改变该策略。

## 新 wheel

- 路径：`D:\neuronBridge\artifacts\wheel_weight_api_fix\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- SHA-256：`0F6EB0E02B65886590C860BE8D55469A5D29A53CB5AB90F5313E2C366BA9066F`
- 大小：4,317,112 bytes
- Wheel 内容审计：PASS
- 隔离环境：`D:\neuronBridge\artifacts\wheel_weight_api_fix_testvenv`
- 实际导入位置：`D:\neuronBridge\artifacts\wheel_weight_api_fix_testvenv\Lib\site-packages\neuronbridge\__init__.py`

## 专项验证

同一混合网络分别通过快速构建和 C++ 直接流式构建进行验证，网络同时包含主网内部连接、dense 内部连接和跨边界连接。

- 快速构建模式：PASS。
- 流式构建模式：PASS。
- `reset()` 公共签名确认为 `(self)`，不再接受权重策略开关。
- main 权重在 reset 前后保持 `2.25`。
- dense 内部权重在 reset 前后保持 `4.25`。
- `NaN`、`+Inf`、`-Inf`：全部拒绝。
- 保存到不存在的父目录：抛出 `RuntimeError`。
- 加载不存在的文件：抛出 `RuntimeError`。
- 主网和 dense 权重正常读取、修改、保存、恢复：PASS。
- 快速构建保存、流式构建加载的跨模式快照：PASS。

专项 JSON：`D:\neuronBridge\artifacts\wheel_weight_api_fix_validation\weight_interface\weight_interface_report.json`

## 回归结果

- 核心自动化回归：`81 passed, 18 deselected`。
- 排除范围：Handwriting 与 `ei_connectivity`，与当前测试约定一致；二者均未执行。
- JUnit：`D:\neuronBridge\artifacts\wheel_weight_api_fix_validation\pytest_core.xml`

## 使用方式

```python
sim.reset()
```

reset 不会恢复初始权重。需要恢复某个权重快照时，应显式调用：

```python
sim.load_weights("weights.txt")
```
