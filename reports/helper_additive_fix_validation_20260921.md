# NeuronLayer helper 与 AdditiveKernalChange 修复验证

日期：2026-09-21

## 修复内容

### NeuronLayer helper

`input_spike`、`input_current`、`lif_decay`、`lif_double` 和 `poisson_rate` 现在显式接收并转发：

- `update_timestep`
- `monitored`
- `output`
- `communication_input`

这些结构字段不再进入模型 `parameters` 字典。

### AdditiveKernalChange

- 将 legacy `AdditiveKernalChange` 补成可直接实例化的学习规则。
- 增加参数构造函数。
- 增加 `LTP_tau`，默认值 `16.8`，与 dense 实现一致。
- 使用 `STDP_State` 初始化突触前 trace。
- 在 legacy `LearningRuleModelFactory` 中注册构造分支。
- `Network::CreateWeightChange` 在工厂返回空指针时抛出带规则名和索引的 `std::runtime_error`，不再继续解引用。

## 新 wheel

- 路径：`D:\neuronBridge\artifacts\wheel_issue12_fix\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- SHA-256：`84B7342F08262031418118C62BF52AE64D372EA7576BE4297B4B04445B6009EC`
- 大小：4,317,540 bytes
- Wheel 内容审计：PASS
- 隔离安装及 CUDA runtime smoke：PASS

## 验证结果

- helper 纯 Python 回归：9 passed。
- helper + nbnet + Additive 专项：22 passed。
- 完整核心回归：80 passed、18 deselected、1 个 pytest cache 权限 warning。
- 未知学习规则：抛出 `RuntimeError`，未发生空指针解引用或进程崩溃。
- `AdditiveKernalChange` 主网：快速和流式构建均通过。
- `AdditiveKernalChange` dense：快速和流式构建均通过。
- 学习规则矩阵：快速/流式初始权重均为 `8.0`，最终权重均为 `8.594202041625977`。
- helper 元数据复验：`actual_output=true`、`actual_monitored=true`、`parameters=[]`。
- 学习规则、OuterDynamic 和 10 种绘图功能综合矩阵：PASS。

## 产物

- Wheel：`D:\neuronBridge\artifacts\wheel_issue12_fix`
- 安装后验证：`D:\neuronBridge\artifacts\wheel_issue12_fix_validation`
- JUnit：`D:\neuronBridge\artifacts\wheel_issue12_fix_validation\pytest_core.xml`
- 学习/OuterDynamic/绘图结果：`D:\neuronBridge\artifacts\wheel_issue12_fix_validation\feature_matrix\learning_outer_plot_report.json`
