# neuronBridge Windows wheel 构建与验证报告

- 日期：2026-09-21
- 源码提交基线：`67c1683`（包含当前工作区未提交修改）
- Python ABI：CPython 3.12
- 平台：Windows x86-64

## 构建产物

- Wheel：`D:\neuronBridge\artifacts\wheel_interface_coverage\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- 大小：4,317,300 bytes
- SHA-256：`CC1C4E417841F8C681AE7B516F06AB770D98DAE64C97FBF6A43DF0D37E66EDD0`
- 内容审计：通过。扩展模块、13 个原生 DLL、CUDA runtime、Pinocchio 与 ZMQ 依赖均已打包，未发现缺失或不安全路径。

## 独立安装验证

Wheel 已安装到新建虚拟环境：

`D:\neuronBridge\artifacts\wheel_interface_coverage_testvenv`

导入路径确认来自该虚拟环境的 `site-packages`，未误用源码目录或旧安装。`Simulation.reset` 签名为 `(self)`；9 个暂缓开放的实时接口均带有 `Temporarily deferred experimental API` 标记。

## 测试结果

| 范围 | 结果 | 说明 |
|---|---:|---|
| 核心回归测试 | 通过 | 92 passed，18 deselected，5.55 s |
| 权重读取、修改、保存、加载、reset | 通过 | fast 与 streaming 均通过；reset 保持主网和 dense 权重；拒绝 NaN/Inf |
| 学习规则 | 通过 | 7 种规则的 fast/streaming 构建与结果一致性通过，包括 `AdditiveKernalChange` |
| OuterDynamic | 通过 | 3 种模型在 fast/streaming 下通过 |
| 绘图 | 通过 | 生成 spike raster、state、weight、OuterDynamic、InputConv 等 10 张图 |
| CPU/GPU/dense GPU | 通过 | 三后端状态有限且可产生预期 spike；GPU 后端就绪 |
| 同步 REQ/REP | 通过 | heap 与 timing wheel 队列均完成 5 个通信窗口往返 |

核心回归按既定范围排除了 `handwriting` 与 `ei_connectivity` 相关用例。9 个实时运行接口仍按计划标记为暂缓开放，不计入本轮发布门禁。

## 证据文件

- JUnit：`D:\neuronBridge\artifacts\wheel_interface_coverage_validation\pytest_interface_expansion.xml`
- 权重接口：`D:\neuronBridge\artifacts\wheel_interface_coverage_validation\weight_interface\weight_interface_report.json`
- 学习规则、OuterDynamic 与绘图：`D:\neuronBridge\artifacts\wheel_interface_coverage_validation\learning_outer_plot\learning_outer_plot_report.json`
- 三后端：`D:\neuronBridge\artifacts\wheel_interface_coverage_validation\triple_backend.json`
- Wheel 内容：`D:\neuronBridge\artifacts\wheel_interface_coverage\wheel-contents.json`

## 结论

本轮 Windows wheel 构建成功，并通过当前发布范围内的独立安装验证与功能回归，可进入后续完整应用测试。实时运行实验接口未作为稳定接口开放。
