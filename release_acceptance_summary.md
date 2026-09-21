# 发布 wheel 验收收尾报告

测试目录：`D:\testfile`

## 已完成测试

### 官方 CUDA/Dense runtime 验证

验证脚本：`validate_neuronbridge_wheel_cuda_runtime.py`

结果：PASS。

- native extension 已加载；
- CUDA 已启用；
- Dense CUDA runtime 已激活；
- 未检测到源码或 build 目录泄漏到安装环境的 `sys.path`；
- `CustomRStdpV1` 权重发生变化；
- `CustomRStdpPersistentV1` 权重发生变化；
- `CustomPairStdpV1` 权重发生变化；
- 三种规则均产生有效 firing trace。

报告：`cuda-runtime-report.json`

### 外部数据清单

已检查 manifest 中的 6 个数据集：

- handwriting_stage1
- handwriting_stage2
- handwriting_stage3
- handwriting_stage3_framework
- handwriting_stage4_different_position
- ei_connectivity

当前 6 个数据集目录均不存在，文件数均为 0。完整数据业务测试暂时无法执行，已明确标记为环境前置条件缺失，不判定为 wheel 缺陷。

清单：`external_data_inventory.json`

## 当前总体结论

在当前可用环境和数据条件下，wheel 已通过静态检查、安装导入、核心 API、模型/网络定义、通信协议、同步/异步 ZeroMQ、Dense/CUDA、学习规则、长时运行、压力、生命周期和公开示例验收。

目前未发现 wheel 功能性 Bug。

## 剩余工作

仅剩外部 handwriting/EI 数据包驱动的完整业务流程测试。获得数据包并设置 `NEURONBRIDGE_DATA_ROOT` 后，应重新执行迁移示例中的 5 个跳过用例，并将结果追加到 `D:\testfile`。
