# NeuronBridge 发布 wheel 测试阶段汇总

测试日期：2026-09-17  
目标 wheel：`D:\neuronBridge\artifacts\wheel\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`  
测试目录：`D:\testfile`  
测试解释器：`D:\testfile\venv\Scripts\python.exe`（Python 3.12.4）

## 当前结论

已执行的 wheel 静态检查、安装导入、核心 API、模型目录、网络定义、仿真模式、InputConv、ZeroMQ 通信和迁移示例测试均通过。当前未发现 wheel 功能性 bug。

## 已完成结果

| 测试阶段 | 结果 |
|---|---:|
| wheel 静态检查 | PASS |
| Python 3.12 隔离安装 | PASS |
| native extension 导入 | PASS |
| CUDA/Dense backend 信息查询 | PASS |
| 核心 API、catalog、native smoke、release 检查 | 17 passed |
| 迁移示例、模型、网络定义、结果和通信集成 | 64 passed |
| 补充仿真/定义矩阵 | PASS |
| 全模型目录 × backend 探测 | PASS（预期不支持项按能力矩阵返回异常） |

## 覆盖情况

- 后端：legacy CPU、legacy GPU、dense GPU 查询已覆盖；运行时显示 CUDA 和 Dense runtime 已加载。
- 神经元：LIF、Izhikevich、Poisson、InputSpike、InputCurrent、TriggerRelay、EDLUT、Custom model 目录已探测。
- 仿真队列：heap、timing wheel、单队列、多队列均完成 smoke test。
- InputConv：bar、grating、plaid、file 定义已验证。
- 通信：数组/队列、同步 REQ/REP、异步 PUB/SUB 和协议非法数据测试已覆盖。
- 外部动力学：SpikeCounter、PlanarArm2DOF、Strict Matlab PlanarArm2DOF 定义已验证。
- 输出：spike、state、monitor、Dense snapshot、权重 I/O 和绘图相关测试已覆盖。

## 跳过项

迁移示例测试中有 5 项跳过，原因都是外部数据未安装，不是 wheel 失败：

- handwriting_stage1 输入数据缺失；
- handwriting_stage2 输入数据缺失；
- handwriting_stage4_different_position 输入数据缺失；
- handwriting_stage3_framework 输入数据缺失；
- ei_connectivity 输入数据缺失。

若需要完成这些项目，需设置 `NEURONBRIDGE_DATA_ROOT` 或补充对应数据包。

## 环境问题记录

第一次 pytest 执行因系统临时目录权限不足产生 3 个 setup error；改用 `D:\testfile\pytest_tmp` 后 17/17 全部通过。该问题属于测试环境权限，不属于 wheel 缺陷。

补充多队列测试出现 OpenMP 队列数超过 CPU 核心数的提示，runtime 自动降至 CPU 核心数；测试仍通过，该提示记录为环境/配置提示。

## 结果文件

- `01_wheel_inspection\wheel_inspection.json`
- `02_install_import\`：隔离环境安装结果
- `03_catalog_model\catalog_matrix.json`
- `05_simulation_modes\wheel_matrix_smoke.json`
- `logs\core_release_pytest_retry.log`
- `logs\examples_pytest.log`
- `logs\examples_pytest_reasons.log`
- `logs\wheel_matrix_smoke.log`
- `summary\core_release_junit_retry.xml`
- `summary\examples_junit.xml`
- `repro\wheel_matrix_smoke.py`

## 下一步

1. 如提供 handwriting/EI 外部数据，执行 5 个跳过项。
2. 继续进行更长时间运行、边界规模和资源生命周期测试。
3. 对通过的通信组合执行重复运行和异常关闭回归。
4. 只有在上述扩展测试完成后，再给出最终发布建议。
