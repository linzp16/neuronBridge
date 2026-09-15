# neuronbridge 长期方案记忆库

本文档是 `network_release` 项目 Python 库化工作的长期基准。后续实现、重构、打包和示例迁移应优先对齐本文档，避免路线漂移。

最新更新：2026-09-07。第三阶段继续推进，真实 `Simulation` 生命周期绑定、dense snapshot/reset、InputConv monitor、通用 output spike 读取、主网 neuron state 批量读取、InputConv output/input 快照和 rate heatmap 入口已完成；第四阶段已迁移 dense/no-debug/EI/pattern-motion/my_first_app/handwriting_stage1/2/3/4/cerebellum/ZMQ 首批示例，并完成多条 C++/Python baseline 对齐；`handwriting_stage3` NumPy full runtime 的 `test/test_d` 和训练主仿真输出已完成 C++ full baseline 对齐。2026-09-07 已完成新加入 InputConv communication driver 的 Python 绑定第一版：动态帧队列、frame source 绑定、同步/异步 ZMQ frame source 注册和 Python wire protocol helper 已进入统一 `Simulation` API。

## 1. 项目定位

`neuronbridge` 是 `network_release` C++/CUDA 神经仿真项目的 Python 用户入口。它不是重写仿真器，也不是为每个 C++ 示例程序单独开一个 Python 通道，而是通过统一的 Python API 桥接现有 C++ 仿真核心，并在 Python 层提供结果读取、分析和可视化能力。

核心定位：

- C++/CUDA 继续负责高性能仿真核心。
- Python 负责用户 API、配置表达、示例迁移、数据分析和可视化。
- 示例 exe 应逐步迁移为 Python 示例，但不应为每个示例设置专用 native 入口。
- Python 用户主要通过 `import neuronbridge as nb` 使用库。

## 2. 已确认的关键决策

- Python 包名：`neuronbridge`。
- Python native 模块：`neuronbridge._core`。
- C++ 内部命名空间：保持 `npgr`，不因 Python 包名而强行重命名。
- 绑定技术：使用 `pybind11`，不长期维护手写 CPython C API。
- Python 发布基线：Python `>=3.12,<3.13`。
- 构建开关：`NR_ENABLE_PYTHON` 默认 `ON`。
- 仿真模式：当前工程按 GPU-only 现实推进，不引入新的 CPU 旁路承诺。
- 发布方向：用户应主要使用 Python 版本，依赖问题要在日常构建和 CI 中尽早暴露。
- 结果读取：DebugMonitor 大文件默认 lazy/chunked 读取，不默认一次性载入 pandas。
- 发布包：正式发布不能要求最终用户安装 MSVC、CMake、pybind11、Pinocchio、ZeroMQ 或手工配置 native DLL 路径。

## 3. 当前实现状态

已完成：

- 新增 `network_release/python` 子工程。
- 顶层 CMake 已接入 `NR_ENABLE_PYTHON`，默认 `ON`。
- 已创建 pybind11 native 扩展入口 `neuronbridge._core`。
- 已绑定 `npgr::DebugMonitorConfig`。
- 已暴露 DebugMonitor component kind enum。
- 已暴露 `backend_info()` / `get_build_info()`。
- 已实现 Python 侧 `DebugMonitorConfig` dataclass。
- 已实现第二阶段第一版 Python 友好描述对象：
  - `Network`
  - `NeuronLayer`
  - `Connection`
  - `LearningRule`
  - `SimulationConfig`
  - `OuterDynamic`
  - `OuterDynamicConnection`
  - `InputConv`
  - `FeedbackProductEncoding`
  - `FeedbackSingleEncoding`
- 已实现 pybind11 侧 bridge 容器：
  - `neuronbridge._core.NetworkDescription`
  - `neuronbridge._core.SimulationConfig`
- 已实现 Python 描述对象到现有 C++ description 结构的转换入口，包括：
  - `NeuronLayerDescription`
  - `ConnectionDescription`
  - `LearningRuleDescription`
  - `OuterDynamicDescription`
  - `OuterDynamicConnectionDescription`
  - `InputConvDescription`
- 已新增精确 native 参数类型标记：
  - `int32()`
  - `float32()`
  - `float64()`
  - `int32_list()`
  - `float32_list()`
  - `float64_list()`
  - `float32_array3()`
  - `float32_array4()`
- 已补充模型专用便捷构造器：
  - `OuterDynamic.spike_counter()`
  - `OuterDynamic.planar_arm_2dof()`
  - `InputConv.v1_bar()`
  - `InputConv.v1_grating()`
  - `InputConv.v1_plaid()`
  - `InputConv.v1_file()`
  - `NeuronLayer.poisson_rate()`
- 已实现 `DebugMonitorResult`，支持懒读取 DebugMonitor 输出目录。
- 已实现初步可视化 API：
  - `plot_spike_raster()`
  - `plot_state_trace()`
- 已完成可视化扩展第一阶段：
  - `plot_weight_trace()`
  - `plot_weight_summary()`
  - `plot_weight_distribution()`
  - `plot_outer_dynamic_trace()`
  - `plot_outer_dynamic_tracking_error()`
  - `plot_outer_dynamic_phase_plane()`
  - `plot_outer_dynamic_torque()`
  - 设计记录：`python/VISUALIZATION_API_PLAN.md`
- 已完成 OuterDynamic 多实例 DebugMonitor schema 升级：
  - `outer_dynamic_state.csv` 现包含 `component_index` 与 `component_name`。
  - C++ 按 OuterDynamic 实例保存最新状态并逐实例采集。
  - OuterDynamic 绘图 API 支持 `outer_dynamic=` 名称或索引筛选，旧四列 CSV 仍可读取。
  - `Simulation.outer_dynamic_states()` 提供所有已上报实例的实时快照；保留单实例兼容入口。
- 已提供示例迁移骨架：
  - `python/examples/dense_debug_monitor.py`
- 已提供第二阶段描述对象示例：
  - `python/examples/phase2_network_description.py`
- 已补充审批说明：
  - `python/BRIDGE_REVIEW.md`
- 已补充发布策略：
  - `python/PACKAGING.md`

已验证：

- Python 3.12.14 本地验证环境可用：
  - `D:\code_optimized\network_release\.conda-neuronbridge-py312`
- Python 工具安装位置：
  - `D:\code_optimized\network_release\.conda-neuronbridge-py312\Lib\site-packages`
- 已安装并验证：
  - `pybind11 3.1.0`
  - `pytest 9.1.1`
  - `scikit-build-core 1.0.3`
  - `numpy 2.5.2`
- 使用 VS2022 Community、MSVC 19.41、CUDA 12.6 构建通过。
- 生成过 Python 3.12 native 扩展：
  - `python/src/neuronbridge/_core.cp312-win_amd64.pyd`
- Python 3.12 导入验证通过：
  - `native_extension_loaded=True`
  - `binding=pybind11`
- Python 网络描述到 native bridge 容器的往返验证通过：
  - `Network.to_native()`
  - `NetworkDescription.to_dict()`
  - `SimulationConfig.to_native()`
- `OuterDynamic` / `OuterDynamicConnection` / `InputConv` 到 native bridge 容器的往返验证通过。
- `float64_list()` 到 C++ `std::vector<double>` 的类型保留验证通过。
- `python/examples/phase2_network_description.py` 可运行并输出 native bridge summary。
- Python 已可构造真实 C++ `Simulation`，并支持：
  - `Simulation.init()`
  - `Simulation.run()`
  - `Simulation.reset()`
  - `Simulation.add_external_spikes()`
  - `Simulation.add_external_currents()`
  - `Simulation.output_spikes()`
  - `Simulation.neuron_state()`
  - `Simulation.neuron_states()`
  - `Simulation.enable_debug_monitor()`
  - `Simulation.disable_debug_monitor()`
  - `Simulation.flush()`
  - `Simulation.result()`
  - `Simulation.get_connection_weight()`
  - `Simulation.set_connection_weight()`
  - `Simulation.dense_subnetwork_count`
  - `Simulation.dense_subnetwork_name()`
  - `Simulation.find_dense_subnetwork()`
  - `Simulation.dense_subnetwork_weights()`
  - `Simulation.dense_subnetwork_snapshot()`
  - `Simulation.reset_dense_subnetwork()`
  - `Simulation.set_dense_subnetwork_full_firing_export_enabled()`
  - `Simulation.input_conv_count`
  - `Simulation.input_conv_output_count()`
  - `Simulation.input_conv_output()`
  - `Simulation.input_conv_input()`
  - `Simulation.input_conv_rate_maps()`
  - `Simulation.plot_input_conv_rate_heatmap()`
  - `Simulation.enable_input_conv_monitor()`
  - `Simulation.disable_input_conv_monitor()`
- 已提供第三阶段最小运行示例：
  - `python/examples/phase3_run_simulation.py`
- 已提供 dense/InputConv baseline 迁移示例：
  - `python/examples/dense_subnetwork_export.py`
  - `python/examples/inputconv_poisson_dense.py`
  - `python/examples/dense_mixed.py`
  - `python/examples/dense_mixed_current.py`
  - `python/examples/dense_run_no_debug.py`
  - `python/examples/dense_subnetwork_smoke.py`
  - `python/examples/ei_program_dense_subnetwork.py`
  - `python/examples/ei_program_main.py`
  - `python/examples/pattern_motion.py`
  - `python/examples/my_first_app.py`
  - `python/examples/handwriting_stage1.py`
  - `python/examples/handwriting_stage2.py`
  - `python/examples/handwriting_stage3.py`
  - `python/examples/handwriting_stage4_different_position.py`
- 已完成首批 C++/Python baseline 对齐：
  - `simulation_weight_smoke`
  - `outer_dynamic_spike_counter_smoke`
  - `gpu_dense_subnetwork_export_real` 的核心 API 行为
  - `inputconv_poisson_dense_real` 的统一 Simulation + DebugMonitor 行为
  - `dense_mixed` 的 dense->main route、layout flags 和 readout conductance 行为
  - `dense_mixed_current` 的 dense runtime 状态和 output spike 行为
  - `dense_no_debug` 的无 DebugMonitor 运行、output spike count 和 readout state 行为
  - `dense_subnetwork_smoke` 中可通过公开 Python API 表达的 dense ownership/dense-to-dense 构图行为
  - `ei_program_main` 的主网 EI 运行和 E/I spike totals 行为
  - `pattern_motion` 的 InputConvV1 file stimulus -> dense Poisson/LIF subnetwork 构图、运行、InputConv rate 和 snapshot 统计通路
  - `handwriting_stage3_framework` 的 framework 入口输出契约
  - `cerebellum_framework` 的统一 API 构网、StrictMatlabPlanarArm2DOFOuterDynamic smoke 和 latest state 读取
  - `zmqcommunication_client` 的本地仿真迁移、协议 pack/unpack 和 optional native async ZMQ driver 注册
- pytest 通过：
  - `29 passed`

## 4. 不应跑偏的边界

后续实现应避免以下方向：

- 不要把每个 C++ 示例 exe 包装成一个独立 Python native 函数。
- 不要让 Python API 直接照搬 C++ 内部结构，尤其是复杂、易变的 description 结构。
- 不要在 Python 层重写仿真核心算法。
- 不要新增与现有 GPU runtime 并行的临时仿真路径。
- 不要把 DebugMonitor 的 GB 级 CSV 默认整体读入内存。
- 不要把开发期 venv/conda 环境当成最终用户发布包。
- 不要要求最终用户安装 MSVC、CMake、pybind11 或手动寻找 DLL。
- 不要把 `npgr` C++ namespace 改名作为 Python 化的前置条件。

## 5. 目标 Python API 形态

最终用户 API 应尽量稳定、简洁，并与 C++ 内部实现解耦。推荐形态：

```python
import neuronbridge as nb

network = nb.Network()
network.add_layer(nb.NeuronLayer(...))
network.connect(nb.Connection(...))

sim = nb.Simulation(
    network=network,
    config=nb.SimulationConfig(
        steps=1000,
        timestep=0.1,
        queues=1,
    ),
)

sim.enable_debug_monitor(
    nb.DebugMonitorConfig(
        output_dir="debug_monitor",
        sample_interval_steps=1,
        neuron_ids=[0, 1, 2],
    )
)

sim.run()
sim.flush()

result = sim.result()
result.plot_spike_raster()
result.plot_state_trace(neuron_id=0, field_name="voltage")
```

注意：这仍是最终目标形态。当前已经完成 `Network` / `NeuronLayer` / `Connection` / `LearningRule` / `OuterDynamic` / `OuterDynamicConnection` / `InputConv` / `SimulationConfig` 的 Python 描述层，以及到 native bridge 容器的转换；第三阶段已完成真实 `Simulation` 生命周期、dense snapshot/reset、InputConv monitor、output spike 读取和首批 baseline 对齐。后续重点转向更多示例迁移、可视化增强和发布打包。

## 6. 阶段计划

### 阶段 1：桥接骨架与结果读取

状态：已基本完成。

目标：

- 建立 `neuronbridge` 包结构。
- 建立 pybind11 native 模块。
- 接入 CMake。
- 默认开启 Python 构建。
- 支持 DebugMonitor 输出读取和基础可视化。

### 阶段 2：Python 友好的描述对象

状态：已完成。

目标：

- 新增稳定的 Python 用户描述层：
  - `Network`
  - `NeuronLayer`
  - `Connection`
  - `LearningRule`
  - `OuterDynamic`
  - `InputConv`
  - `SimulationConfig`
- 在 C++/binding 层实现从 Python 描述对象到现有 C++ description 结构的转换。
- 避免把 `NeuronLayerDescription`、`ConnectionDescription` 等内部结构原样暴露为主要用户 API。

已完成：

- Python 侧：
  - `Network`
  - `NeuronLayer`
  - `Connection`
  - `LearningRule`
  - `SimulationConfig`
  - `OuterDynamic`
  - `OuterDynamicConnection`
  - `InputConv`
  - `FeedbackProductEncoding`
  - `FeedbackSingleEncoding`
  - native 参数类型标记函数：`int32()`、`float32()`、`float64()`、`int32_list()`、`float32_list()`、`float64_list()`、`float32_array3()`、`float32_array4()`
- C++/pybind11 侧：
  - `NetworkDescription`
  - `SimulationConfig`
  - Python 参数字典到 `std::map<std::string, boost::any>` 的转换。
  - layer、connection、learning rule、outer dynamic、outer dynamic connection、input conv 到 C++ description list 的追加接口。
- 示例侧：
  - `phase2_network_description.py` 展示统一 Python API 如何表达普通网络、OuterDynamic、InputConv 和 typed parameters。

最新验收结果：

- Python 3.12 下 `pytest network_release\python\tests -q` 通过，结果为 `8 passed`。
- `neuronbridge_core` 已用 VS2022 + CMake Release 构建通过。
- `OuterDynamic` / `OuterDynamicConnection` / `InputConv` native 往返验证通过。
- `float64_list()` 可保持为 C++ `std::vector<double>`，避免 `PlanarArm2DOFPinocchio` 这类模型参数类型被误降为 float32。
- `phase2_network_description.py` 可运行并输出 native bridge summary。

留到第三阶段处理：

- 与真实 `Simulation` 构造器的参数对接。
- 与 C++ 示例中的复杂 description 构造逻辑做逐项 baseline 对齐。
- 根据第三阶段运行验证结果继续补充模型专用便捷构造器。

验收标准：

- 可以用 Python 构建一个最小网络描述。
- 转换后的 C++ description 与现有 C++ 示例构造逻辑一致。
- 不改变现有 C++ smoke target 行为。

### 阶段 3：绑定 Simulation 生命周期

状态：主体已完成，继续扩展更多示例和结果访问器。

目标：

- 绑定或封装 C++ `Simulation` 生命周期。
- 支持：
  - `run()`
  - `reset()`
  - `flush()`
  - `enable_debug_monitor()`
  - `disable_debug_monitor()`
  - `add_external_spikes()`
  - `add_external_currents()`
  - `get_connection_weight()`
  - `set_connection_weight()`
  - `get_dense_subnetwork_snapshot()`
  - `get_dense_subnetwork_weights()`
  - `reset_dense_subnetwork()`
  - `InputConv` monitor enable/disable
  - `output_spikes()`
  - `neuron_state()`
  - `neuron_states()`

验收标准：

- Python 能运行一个真实 C++ 仿真，而不是只读取已有输出。
- Python 运行结果与对应 C++ smoke/example baseline 对齐。
- native 生命周期、所有权和异常转换清晰。

已完成：

- C++/pybind11 侧新增 `neuronbridge._core.Simulation` 包装类。
- Python 侧 `neuronbridge.Simulation` 从占位类变成真实 facade。
- 可从 `Network.to_native()` 和 `SimulationConfig.to_native()` 构造真实 C++ `Simulation`。
- 已支持初始化、运行、reset、外部 spike/current 注入、DebugMonitor enable/flush/disable/result、连接权重读写、dense subnetwork 基础查询。
- 已支持 dense subnetwork 结构化 snapshot、按名字查找、权重读取、单个 dense reset、full firing 导出开关。
- 已支持 InputConv 数量、输出数量查询，以及按 index/name 启停 InputConv monitor。
- 已支持 `Simulation.input_conv_output(index)` 和 `Simulation.input_conv_input(index)`，通过 C++ `InputConvModel::ExportMonitorOutput/Input` 通用读取最新 InputConv 输出/输入快照。
- 已支持 `Simulation.input_conv_rate_maps()` 和 `Simulation.plot_input_conv_rate_heatmap()`，用于把 V1 rate buffer 拆成方向热图并可视化。
- 已评估 C++ 新增 InputConv communication driver：`InputConvFrameDriver` / `InputConvFrameQueue` / `AsyncInputConvFrameDriver` 提供动态帧抽象和本地队列；`ZMQInputConvFrameDriver` 提供同步 request/response 拉帧；`ZMQAsyncInputConvFrameDriver` 提供异步 pub/sub 帧流订阅；`Simulation` 新增 named frame source 注册、InputConv 绑定、外部帧推入、队列清空和 ZMQ frame source 注册入口。
- 已支持 `Simulation.output_spikes()` 读取 `ArrayOutputSpikeDriver` 当前 buffered output spikes；该接口遵循 C++ 现有行为，读取后会清空缓冲区。
- 已支持 `Simulation.neuron_state(original_neuron_id)` 读取仍保留在主网络中的 neuron state、main id、local id、last spike/update 等轻量状态。
- 已支持 `Simulation.neuron_states(original_neuron_ids)` 批量读取主网络 neuron 状态，用于 `ei_program_main` 这类每步扫描整个人口的示例，避免 Python 逐 neuron 跨绑定调用。
- 已支持 `Simulation.outer_dynamic_state()` 读取 C++ `GetLatestOuterDynamicState` 的最新 2-DOF joint state，字段包括 `time_step`、`q`、`qv`、`qdd`、`q_des`、`qv_des`、`tau_total`。
- 已支持 `Simulation.add_zmq_async_input_output_spike_driver()`，用于通过统一 Python API 注册 C++ async ZMQ spike driver，不为 ZMQ 示例新增专用 native 入口。
- 已补充 `float32_array3()` / `float32_array4()`，用于 `random_sigma` 等 C++ `std::array<float,N>` 参数。
- 已补充 `NeuronLayer.poisson_rate()`，用于通过统一 Python API 表达 dense PoissonRate 子网络。
- 已补充 `OuterDynamic.strict_matlab_planar_arm_2dof()`，用于对齐 `projects/cerebellum_framework` 当前使用的 `StrictMatlabPlanarArm2DOFOuterDynamic` 模型。
- `python/examples/phase3_run_simulation.py` 运行通过，可产生 DebugMonitor 输出目录。
- `python/examples/dense_subnetwork_export.py` 运行通过。
- `python/examples/inputconv_poisson_dense.py` 运行通过，可产生 DebugMonitor 输出目录。
- `python/examples/dense_mixed.py` 运行通过，可产生 DebugMonitor 输出目录；Python 迁移版采用当前 splitter 接受的表达方式：dense 子网络层不再直接标记为 `output=True`，而是通过 dense->main route 输出到主网络 readout。
- `python/examples/dense_mixed_current.py` 运行通过，输出 `buffered_spike_count=2`，并与 C++ baseline 对齐为 `cell=2,t=4` 和 `cell=3,t=4`。
- `python/examples/dense_run_no_debug.py` 运行通过，输出 `buffered_output_spike_count=4`，readout neuron 8/9 的 g 值与 C++ baseline 对齐为 `2.945720370614717e-32`。
- `python/examples/dense_subnetwork_smoke.py` 运行通过，迁移 C++ smoke 中可通过公开 `Simulation` API 表达的 ownership reject、dense output reject、main/dense TriggerRelay 构图和 dense-to-dense 构图检查。
- `python/examples/ei_program_dense_subnetwork.py` 运行通过，可读取 `D:/computing_ref/Demo/Python_plot` connectivity 数据并构造 4000-neuron dense EI 网络；C++ baseline 自身 spike totals 非确定，Python 示例以结构、可运行性、dense count 和活动量级作为迁移验收。
- `python/examples/ei_program_main.py` 运行通过，可读取相同 connectivity 数据并构造 4000-neuron 主网 EI 网络；完整 2000-step 输出与 C++ baseline 对齐为 `total_e_spikes=7545`、`total_i_spikes=1861`。
- `python/examples/pattern_motion.py` 已迁移为 Python builder + runner：通过统一 `Network` / `NeuronLayer.poisson_rate()` / `InputConv.v1_file()` / `Simulation` API 表达 InputConvV1 文件刺激到 dense subnetwork 的链路，不新增单示例 native 入口。当前 quick 规模已完成 native 初始化、运行和 dense snapshot 统计验证；full 32x32x8 长跑已完成新版 C++ unified baseline 与 Python 对比，方向 winner 指标已收敛：`v1_spike_winner=2`、`v1_rate_winner=2`、`v1_current_update_mean_winner=2`、`cds_winner=2`、`pds_winner=2`、`pds_fs_winner=6`、`lip_winner=2`。Python 示例可无 matplotlib 生成 `inputconv_rate_heatmap.bmp`。
- `python/examples/my_first_app.py` 已迁移为最小用户入门示例：通过统一 `Network` / `NeuronLayer` / `Connection` / `Simulation` API 表达 4 个 InputSpike 到 4 个 LIF 输出神经元的 all-to-all 网络；C++ baseline 和 Python 示例均输出 `mode=my_first_app`、`steps=100`、`dt_ms=0.1`、`input_count=4`、`lif_count=4`，Python 侧额外通过统一 API 读取 `buffered_output_spike_count=40`。
- `python/examples/handwriting_stage1.py` 已迁移：Python 侧复现 C++ `PlanarArm2DStage1` 的文本序列读取、逆运动学、梯度、期望力矩、正向动力学重放和 TSV 输出，不新增单示例 native 入口；三个 stroke 均为 400 samples，summary 与 C++ baseline 在 `1e-13` 容差内对齐。
- `python/examples/handwriting_stage2.py` 已迁移：Python 侧复现 C++ `HandwritingStage2` 的 binary spike/weight 输入读取、CM forward、population decode、forward replay 和 TSV 输出；完整 400-window、3-stroke 运行通过，summary 与 C++ baseline 最大差异约 `3.6e-12`。
- `python/examples/handwriting_stage4_different_position.py` 已迁移：Python 侧复现 9 个位置偏移、力矩读取、位移重放、combined path 和 summary 输出；stroke 1 与 C++ baseline summary 最大差异约 `2.1e-10`，来源于 C++ 文本默认输出精度。
- `python/examples/handwriting_stage3.py` 已从输入/路径 scaffold 推进为 NumPy runtime：支持 `test/test_d/train_g/train_d` 模式的 90,000-step 多群体 recurrent 主循环、CM/BG/MM/E/I 更新、population decode、replay 和 `QQ/pop_spk/replay_path/target_path/summary/w_mmbg*_final` 输出；full test 与 C++ baseline 对齐，双方 `pop_spk.tsv` 逐 cell 完全一致，`total_population_spikes=30797`；`QQ.tsv` 最大差异约 `5e-8`，`summary.tsv` 最大差异约 `1.4e-11`，`replay_path.tsv` / `target_path.tsv` 最大差异约 `5e-7`。`train_g` 主输出与 C++ baseline 对齐，`pop_spk.tsv` 逐 cell 完全一致，`total_population_spikes=30614`；`train_d` 主输出与 C++ baseline 对齐，`pop_spk.tsv` 逐 cell 完全一致，`total_population_spikes=23471`，且 `w_mmbg2_final.tsv` 完全一致。训练权重诊断已确认剩余差异来自隐藏 BG/MM spike 轨迹的训练窗口 update-count 分叉，根因是 NumPy 向量化矩阵乘法与 C++ 手写逐列 double 累加在阈值附近产生细微分叉；若要求 `w_mmbg1_final.tsv` 逐元素完全一致，应新增 C++/Numba parity kernel，而不是把 Python 主 runtime 改成不可维护的全手写循环。
- `python/examples/handwriting_stage3_framework.py` 已迁移：保留 C++ framework 版本的命令行形态和输出契约，复用已对齐的 stage-3 NumPy runtime，不新增单示例 native 入口；smoke 可读取 `projects/handwriting_stage3_framework/input_data` 并生成 `QQ/pop_spk/replay_path/target_path/summary`。
- `python/examples/cerebellum_framework.py` 已迁移为可缩小规模的真实 `Simulation` 示例：通过统一 `Network` / `LearningRule` / `OuterDynamic.strict_matlab_planar_arm_2dof()` / `OuterDynamicConnection` 表达 GC/CF/PC/DCN/StrictMatlabPlanarArm2DOFOuterDynamic 闭环；Python 侧已补齐 `reset_outer_dynamic_state()` / `set_outer_dynamic_desired_state()` 控制入口并按 C++ 样本循环注入 desired state。小规模 analytic C++ baseline 与 Python `joint_state.tsv` 对齐，最大差异约 `4.9e-4`，主要来自 C++ 默认文本输出精度。
- `python/examples/zmqcommunication_client.py` 已迁移：默认通过统一 API 构造并运行本地仿真网络；`--enable-zmq` 可注册 native async ZMQ driver；内置与 C++ `ZmqTopicProtocol.h` 对齐的 `pack_spike_batch()` / `unpack_spike_batch()`，`--server` 提供可选 pyzmq 兼容测试服务。已安装 `pyzmq 27.2.0` 到 `D:\code_optimized\network_release\.conda-neuronbridge-py312\Lib\site-packages`，并完成 Python server + native async ZMQ driver 端到端验证，server 收到 51 个 SNN output batch。
- Python 3.12 下 `pytest network_release\python\tests -q` 通过，结果为 `29 passed`。
- `simulation_weight_smoke` C++ exe 构建并运行通过；Python baseline 覆盖 main/dense/boundary 三类连接权重读写、拒绝 boundary runtime 写入、save/load 恢复。
- `outer_dynamic_spike_counter_smoke` C++ exe 构建并运行通过；Python baseline 覆盖 slot count、总 spike count/weight、按 type count/weight、slot clear 和 clear all。
- `gpu_dense_subnetwork_export_real` 相关 Python baseline 覆盖 dense count/name/find、snapshot 状态字段、GPU backend flags、权重向量、输入进入 dense runtime、单 dense reset。
- `inputconv_poisson_dense_real` 相关 Python baseline 通过统一 `Simulation` + `InputConvDescription` 路径覆盖 InputConvV1 输出到 dense PoissonRate 子网络、InputConv 输出数量、DebugMonitor input/output/pending_channels 文件和非零 InputConv 输出。
- `dense_mixed_current` C++ exe 在 `build_neuronbridge_baseline_migration` 中构建并运行通过；关键输出为 `buffered_spike_count=2`、`output_spike[0] cell=2 t=4`、`output_spike[1] cell=3 t=4`、`readout_g neuron9=0.505726 neuron10=0.505726`。Python baseline 已对齐 output spike 和 dense `gexc` 非零状态。
- `dense_mixed` C++ baseline 已修复过时的 dense layer `isOutput=true` 标记；C++ exe 在 `build_neuronbridge_baseline_migration` 中构建并运行通过，关键输出为 `relay_present=1 dense1_removed=1 dense2_removed=1 readout_present=1`、`readout_g_with_relay neuron10=3.65232 neuron11=3.65232`、`stage=done`。Python 迁移版使用相同的 dense->main route 表达并运行通过。
- `dense_no_debug` C++ exe 在 `build_neuronbridge_baseline_migration` 中构建并运行通过；Python 迁移版对齐 `buffered_output_spike_count=4` 和 readout g 值。
- `ei_program_dense_subnetwork` C++ baseline 已修复共享 EI dense 构造中的过时 `isOutput=true` 标记；C++ exe 构建并运行通过。该示例在 C++ 自身重复运行时 spike totals 会变动，因此不作为固定 spike-count 单元基准。
- `ei_program_main` C++ exe 在 `build_neuronbridge_baseline_migration` 中构建并运行通过；Python 迁移版完整 2000-step 对齐 `total_e_spikes=7545`、`total_i_spikes=1861`。

仍需完成：

- `handwriting_stage3` 训练模式主仿真已对齐，仍需继续追踪 `w_mmbg1_final.tsv` 中 56 个局部饱和差异。
- `cerebellum_framework` 已完成约定缩小规模 analytic C++ baseline 对齐；后续可选择继续迁移 Pinocchio desired trajectory 生成或跑默认规模 baseline。
- ZMQ spike driver 已完成 Python `pyzmq` server 与 native async ZMQ driver 端到端 pub/sub 联调；InputConv frame communication driver 已完成 Python 绑定第一版、本地队列 native smoke 和同步 ZMQ InputConv native e2e。

### 2026-09-07 InputConv communication driver 评估

C++ 已新增能力：

- `InputConvPixelFormat` / `InputConvFrame` / `InputConvFrameRequest` / `InputConvFrameSourceBinding`：定义动态视觉帧、请求和 source-camera 绑定语义。
- `InputConvFrameDriver`：同步帧源抽象，按 `InputConvFrameRequest` 返回一帧。
- `InputConvFrameQueue` / `QueuedInputConvFrameDriver`：内存队列帧源，支持多 camera、按 time step 选择最新可用帧、消费旧帧并保留未来帧。
- `AsyncInputConvFrameDriver`：后台线程轮询帧源并推入队列，`Stop()` 在析构中 join。
- `ZMQInputConvFrameDriver`：同步 ZMQ request/response 帧源，wire magic 为 request `0x49434652`、response `0x49434650`。
- `ZMQAsyncInputConvFrameDriver`：异步 ZMQ pub/sub 帧源，wire magic 为 `0x41494346`，version 为 `1`。
- `InputConvV1` 已支持 `UsesDynamicFrameInput()`、`GetFrameRequest()`、`AcceptFrame()` 和 missing-frame policy。当前 `AcceptFrame()` 实际接受 `UInt8Gray`、`UInt8RGB`、`UInt8BGR`；`Float32*` 像素格式虽然在枚举中存在，但尚未被 `InputConvV1` 接收。
- `Simulation` 新增：
  - `AddInputConvFrameSource()`
  - `AddAsyncInputConvFrameSource()`
  - `BindInputConvFrameSource()`
  - `HasInputConvFrameSourceBinding()`
  - `LoadInputConvFrame()`
  - `AddExternalInputConvFrames()`
  - `ClearInputConvFrameQueue()`
  - `AddZMQInputConvFrameSource()`
  - `AddZMQAsyncInputConvFrameSource()`

Python 侧已完成扩展：

- 新增 `python/src/neuronbridge/frames.py`。
- 新增 `InputConvPixelFormat` Python 枚举，支持 `uint8_gray`、`uint8_rgb`、`uint8_bgr`、`float32_gray`、`float32_hwc`、`float32_chw` 的协议表达；当前 `InputConvV1` native 实际接收仍以 UInt8 三类为准。
- 新增 `InputConvFrame` Python dataclass，字段对齐 C++：`time_step`、`source_camera_index`、`width`、`height`、`channels`、`pixel_format`、`bytes`，并校验 payload 长度。
- 新增 `InputConvFrameRequest` Python dataclass。
- 新增 `Simulation.add_input_conv_frames(source_name, frames)`，对应 C++ `AddExternalInputConvFrames()`；Python facade 会在首次推帧前自动 `init()`，避免 C++ `InitSimulation()` / `ResetForNextRound()` 清空预先写入的 frame queue。
- 新增 `Simulation.clear_input_conv_frame_queue(source_name)`，对应 C++ `ClearInputConvFrameQueue()`。
- 新增 `Simulation.bind_input_conv_frame_source(index_or_name, source_name, source_camera_index=0)`，对应 C++ `BindInputConvFrameSource()`。
- 新增 `Simulation.has_input_conv_frame_source_binding(index)`。
- 新增 `Simulation.add_zmq_input_conv_frame_source(source_name, address, port, max_payload_bytes=0)`，对应同步 ZMQ 拉帧。
- 新增 `Simulation.add_zmq_async_input_conv_frame_source(source_name, subscribe_address, subscribe_port, topic, max_payload_bytes=0, max_buffered_frames_per_camera=8)`，对应异步 ZMQ 订阅。
- 新增 Python 协议 helper：`pack_input_conv_frame()`、`unpack_input_conv_frame()`、`pack_input_conv_frame_request()`、`unpack_input_conv_frame_request()`，用于测试和外部 peer 对接。
- `InputConv.v1_bar()` / `v1_grating()` / `v1_plaid()` / `v1_file()` 已显式支持 `input_frame_missing_policy` 与 `input_frame_max_lag_steps` 参数。

验证状态：

- C++ 新增 smoke target：`inputconv_dynamic_frame_smoke`、`inputconv_async_frame_smoke`、`inputconv_zmq_frame_smoke`。
- Python 测试已更新并通过：`pytest python\tests -q` 为 `32 passed`。
- 新增 `test_input_conv_dynamic_frame_protocol_helpers`，覆盖 sync/async frame payload 和 request payload round-trip。
- 新增 `test_input_conv_dynamic_frame_queue_native`，覆盖 Python `InputConvFrame` -> C++ queued frame source -> `InputConvV1` -> `Simulation.input_conv_input()`。
- 新增 `test_input_conv_zmq_frame_source_native`，覆盖 Python REP server 连续响应 C++ `ZMQInputConvFrameDriver` request，并验证帧进入 `InputConvV1` 输入缓冲。注意：C++ `sim.run(1)` 会触发不止一次 InputConv request，因此 Python server 必须用 poll loop 连续响应，不能只服务一帧。
- `python/CMakeLists.txt` 当前已通过 `NR_ZMQ_CORE_SOURCES` 链入 `ZMQInputConvFrameDriver.cpp` 和 `ZMQAsyncInputConvFrameDriver.cpp`；基础 `InputConvFrameQueue` / `AsyncInputConvFrameDriver` 由 `NR_LEGACY_CPP_SOURCES` 进入公共 core。`build_neuronbridge_wheel.ps1` 已重新构建成功，说明 `neuronbridge_core` 链接路径实际可用。
- Windows wheel 已刷新：`D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`，大小约 `9506274` bytes，并已在 `D:\code_optimized\network_release\build_neuronbridge_wheel_validate\venv_inputconv_frames` 通过 native import 和新增公开 API 可见性检查。
- 基础离线包已用新 wheel 刷新：`D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`。

### 阶段 4：迁移 C++ 示例到 Python

状态：已开始，已完成 dense/InputConv 首批示例迁移。

迁移原则：

- 通过统一 `neuronbridge` API 迁移。
- 不为单个示例写专用 native 入口。
- 每迁移一个示例，都应保留原 C++ exe 作为回归 baseline。

优先顺序：

1. `dense_debug_monitor`：已提供 Python 可视化骨架。
2. `dense_subnetwork_export`：已迁移为 `python/examples/dense_subnetwork_export.py`。
3. `inputconv_poisson_dense`：已迁移为 `python/examples/inputconv_poisson_dense.py`。
4. `dense_mixed`：已迁移为 `python/examples/dense_mixed.py`，并已修复 C++ baseline 的过时 `isOutput=true` 配置。
5. `dense_mixed_current`：已迁移为 `python/examples/dense_mixed_current.py`，并完成 output spike 关键指标对齐。
6. `dense_subnetwork_smoke`：已迁移公开 API 可表达部分为 `python/examples/dense_subnetwork_smoke.py`。
7. `simulation_weight_smoke`
8. `dense_no_debug`：已迁移为 `python/examples/dense_run_no_debug.py`，并完成 C++/Python 对齐。
9. `ei_program_dense_subnetwork`：已迁移为 `python/examples/ei_program_dense_subnetwork.py`，C++ baseline 已修复并运行通过；精确 spike totals 不作为固定基准。
10. `ei_program_main`：已迁移为 `python/examples/ei_program_main.py`，并完成 C++/Python spike totals 对齐。
11. `pattern_motion`：已迁移为 `python/examples/pattern_motion.py` 和 `python/examples/_pattern_motion_builder.py`；quick native smoke 已通过，full 32x32x8 C++ unified baseline 与 Python 方向 winner 指标已收敛。
12. `my_first_app`：已迁移为 `python/examples/my_first_app.py`，作为最小用户入门示例，C++/Python 基础输出字段已对齐。
13. `handwriting_stage1`：已迁移为 `python/examples/handwriting_stage1.py` 和 `python/examples/_handwriting_stage1.py`，C++/Python summary MSE 已对齐。
14. `handwriting_stage2`：已迁移为 `python/examples/handwriting_stage2.py` 和 `python/examples/_handwriting_stage2.py`，完整 C++/Python summary MSE 已对齐。
15. `handwriting_stage3`：已迁移为 `python/examples/handwriting_stage3.py` 和 `python/examples/_handwriting_stage3.py`；`test/test_d/train_g/train_d` NumPy runtime 可运行，`test` 与训练主输出已完成 C++ full baseline 对齐；`train_d` 的 `w_mmbg2_final.tsv` 完全一致，`w_mmbg1_final.tsv` 仍有 56 个局部饱和差异待追踪。
16. `handwriting_stage4_different_position`：已迁移为 `python/examples/handwriting_stage4_different_position.py`，C++/Python summary MSE 已对齐。
17. `handwriting_stage3_framework`：已迁移为 `python/examples/handwriting_stage3_framework.py`，复用 stage-3 shared runtime。
18. `cerebellum_framework`：已迁移为 `python/examples/cerebellum_framework.py`，小规模真实 native smoke 通过，analytic C++ baseline 已对齐。
19. `zmqcommunication_client`：已迁移为 `python/examples/zmqcommunication_client.py`，默认本地仿真 smoke 通过，Python `pyzmq` server + native async ZMQ driver 联调已通过。

验收标准：

- Python 示例可运行。
- 输出结果与 C++ 示例关键指标一致。
- 示例代码体现最终用户推荐用法。

### 阶段 5：可视化 API 增强

状态：基础能力已开始。

目标：

- 增强 DebugMonitor 结果读取。
- 增加常用图：
  - spike raster
  - voltage/state trace
  - firing rate curve
  - dense subnetwork heatmap
  - weight evolution
  - pending channel heatmap
  - InputConv output trace
  - OuterDynamic state trace
- 支持保存图片和返回 matplotlib axes。

验收标准：

- 对 GB 级输出文件不会默认爆内存。
- 用户可以按 neuron、field、time range、component 过滤。
- 可视化函数可组合，不强制 `plt.show()`。

### 阶段 6：wheel 发布

状态：已开始，首个 Python 3.12 Windows wheel 已构建并完成 `--no-deps` 干净 venv native import 验证。

目标：

- 生成 Python 3.12 Windows wheel。
- wheel 内包含：
  - `neuronbridge` Python 文件
  - `_core.cp312-win_amd64.pyd`
  - 必需 native DLL
- wheel 不要求用户安装：
  - MSVC
  - CMake
  - pybind11
  - scikit-build-core
  - Pinocchio
  - ZeroMQ
  - 项目私有 native DLL

仍可要求：

- 用户已有 Python 3.12。
- 用户机器有兼容 NVIDIA GPU driver。

验收标准：

- 在干净 Python 3.12 venv 中 `pip install neuronbridge-*.whl` 后可导入。
- `import neuronbridge` 不依赖开发机 PATH。
- DebugMonitor 读取和基础可视化可运行。

当前产物：

- `D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
- 构建脚本：`D:\code_optimized\network_release\scripts\build_neuronbridge_wheel.ps1`
- 验证脚本：`D:\code_optimized\network_release\scripts\validate_neuronbridge_wheel.ps1`
- 依赖检查脚本：`D:\code_optimized\network_release\scripts\check_neuronbridge_runtime_deps.ps1`
- 离线包脚本：`D:\code_optimized\network_release\scripts\build_neuronbridge_offline_bundle.ps1`
- wheel 内容已排除历史 `neuronbridge/Release/**` 目录；当前包含 cp312 `_core` 和同级 native DLL。
- `validate_neuronbridge_wheel.ps1 -VenvDir D:\code_optimized\network_release\build_neuronbridge_wheel_validate\venv_nodeps_final` 已通过，`backend_info()` 返回 `native_extension_loaded=True`。

### 阶段 7：完整离线包

状态：基础离线 wheelhouse 包已完成；完整便携版仍需补 Python runtime 和干净机器验证。

目标：

- 面向没有 Python 环境的用户提供完整包。
- 集成：
  - Python 3.12 runtime
  - `neuronbridge` wheel
  - native DLL
  - 可视化依赖
  - 示例脚本
  - 小型验证数据
  - 启动脚本

验收标准：

- 用户无需安装 Python、MSVC、CMake、pybind11、Pinocchio、ZeroMQ。
- 用户解压后可运行示例。
- GPU driver 作为唯一主要系统级前置条件单独说明。

当前产物：

- 离线包目录：`D:\code_optimized\network_release\release\neuronbridge_offline_py312`
- 离线包压缩文件：`D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`
- 包含 `neuronbridge` wheel、`numpy/matplotlib/pandas/pyzmq` wheelhouse、迁移示例、文档、离线安装脚本和 smoke-test 脚本。
- 依赖 wheel 已下载到：`D:\code_optimized\network_release\release\neuronbridge_offline_py312\wheels`
- 同机离线 venv 验证已通过：`D:\code_optimized\network_release\release\neuronbridge_offline_py312\.venv-offline-validate`
- 离线安装只使用本地 `wheels`，`backend_info()` 返回 `native_extension_loaded=True`。
- `run_smoke_test.ps1` 已通过，覆盖 native import 和 `examples\dense_run_no_debug.py --help`。

## 7. 构建与验证基线

推荐本地验证命令应使用 VS Developer 环境和 clean-env wrapper，避免 Codex/PowerShell 中 `PATH`/`Path` 重复导致 MSBuild `MSB6001`。

Python 3.12 配置时需要显式指定：

- `Python3_EXECUTABLE`
- `Python3_ROOT_DIR`
- `Python3_INCLUDE_DIR`
- `Python3_LIBRARY`
- `pybind11_DIR`

原因：当前工程 `CMAKE_PREFIX_PATH` 中存在 Pinocchio/Anaconda 环境，可能导致 Python executable 与 Python include/lib 混配。

## 8. 当前已知风险

- C++ `Simulation` 构造器依赖的 description 结构复杂，直接绑定会污染 Python API。
- `boost::any`、裸指针、GPU device memory 和 C++ 生命周期需要谨慎封装。
- DebugMonitor 输出可能达到 GB 级，结果 API 必须继续坚持懒读取。
- Windows wheel 的 DLL 收集需要干净环境验证，不能只依赖开发机 PATH。
- CUDA runtime DLL 可打包，但 NVIDIA GPU driver 仍是系统级依赖。
- Python 3.12 是当前发布基线，不应回退到 3.14 作为发布目标。

## 9. 后续优先级

最近一轮最重要的工作顺序：

1. 如需更严格验收，对 `pattern_motion` 的连续值指标继续做数值级对齐分析，例如 `pds_max_gexc` / `pds_fs_max_gexc` 的逐值差异；当前方向 winner baseline 已收敛。
2. `handwriting_stage3` 后续如需逐权重完全一致，应新增 C++/Numba parity kernel 复刻隐藏层矩阵累加顺序；当前主仿真输出已可作为会议报告中的训练 baseline。
3. `cerebellum_framework` 后续可继续补 Pinocchio desired trajectory 的 Python 生成/绑定，或跑默认规模 C++/Python baseline。
4. `zmqcommunication_client` 后续可继续补真实外部 peer 联调；当前 `pyzmq` optional dependency、native ZeroMQ DLL 和同机端到端验证已经纳入 wheel/离线包路径。
5. 根据真实运行结果继续补充模型专用便捷构造器和参数类型提示。
6. 将离线包升级为真正完整的便携版：加入或指向嵌入式 Python 3.12 runtime，让用户不需要预装 Python。
7. 在无 Visual Studio、CMake、pybind11、项目构建目录 PATH 的干净机器上验证离线包。

## 10. 一句话原则

`neuronbridge` 应该把现有 C++/CUDA 仿真核心稳定地桥接给 Python 用户，而不是复制、重写或绕开核心；Python 层负责表达、运行编排、结果分析和可视化，发布包负责把依赖收拢到用户不需要手工配置的程度。

## 10.1 2026-09-10 声明式模型代码生成维护架构

当前工程已完成 Dense model 的模块化重构，并通过：

```text
DenseNeuronModelList.inc / DenseLearningModelList.inc
```

把 host 注册、稳定 id 与 CUDA dispatch 局部收敛到同一宏清单。后续维护方向已经升级为“声明式模型编译”，而不是继续为每个模型手写多处 glue code。

**完整架构文档：**

```text
docs/model_codegen_maintenance_architecture.md
```

已确认的长期决策：

1. 现有主网 neuron、主网 learning rule、Dense neuron 和 Dense learning rule 是权威**默认实现**；它们保持不变。自动生成模型不继承具体旧 LIF/STDP 模型，而是继承新增通用基类：`CustomTimeDrivenNeuronModel : TimeDrivenModel`、`CustomLearningRuleModel : LearningRule`、`CustomDenseNeuronModel`、`CustomDenseLearningRuleModel`。
2. 每个 spec 必须声明通用 `Custom*` base contract version/digest 和独立 variant id。规格以确定方程为核心：神经元连续状态写成 `dx/dt=f(x,u,t)`，阈值/复位/输入写成 event jump；学习规则 trace 写成 ODE，pre/post/trigger/weight update 写成事件方程。生成器将其降级为 Equation Model/Rule IR，再生成主网 CPU/GPU custom model、主网 custom learning rule、Dense host model、CUDA static entry、variant registry、Python schema/友好构造器、reference/test/doc 产物。
3. 主网与 Dense 共享方程语义 IR，但不共享运行时 ABI：主网通用 base 保持 interface/integration/event runtime；Dense 保持 flat fields -> unified CUDA kernel -> model-id switch -> inline device update。host 侧使用 `Custom*` C++ 派生；CUDA 侧不使用继承，只组合通用 base device helper。两侧均不引入运行时 DSL 解释。
4. generated files 一律写入 CMake build directory，不提交源码树；wheel 内包含已生成的 Python 辅助产物，最终用户无需安装 generator 或 Python 构建工具。
5. 新增 `NR_ENABLE_MODEL_CODEGEN=ON`，它独立于 `NR_ENABLE_PYTHON` 且默认开启，让模型规格与 CUDA 编译错误尽早暴露。
6. generated custom variant 绝不占用已有 canonical name；使用隐藏 implementation name 和显式 selector，仅用于测试/实验。baseline 通过不意味着删除或替换手写模型。新公共模型须作为新的方程模型由用户明确批准后发布。
7. 主网 neuron 自动生成必须通过 `LegacyTimeDrivenProfile`：明确 CPU/GPU state-vector、事件时间、输入递送、integration capability、GPU queue partition 与资源生命周期。主网 learning rule 自动生成必须通过独立 `LegacyLearningRuleProfile`：明确 `LearningRule`/`SynapseState` 所有权、pre/post/trigger 时序、connection index、eligibility 和权重修改语义。两者不得与 Dense ABI 混用。
8. 首批主网 neuron 仅覆盖标准 ForwardEuler 的 `CustomLifConductanceV1` 方程模型，并以既有 LIF double 为 oracle；Input/Event/Handwriting 等特殊模型先为 `manual` backend。首批主网 learning rule 试点为 `CustomRStdpV1`，以既有 `R_STDP` 为 event-trace oracle；原始 `R_STDP`、`STDP` 与 Cerebellar rule 继续作为默认 factory 结果。
9. 迁移顺序固定为：全后端 inventory + `Custom*` base contract -> `CustomLifConductanceV1` hidden neuron shadow -> `CustomRStdpV1` hidden rule event-trace shadow -> 受控 variant registry -> 其余普通主网/Dense custom models -> Dense custom learning -> 主网特殊模型/OuterDynamic 收敛。既有 runtime 默认路径始终保留。
10. C++/CUDA/Python 三侧采用同一 Equation IR 生成；主网 neuron CPU/GPU、主网 rule event trace 与 Dense reference/CUDA 分别以明确 numerical tolerance 验证，离散事件（event timestamp、pre/post/trigger、input clear、threshold、reset、refractory）必须精确一致。
11. 维护者方程规格借鉴 Brian2 的分层思想，但实现为自有受限 DSL：在 JSON envelope 的中性字段 `equations` 中声明 `dx/dt = f : unit`、子表达式与参数；`threshold/reset/refractory/on_pre/on_post/on_trigger` 使用独立 AST block。字符串绝不执行为 Python/C++；只允许注册函数、显式参数和输入，并做单位/维度、依赖和调度校验。规格仅在受控源码仓库中由维护者维护。
12. 首版支持 Forward Euler、ODE、子表达式、参数、`constant`/`constant_over_dt`；`shared`、`unless_refractory` 后续扩展；自动 `event_driven` 仅为可证明的一维线性 trace 开放，随机 `xi` 方程延后到可复现 RNG contract 成熟后。
13. 生成器必须按“模型种类 + 后端”组织，禁止按具体模型建立一个 Python emitter。新增一个已受 DSL 支持的 neuron 或 learning rule 时，只能新增 `.nbmodel.json`、baseline fixture 和必要的发布元数据，不得修改生成器 Python、Factory、Catalog、CUDA dispatch 或 CMake 模型清单。只有新增一种此前不支持的语言特性、积分算法或运行时 primitive 时，才允许扩展通用 parser/IR/emitter。现有 `lif_runtime.py`、`rstdp_runtime.py` 是首批闭环验证产生的过渡实现，必须迁移到通用 `neuron_ir`/`learning_rule_ir` 和按后端划分的 emitter 后删除，不能作为后续模型复制模板。

维护者方程规格的完整独立规范：

```text
docs/neuronbridge_custom_equation_format.md
```

代码级实施设计（含运行时基类、生成产物、CMake 接入和性能门禁）：

```text
docs/custom_model_codegen_implementation_design.md
```

逐文件的详细修改清单（含 Catalog/Factory 注册、主网/Dense 事件接入、Python、CMake、测试和分阶段验收）：

```text
docs/custom_model_codegen_change_checklist.md
```

首个审批包的函数签名、生成 ABI、CMake target 和测试夹具蓝图：

```text
docs/custom_model_codegen_patch_blueprint.md
```

2026-09-10 已开始 Phase A，并已实现维护者 generator skeleton 与神经元 Catalog 的最小组合注册入口：`tools/model_codegen` 使用标准库校验受控 spec、输出确定性 manifest，并在无 spec 时生成空 C++ 注册产物；`NR_ENABLE_MODEL_CODEGEN` 默认开启并将生成目标纳入核心 CMake target。关闭开关时，核心改用 `GeneratedNeuronCatalogEmpty.cpp` 的同签名空注册实现。随后已实现并编译主网 CPU custom runtime 的最小 ABI：`CustomEquationDescriptor` 保存静态 state/event layout，`CustomTimeDrivenNeuronModel` 统一处理到达 spike、`CurrentSynapse` 聚合电流、状态向量初始化和输出 spike 标记；方程计算、阈值/复位和具体 `ForwardEulerMethod<ConcreteModel>` 保持由后续生成类静态绑定。受限 Equation AST 已可解析 ODE、标识符、数值、括号、一元符号和四则运算，并以确定性 C++ 表达式发射；函数调用和未批准语法在规格校验阶段拒绝。该阶段仍未实现 neuron/rule emitter、LearningRule Catalog 拆分或 Python facade；生成器在发现非空 spec 时会明确失败，避免把未实现的发射能力伪装为可用功能。

代码生成是 NeuronBridge 维护者的内部维护能力：spec、generator、CMake target 与 CI 均位于受控源码仓库中，生成模型仅随官方 wheel 发布。Python 用户可以选择官方公开的模型和参数，但没有用户 spec 导入、`model-build` extra、用户代码生成 CLI 或自建 custom wheel 的维护入口。主网 GPU/Dense 生成模型仍是维护者路线的一部分，须在完整 native core 重编译和 CUDA baseline 后随官方发布开放。

### 10.1.1 2026-09-10 首个生成 LIF 已完成运行验证

本节更新并取代上文关于“非空 spec 仍会失败、尚无 neuron emitter”的历史状态。`tools/model_codegen/lif_runtime.py` 现已将批准的 `CustomLifConductanceV1` spec 生成 concrete C++ class、descriptor、Factory registry 和 Catalog entry；CMake 显式编译生成的 `.cpp`，正式 Factory 已能构造该类型。该模型保持 LegacyCpu-only、experimental，不替换手写默认模型。

核心库 ON/OFF 构建均通过；`model_codegen_lif_baseline` 通过正式 Factory 比较两类模型：12 场景，各 2000 步，共 288000 个状态比较，最大绝对误差 0；2197 次 spike 匹配，spike 不一致数 0；CTest 1/1、生成器 unittest 9/9 通过。

兼容性细节：严格 `>` 阈值、LastSpike tick 不应期、指数电导衰减、聚合电流保持，以及既有 ForwardEuler 的 stride 双重乘法均被显式固定在 profile。电压 RHS 从 AST 发射，Euler differential count 为 1。该 profile 限于维护者首个 CPU LIF，不代表任意模型 DSL、GPU emitter 或 wheel 发布已经完成。尚需单独验证完整 Simulation 事件传播和性能吞吐。

报告与复现命令：`reports/model_codegen_lif_baseline_20260910.md`。运行 `ctest --test-dir build_model_codegen_scaffold_check -C Release -R model_codegen_lif_baseline -V` 可重现对比结果。

### 10.1.2 2026-09-10 Simulation 与积分计时补充

已在 `tests/model_codegen/lif_baseline.cpp` 增加 InputSpike → generated LIF → handwritten LIF 的完整 Simulation 回归。相同模型分组、延迟 1/3/7 三种场景共匹配 537 个输出事件，CTest 通过。4096 神经元、1000 次更新、七轮中位数：手写 22.6529 ms，生成 15.315 ms，耗时比 0.676072；只代表当前单线程积分场景。

**发布前未解决事项：** 两层同参数手写 LIF 会合并为同一模型组，替换其中一层后会分组；该情况下观察到输出序列差异。当前通过不同 t_ref 保持双方同样分组的测试已通过，但不代表跨分组透明替换通过。下一步优先调查模型组变化与同 tick 事件顺序，保留 generated 模型 experimental 状态。报告：`reports/model_codegen_simulation_20260910.md`。

### 10.1.3 2026-09-11 分组差异已定位

已将合并/拆分失败场景恢复到测试。根因是堆只按时间和优先级排序，模型分组变化导致同刻输出记录顺序不同，并非本夹具中的物理时间或状态变化。baseline 现要求输出时间单调、完整 `(time, neuron)` 向量（保留重复次数）排序后一致，另逐步比较四状态。六场景共 1116 个事件匹配，48000 个 Simulation 状态比较最大误差 0；其中 3 场景存在原始同刻顺序差异。原有 288000 状态比较和 2197 spike 对比仍通过。

无需改生产调度器；上节“待调查”的判断由此结果取代。带非交换学习规则和多队列并发的分组等价性仍需专门测试。报告：`reports/model_codegen_scheduling_20260911.md`。已开始本地候选 wheel 构建及独立 Python 3.12 venv 验收，不执行公开发布。

### 10.1.4 2026-09-11 候选 wheel 安装运行通过

`build_model_codegen_release/dist/neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl` 已构建，大小 9539312 bytes，SHA256 为 `fbe502d0ffb42063548ad918893fe6657a011550b95e8703bb8dd9b9c1bae82d`。在 `build_model_codegen_release/venv` 使用 `--no-index --no-deps` 安装，并以 `python -I tests/model_codegen/wheel_smoke.py` 确认导入来自安装包、native 加载成功、生成 LIF 与手写 LIF 的完整网络 193 个事件匹配。未公开发布；图形依赖、完整离线包及干净独立机器验证仍待后续。报告：`reports/model_codegen_wheel_20260911.md`。

### 10.1.5 2026-09-11 主网学习规则注册基础

已开始 Phase C 前置的 W1 learning rule Catalog 拆分：`LearningRuleCatalog` 的实现移入 shared `.cpp`，通过 `NR_SHARED_SOURCES` 同时纳入 codegen ON/OFF 构建；保留四个既有名称、`R-STDP` 别名及 Dense 支持列表。新增构造时附加条目的校验入口，拒绝空名称以及 canonical/alias 的全局冲突。附加目录实例不修改全局 singleton；生产 singleton 暂时仍只有手写条目，未接入 generated rule provider，更不代表 Factory 能创建新规则。

本批实际验证：ON/OFF `nr_snn_core_gpuaware_cpp` 构建通过；新增 Catalog 测试覆盖既有行为、隔离扩展及 9 类冲突，连同 LIF baseline 共 CTest 2/2 通过。LIF 原有 288000 状态比较、2197 spike，以及六组 Simulation 的 48000 状态比较、1116 事件仍一致。没有修改手写 R_STDP 事件实现、生产调度器或 Python API，本批未重建 wheel。

下一批仍按原计划完成 `CustomLearningRuleModel` / custom synapse state、`CustomRStdpV1` 的 pre/post/trigger emitter 和正式 Factory/provider 接入。必须对齐到达时刻的 trace decay、连接索引、trigger connection 排除、奖励/惩罚、权重边界、eligibility 清零及同 tick 事件顺序。不能把本批 Catalog 基础设施算作学习规则生成完成。报告：`reports/model_codegen_learning_catalog_20260911.md`。

### 10.1.6 2026-09-11 首个生成 R-STDP 权重轨迹闭环

Phase C 的首个主网 CPU 学习规则试点已落地。维护者 spec `model_specs/custom_rstdp_v1.nbmodel.json` 定义两条指数 trace ODE、保持型 eligibility、严格顺序的 pre/post/trigger 事件块、参数及 trigger contract；`rstdp_runtime.py` 校验批准 profile，并从事件表达式 AST 发射 `CustomRStdpV1`、专用 `SynapseState`、Factory registry 和 experimental Catalog entry。生成规则继承通用 `CustomLearningRuleModel`，不继承手写 `R_STDP`/`STDP`，手写 `R_STDP` 与 `R-STDP` 别名仍走原 Factory 默认路径。

为保证生成开关在所有调用点一致，`LearningRuleModelFactory` 的实现已由 header 移入 `.cpp`；codegen ON 时使用生成 registry，OFF 时使用空 generated Catalog provider。ON/OFF `nr_snn_core_gpuaware_cpp` 均通过编译。生成器 unittest 15/15、codegen CTest 3/3 通过。

直接事件轨迹覆盖 dt 0.1/1.0、eligibility clear 开/关、普通/放大奖励、同 tick pre/post/trigger 的 6 种排列、两个目标神经元、跳过索引、trigger 连接排除、百万 tick 衰减间隔以及上下限钳位：48 场景、13152 个事件、79200 行逐连接快照、361800 次数值比较，weight/pre/post/eligibility/LastUpdate 最大绝对误差均为 0；下界样本 6895、上界样本 18344。完整 Simulation 另覆盖 clear 开/关与 delay 1/3/7，共 6 场景、45000 个逐步样本和 467 个输出 spike，全部一致。七轮交替事件微基准中位数为手写 4.5548 ms、生成 4.1075 ms、比值 0.901796，只作为当前单线程夹具证据。

轨迹、独立完整性汇总和图位于 `build_model_codegen_scaffold_check/rstdp_trace/`，报告为 `reports/model_codegen_rstdp_trajectory_20260911.md`。这表示首个 Legacy CPU R-STDP profile 的端到端闭环完成；通用任意学习规则 DSL、多生成规则、GPU/Dense rule emitter、Python 自动 schema 和 wheel 验收仍未完成。

### 10.1.7 2026-09-11 多规则生成与 Dense GPU shadow 验证

同一 `legacy_rstdp_event_trace_v1` emitter 已从单 spec 改为遍历批准规则集合，一次生成独立 CPU class/source、聚合 include、Factory registry、Catalog provider、Dense host wrapper 和 Dense static model list。新增 `CustomRStdpPersistentV1`（variant 20002，默认不清除 eligibility），与 `CustomRStdpV1`（variant 20001，默认清除）同批生成；两者拥有稳定且不受文件排序影响的 Dense model id 5 和 4。manifest 现含 LIF 与两个 R-STDP spec，共 3 项。生成器仍显式限制批准的两个 R-STDP identity/profile，不是任意规则输出。

CPU 多规则回归把两个生成类分别与手写 `R_STDP` 对齐：96 场景、26304 事件、158400 行快照、723600 次数值比较，weight/pre/post/eligibility/LastUpdate 最大误差 0；12 个 Simulation 场景含 90000 个逐步样本与 934 个 spike，全部一致。独立 CSV 审计通过。生成器 unittest 16/16、codegen CTest 3/3 通过；LIF baseline 未回归。事件微基准本轮手写 4.3547 ms、生成 4.1519 ms、比值 0.95343，只代表当前夹具。

Dense GPU 路径已接入 generated model list：`DenseCustomRStdpV1` 与 `DenseCustomRStdpPersistentV1` 使用独立 host model/field span/model id，并通过静态 CUDA switch 复用既有 `ApplyRStdpPre/Post/TriggerDeviceEntry`。独立 `build_model_codegen_gpu_rules` 使用 MSVC 19.41、CUDA 12.6 编译成功，NVCC 实际编译含新增 case 的 `GpuPropagationRuntime.cu`；真实 `dense_subnetwork_smoke` 在 GPU 上通过，内建 Dense `R_STDP` 与两个生成规则的逐 tick 权重采样完全一致，输出 `generated_dense_rstdp_models=2 model_ids=4,5 gpu_weight_trajectory=match`。初次测试失败源于沿用 Additive 场景把 postsynaptic threshold 设为 1000，修正为 pair-trace 场景显式产生 post spike 并延迟 trigger 后通过，并非生成 dispatch 数值错误。

`NR_ENABLE_MODEL_CODEGEN=OFF` 下 `nr_snn_core_gpuaware_cpp` 与 `nr_dense_runtime_gpuaware` 均重新编译通过。当前 GPU 能力准确表述为“生成式 host/registry/field/static-dispatch + 复用已验证 R-STDP CUDA device kernel”；尚未实现事件 AST 直接发射 CUDA 数学函数，也未覆盖异构第二种学习方程、GPU 全字段轨迹导出、Python schema 或 wheel。报告：`reports/model_codegen_multirule_gpu_20260911.md`。

### 10.1.8 2026-09-11 受控 AST-to-CUDA 与发布验收

`legacy_rstdp_event_trace_v1` 已新增受控 CUDA lowering。生成器对批准的方程、积分方法、状态布局和事件目标完成统一校验，再把 `on_pre`、`on_post`、`on_trigger` 的表达式 AST 分别发射为每个模型独立的 `__device__` entry；generated Dense model list 已改为引用 `ApplyDenseCustomRStdp*`，不再引用手写 `ApplyRStdp*DeviceEntry`。指数 ODE 仍由批准 profile 降低为 `DecayPairTrace`，任意 ODE、分支、函数调用和动态 CUDA 编译仍不在支持范围。

新增独立 `model_codegen_rstdp_gpu_trajectory` fixture：真实 CUDA 运行时先导出实际 pre/post/trigger 事件流和逐 tick 权重，再将完全相同的事件顺序重放到同名生成 CPU 规则。两个规则共 22 个 CPU/GPU 权重样本，最大绝对误差 `1.90734863e-06`，低于明确阈值 `5e-6`；CSV、JSON 审计和轨迹图位于 `build_model_codegen_scaffold_check/rstdp_gpu_trace/`。codegen CTest 已扩展为 4/4，通过原 Catalog、LIF、CPU R-STDP 与新增 CPU/GPU 测试；generator unittest 17/17 通过。

Python 3.12 wheel 已重新构建为 `python/dist/neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`，SHA-256 为 `DBAF2C10487C40BA845E2BB25EB8D05C49242457ACAFAA06690F8C45F6BCD480`。`validate_neuronbridge_wheel.ps1` 默认增加 installed-wheel CUDA smoke，在隔离 venv 中仅使用公开 API 构建两个生成规则的 Dense 网络，确认 native extension、CUDA build、GPU backend、CARLsim-like path、生成规则注册和 Dense firing 均有效；报告位于 `build_neuronbridge_wheel_cuda_validate/cuda_runtime_report.json`。wheel 已包含 `cudart64_12.dll`、Pinocchio、Boost、ZeroMQ、OpenMP 等运行 DLL；用户仍需兼容的 NVIDIA driver 和 Python 3.12，完整嵌入式 Python 离线包及无开发环境独立机器验收仍需后续执行。

本阶段未增加 LearningRule Monitor。wheel 公共 API smoke 的黑盒拓扑未观察到 reinforcement 权重变化，因此权重数学验收以独立底层 GPU fixture 为准；公开 `Simulation` 的 trigger relay 示例调度仍应单独对齐后再升级为 wheel 端权重断言。专项报告：`reports/model_codegen_ast_cuda_wheel_20260911.md`。

### 10.1.9 2026-09-11 Dense 生成神经元与异构 Pair-STDP 闭环

`CustomLifConductanceV1` 已增加真正的 Dense CUDA 派生实现。`lif_runtime.py` 从同一受控电压 ODE AST 生成 Dense host metadata、field slots、输入 channel、spike effect、初值/派生量/reset 逻辑和独立 `UpdateDenseCustomLifConductanceV1DeviceEntry`；稳定 Dense factory id 为 10001。CUDA 热路径只执行预编译数组访问与数学表达式，不包含 parser、AST、Python、虚函数或逐神经元动态分配。真实 GPU baseline 覆盖 excitatory、inhibitory、current、threshold、reset 和 refractory，160 个逐步样本、20 个 spike，与手写 Dense LIF 最大绝对误差为 0。

新增第二种异构学习规则 `CustomPairStdpV1`（variant 20003，Dense model id 6）。它使用二状态布局 `pre_trace/post_trace`，没有 eligibility 和 trigger；`on_pre` 立即执行 LTD，`on_post` 立即执行 LTP。该布局和事件方程不同于三状态、trigger 驱动的 `CustomRStdpV1`/`CustomRStdpPersistentV1`。生成器同时输出独立 Legacy CPU class/state、Dense host wrapper、字段映射和 AST-to-CUDA pre/post entry，并拒绝 Pair profile 中的 eligibility、trigger 或第三状态。GPU 真实发放事件重放测试共 17 个权重样本、8 个变化步，生成 CPU 对手写 STDP、生成 CPU 对生成 CUDA 的最大误差均为 0。

异构规则首次暴露并修复了一个 Dense registry 顺序缺陷：生成文件按名称排列为 model id 6/5/4，而旧 `BuildRegisteredFieldTable()` 按注册顺序构建 span，kernel 却按 model id 索引。R-STDP 规则字段布局相同，过去会掩盖该错误；Pair-STDP 的不同布局使其可观测。现实现按 `FactoryModelId()` 排序并强制 id 从 0 连续，保证 device span 与 dispatch id 一致。修复后原有 R-STDP 与新增 Pair-STDP 测试均通过。

当前 codegen suite 为 generator unittest 20/20、CMake/CTest 6/6；`NR_ENABLE_MODEL_CODEGEN=OFF` 的 `nr_snn_core_gpuaware_cpp` 和 `nr_dense_runtime_gpuaware` 继续构建通过。Python 3.12 wheel 已刷新，SHA-256 为 `3045B7C6A0D82CB2405E57160D58AA0DCDF75CFD3F98C0F7E9CE6F79D62E9783`。隔离 wheel 验收仅使用公开 API 运行生成 Dense LIF 与三种生成规则：`CustomRStdpV1` 权重 `8.0 -> 13.3458414`，`CustomRStdpPersistentV1` 为 `8.0 -> 15.5036612`，`CustomPairStdpV1` 为 `8.0 -> 15.1958809`，CUDA backend 与 CARLsim-like GPU path 均为 active。

权重能力边界已明确：不新增 Dense 实时权重批量访问器，也不把 `dense_subnetwork_weights()` 改造成隐式 GPU 同步接口。正式仿真结果、baseline 留档和用户工作流统一使用 `Simulation.save_weights()` 输出权重文件；已有 `Simulation.get_connection_weight()` 只保留用于单连接控制和测试诊断。这样可以避免为了可视化或审计而在运行热路径中引入频繁的 device-to-host 批量同步。需要权重演化曲线时，应使用按检查点保存的权重文件或专用离线轨迹文件，而不是扩展实时批量 API。

本阶段表示 W4 Dense custom LIF 和“第二种异构状态布局/事件方程”的 W5 试点均已闭环，但仍不是任意方程 CUDA 编译器。后续优先项为：扩大受控 neuron/rule profile 集合；补吞吐量 >= 手写基线 98% 的正式性能门禁；完成 Python generated facade/schema；在无 Visual Studio/CMake/CUDA toolkit 的独立机器上做 wheel 与离线包验收。专项报告：`reports/model_codegen_dense_lif_pair_stdp_20260911.md`。

## 11. 2026-09-07 InputConv Communication Driver 闭环

本轮目标：把 InputConv communication driver 的 C++/Python 双侧验证打牢，覆盖本地队列、同步 ZMQ request/reply、异步 ZMQ publish/subscribe 三类动态视觉帧入口。

已完成：

- Python facade 已补生命周期保护：`Simulation.add_zmq_input_conv_frame_source()` 和 `Simulation.add_zmq_async_input_conv_frame_source()` 会在注册通信源前确保 native simulation 已初始化，避免首次 `run()` 触发的 `InitSimulation()` / `ResetForNextRound()` 清空或重置早到的动态帧状态；该策略与 `add_input_conv_frames()` 保持一致。
- Python 协议测试已新增异步 ZMQ e2e：`test_input_conv_zmq_async_frame_source_native` 从 Python `PUB` 发布 `topic + async header + payload` 三段消息，C++ `ZMQAsyncInputConvFrameDriver` 后台接收并入队，`Simulation.run()` 消费后可通过 `input_conv_input(0)` 读回 `[88.0, 88.0, 88.0, 88.0]`。
- Python 全量测试通过：`D:\code_optimized\network_release\.conda-neuronbridge-py312\python.exe -m pytest python/tests/test_results.py -q`，结果 `33 passed in 2.77s`。
- C++ smoke 已形成三条 runtime 闭环：
  - `inputconv_dynamic_frame_smoke`：本地 queued frame source，验证两个 InputConv 分别消费 camera 0/1 的动态帧。
  - `inputconv_async_frame_smoke`：自定义 async driver 后台入队，验证 Simulation 路径消费动态帧，并验证 reset 后队列清空。
  - `inputconv_zmq_frame_smoke`：同步 ZMQ `REQ/REP` 与异步 ZMQ `PUB/SUB` 同时覆盖，验证 frame 进入 InputConv input buffer 且 output 与内建 bar reference 对齐。
- C++ 三个 smoke 最终运行通过：
  - `D:\code_optimized\network_release\build_neuronbridge_wheel\projects\inputconv_dynamic_frame_smoke\Release\inputconv_dynamic_frame_smoke.exe`
  - `D:\code_optimized\network_release\build_neuronbridge_wheel\projects\inputconv_async_frame_smoke\Release\inputconv_async_frame_smoke.exe`
  - `D:\code_optimized\network_release\build_neuronbridge_wheel\projects\inputconv_zmq_frame_smoke\Release\inputconv_zmq_frame_smoke.exe`
- 修复了 C++ `inputconv_dynamic_frame_smoke` baseline 的生命周期顺序：现在先 `InitSimulation()`，再 `AddExternalInputConvFrames()` / `BindInputConvFrameSource()`，避免初始化阶段清掉预先塞入的 frame queue。
- Windows wheel 已刷新：`D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`，大小 `9506279` bytes；`validate_neuronbridge_wheel.ps1 -VenvDir D:\code_optimized\network_release\build_neuronbridge_wheel_validate\venv_inputconv_comm` 已通过，`backend_info()` 返回 `native_extension_loaded=True`。
- 离线包内的 `neuronbridge` wheel 已替换为最新 wheel，并重新生成压缩包：`D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`，大小约 `134433932` bytes。
- 离线 wheelhouse 安装验证已通过：`install_neuronbridge_offline.ps1 -VenvDir D:\code_optimized\network_release\release\neuronbridge_offline_py312\.venv-inputconv-comm` 使用 `--no-index --find-links` 安装 `neuronbridge[communication]`，并成功 `import neuronbridge`，`backend_info()` 返回 `native_extension_loaded=True`。

构建环境记录：

- 直接 MSBuild 在部分 target 上仍可能触发 `MSB6001` / `PATH` 与 `Path` 重复问题；本轮新增本地辅助脚本 `D:\code_optimized\network_release\tools\run_clean_env_passthrough.ps1`，用于清理重复环境变量且不重定向 MSBuild 输出，避免原 clean-env wrapper 在大输出场景下卡住。
- `build_inputconv_comm_probe2` 的独立重新 configure 仍会长时间无输出；当前可靠路径是复用 `build_neuronbridge_wheel` 构建树，通过 VS Developer 环境 + VS bundled CMake 单目标构建 smoke。

下一步建议：

1. 把 `run_clean_env_passthrough.ps1` 固化为正式构建工具，或修复既有 clean-env wrapper 的 stdout/stderr 并发读取，避免 MSBuild 大日志时管道阻塞。
2. 为 InputConv communication driver 增加更严格的协议负例测试：错误 magic、错误 topic、payload 过大、尺寸/通道不匹配、max lag 过期。
3. 把异步动态视觉输入接入实际示例迁移路径，例如 camera/event frame source 到 `pattern_motion` 或后续视觉输入 demo。
4. 后续完整便携发布包仍需补嵌入式 Python 3.12 runtime，并在无 MSVC/CMake/开发 PATH 的干净机器上验证。

## 12. 2026-09-07 InputConv 协议负例与示例迁移

本轮目标：在已跑通 communication driver 正向闭环后，补充更严格的负例测试，并把异步动态视觉输入接入真实示例迁移路径。

已完成：

- `python/tests/test_results.py` 新增/扩展 InputConv frame protocol 负例：
  - 同步 request magic 错误。
  - 同步 response magic 错误。
  - 异步 frame magic/version header 错误。
  - payload 截断导致 incomplete payload。
- `python/tests/test_results.py` 新增 native 负例：
  - 本地 queued frame 尺寸与 InputConvV1 模型不匹配时，`input_frame_missing_policy="zero"` 生效，InputConv input buffer 为零。
  - max lag 过期：仿真先前进，再绑定过旧 frame，下一步无法消费旧帧并按 zero policy 处理。
  - 同步 ZMQ response magic 错误时，C++ `ZMQInputConvFrameDriver` 拒绝帧，Simulation 不崩溃且 InputConv input buffer 为零。
  - 异步 ZMQ wrong topic 被 SUB filter 忽略，随后 good topic 可以恢复并被 C++ `ZMQAsyncInputConvFrameDriver` 消费。
- `python/examples/_pattern_motion_builder.py` 增加动态视觉帧生成能力：
  - `make_stimulus_frames()`
  - `make_input_conv_frames()`
  - `build_pattern_motion_network(..., dynamic_input=True, stimulus_mode=...)`
- `python/examples/pattern_motion.py` 已接入实际动态输入示例路径：
  - `--input-source file`：保留原 file stimulus 路径。
  - `--input-source queue`：使用 `Simulation.add_input_conv_frames()` 通过统一 API 注入动态视觉帧。
  - `--input-source zmq-async`：Python `PUB` 发布 async InputConv frame，C++ `ZMQAsyncInputConvFrameDriver` 订阅、入队并由 `Simulation.run()` 消费。
  - `--frame-topic` 可指定 async ZMQ topic。
- `test_pattern_motion_dynamic_input_sources` 覆盖 `pattern_motion` 小规模 queue 和 zmq-async 两条实际示例路径，并验证两个动态输入源关键 winner 指标一致。

验证结果：

- 目标测试子集通过：`7 passed, 30 deselected in 2.71s`。
- Python 全量测试通过：`37 passed in 4.42s`。
- 示例 CLI smoke 通过：
  - `pattern_motion.py grating 0 --steps 2 --input-source queue`
  - `pattern_motion.py grating 0 --steps 2 --input-source zmq-async --frame-topic cli_pattern_motion_inputconv`
  两者输出均为 `v1_rate_winner=2`、`v1_current_update_mean_winner=0`、`dense_subnetwork_count=1`。
- Windows wheel 已刷新并通过干净 venv 验证：`D:\code_optimized\network_release\build_neuronbridge_wheel_validate\venv_inputconv_negative_examples`。
- 离线包已用 `build_neuronbridge_offline_bundle.ps1 -SkipDownload` 重新生成，排除 `.venv*` 验证目录；当前 zip 为 `D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`，大小约 `52037528` bytes。
- 离线 wheelhouse 安装验证已通过：`D:\code_optimized\network_release\release\neuronbridge_offline_py312\.venv-inputconv-negative-examples` 使用本地 wheels 安装 `neuronbridge[communication]`，并成功加载 native extension。

后续建议：

1. 把 async ZMQ frame source 的发布/订阅握手封装为 reusable helper，减少示例里对 `time.sleep()` 的依赖。
2. 如需更严格 CI，可在 C++ 侧也补 wrong magic、payload too large、wrong topic、max lag 过期 smoke。
3. 后续可接真实相机/event camera 数据源，把 `pattern_motion --input-source zmq-async` 的 Python publisher 替换为外部 producer。

## 13. 2026-09-08 InputConv Communication Python 正式封装

本轮目标：按照“REQ/REP 保留、PUB/SUB 保留，并在 Python 侧正式封装”的方案落地，减少示例和测试中的裸 ZMQ 代码。

已完成：

- 新增 `python/src/neuronbridge/communication.py`：
  - `InputConvFramePublisher`：封装异步 `PUB/SUB` 动态视觉帧发布。
  - `InputConvFrameServer`：封装同步 `REQ/REP` 按需供帧服务。
- `InputConvFramePublisher` 能力：
  - `port=0` 自动绑定随机端口。
  - `topic + async header + payload` 三段 multipart 自动打包发送。
  - `publish_frame()` / `publish_frames()`。
  - `wait_ready()` 封装 PUB/SUB warmup 等待。
  - context manager 自动关闭 socket/context。
  - `summary()` / `published_frames` / `published_messages` / `last_time_step` / `last_source_camera_index`。
- `InputConvFrameServer` 能力：
  - 后台线程运行 REP server。
  - 自动解析 C++ `InputConvFrameRequest`。
  - 调用 Python `frame_provider(request)` 返回 `InputConvFrame`。
  - provider 返回 `None` 或抛错时返回零帧，保持 REQ/REP socket 状态机不死锁。
  - `summary()` 记录 request/frame/error 统计。
- `python/src/neuronbridge/__init__.py` 已导出：
  - `InputConvFramePublisher`
  - `InputConvFrameServer`
- `python/examples/pattern_motion.py` 已去掉 `zmq.Context/socket/send` 裸代码，`--input-source zmq-async` 改用 `nb.InputConvFramePublisher`；summary/CLI 输出新增：
  - `frame_topic`
  - `published_frames`
  - `published_messages`
- `python/tests/test_results.py` 已新增 `test_input_conv_frame_publisher_summary`，并把同步正向测试切到 `InputConvFrameServer`，异步正向和 wrong-topic 恢复测试切到 `InputConvFramePublisher`。

验证结果：

- 目标测试子集通过：`5 passed, 33 deselected in 4.74s`。
- Python 全量测试通过：`38 passed in 8.23s`。
- CLI 示例 smoke 通过：
  - `pattern_motion.py grating 0 --steps 2 --input-source queue` 输出 `published_frames=4`、`published_messages=4`。
  - `pattern_motion.py grating 0 --steps 2 --input-source zmq-async --frame-topic publisher_api_cli_smoke` 输出 `published_frames=4`、`published_messages=48`。
  两者关键 winner 一致：`v1_rate_winner=2`、`v1_current_update_mean_winner=0`、`dense_subnetwork_count=1`。
- Windows wheel 已刷新：`D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`，大小约 `9508966` bytes。
- 干净 wheel venv 验证通过：`D:\code_optimized\network_release\build_neuronbridge_wheel_validate\venv_inputconv_comm_api`。
- 离线包已用 `build_neuronbridge_offline_bundle.ps1 -SkipDownload` 重新生成，并通过本地 wheelhouse 安装验证：`D:\code_optimized\network_release\release\neuronbridge_offline_py312\.venv-inputconv-comm-api`。
- 离线 venv 中已确认新 API 可见：`InputConvFramePublisher InputConvFrameServer`。

当前设计判断：

- REQ/REP 不需要改协议，继续作为 Simulation clock 驱动的同步拉帧模式。
- PUB/SUB 不需要改协议，继续作为外部视觉源持续推帧模式。
- 对用户暴露的是 `neuronbridge` Python API，而不是裸 ZMQ socket；底层协议保持与 C++ driver 一致，维护成本较低。

后续建议：

1. 给 `InputConvFramePublisher.wait_ready()` 增加更可靠的 ready/ack 可选机制；当前第一版仍是 warmup sleep。
2. C++ 侧补 frame source status/stats 绑定后，可让 Python 查询真实 received/dropped/last_error，而不只看 publisher 侧统计。
3. 将真实图片序列、视频或相机 producer 改造成 `InputConvFramePublisher` 的上层 helper。

## 14. 2026-09-08 Async InputConv Driver 运行模式判断

本轮问题：确认 async ZMQ InputConv driver 是否只能在看门狗/同步线程模式下运行，还是普通 `Simulation.run()` / `RunSimulationStep()` 也可以运行。

源码判断：

- `AsyncInputConvFrameDriver` 自己维护独立 worker 线程：
  - `Start()` 通过 `std::thread(&AsyncInputConvFrameDriver::WorkerLoop, this)` 启动后台线程。
  - `WorkerLoop()` 负责调用具体 driver 的 `PollFrame()`，成功后 `queue_.PushFrame(frame)` 入队。
- 启动路径来自 frame source 注册，而不是 watchdog：
  - `Simulation::AddZMQAsyncInputConvFrameSource(...)`
  - 调用 `Simulation::AddAsyncInputConvFrameSource(..., true)`
  - 内部执行 `driver->Start()`
- Simulation 正常运行路径消费队列：
  - `Simulation.run()` / pybind `NativeSimulation::run()` 调用 `simulation_->RunSimulationStep(steps)`。
  - `InputConvV1::Update()` 中如果已绑定动态 frame source，会调用 `simulation->LoadInputConvFrame(...)`。
  - `LoadInputConvFrame()` 从对应 `InputConvFrameDriver` 读取 frame；对于 async driver，即从内部队列取可用帧。
- watchdog / `SyncThread` 属于实时 pacing 和同步事件机制：
  - `Simulation::RunSimulationSlot(...)` / `RealTimeRestriction` / `Watchdog()` 负责实时节拍限制。
  - 这套机制和 `AsyncInputConvFrameDriver::WorkerLoop()` 不是同一条启动路径。

结论：

- async ZMQ InputConv driver 不依赖看门狗同步线程。
- 普通模式下，只要调用 `sim.add_zmq_async_input_conv_frame_source(...)`，C++ 侧就会启动 async driver worker；随后普通 `sim.run(...)` 即可消费已入队 frame。
- 已有验证覆盖普通运行模式：
  - `test_input_conv_zmq_async_frame_source_native`
  - `test_pattern_motion_dynamic_input_sources`
  - `pattern_motion.py --input-source zmq-async`

需要避免的误解：

- async driver 的收帧线程是 communication worker，不是 Simulation watchdog。
- `PUB/SUB` 早期消息可能丢失来自 ZeroMQ 订阅握手语义，不是 watchdog 未开启导致。
- 当前 `InputConvFramePublisher.wait_ready()` 仍是 warmup sleep；后续应通过 source status/stats 或 ack 机制提高可靠性。

后续代码目标：

1. C++ `InputConvFrameDriver` / `AsyncInputConvFrameDriver` 增加 source status/stats，让 Python 能查询 `received_frames`、`consumed_frames`、`dropped_frames`、`queued_frames`、`last_error`。
2. Python `Simulation.input_conv_frame_source_status(...)` 绑定上述 C++ 状态。
3. 用 C++ 侧真实接收状态增强 `InputConvFramePublisher.wait_ready()`，减少固定 sleep。
4. ROS 耦合时可沿用该设计：ROS image/spike producer 异步发布，Simulation 普通 `run()` 消费，不要求开启实时 watchdog；只有需要 wall-clock pacing 时才启用实时机制。

## 15. 2026-09-08 InputConv Frame Source Status/Stats 实现

本轮目标：按照上一阶段计划，先把 InputConv communication driver 的 C++/Python 双侧可观测性打牢，让 queue、sync ZMQ、async ZMQ 都能通过统一 Simulation API 查询源状态。

已完成：

- C++ 新增通用 `InputConvFrameSourceStats`：
  - `requested_frames`
  - `received_frames`
  - `consumed_frames`
  - `failed_requests`
  - `dropped_frames`
  - `queued_frames`
  - `last_received_time_step`
  - `last_consumed_time_step`
  - `last_source_camera_index`
  - `running`
  - `last_error`
- `InputConvFrameDriver` 新增虚函数 `GetStats()`，保持基类默认零状态，避免影响已有 driver。
- `InputConvFrameQueue` / `QueuedInputConvFrameDriver`：
  - `PushFrame()` / `PushFrames()` 返回被 buffer limit 裁剪掉的 frame 数。
  - `Size()` 返回当前 queued frame 数。
  - queued driver 统计 received/consumed/dropped/queued 与最后时间步。
- `AsyncInputConvFrameDriver`：
  - 统计后台 worker 收到的 frame、被仿真消费的 frame、丢弃 frame、失败 poll/request、当前 queued frame。
  - `running` 显示后台 worker 是否正在运行。
  - `last_error` 保留最近一次非空协议/IO 错误。
- `ZMQInputConvFrameDriver`：
  - 统计同步 REQ/REP 请求数、成功收帧数、失败请求数、最后成功 frame 时间步和相机索引。
  - 修复坏 response magic 负例：收到非法 magic 后会吞掉同一 multipart response 的 payload，避免 REQ socket 下次发送进入 `Operation cannot be accomplished in current state`。
  - 零长度 payload 也按成功响应更新 received stats。
- `Simulation` 新增 C++ API：
  - `GetInputConvFrameSourceStats(const std::string& source_name, ...)`
  - `GetInputConvFrameSourceStats(int source_index, ...)`
- pybind 新增：
  - native `Simulation.input_conv_frame_source_status(source_name)`
  - native `Simulation.input_conv_frame_source_status(source_index)`
- Python 高层新增：
  - `Simulation.input_conv_frame_source_status(source_name_or_index: str | int) -> dict`

验证覆盖：

- queued frame source：验证注入后 `received_frames/queued_frames`，运行后 `consumed_frames/queued_frames/last_consumed_time_step`。
- sync ZMQ frame source：验证 `requested_frames/received_frames/failed_requests/last_received_time_step/last_error`。
- async ZMQ frame source：验证 worker `running`、发布后 `received_frames`、运行后 `consumed_frames`。
- sync ZMQ 坏 response magic：验证 `failed_requests` 增加、`received_frames == 0`、`last_error` 含 magic，并确认 REQ/REP 状态机不被残留 payload 破坏。
- async ZMQ wrong topic：验证错 topic 不入队，good topic 后恢复入队和消费。

验证结果：

- 目标测试子集通过：`5 passed, 33 deselected in 1.87s`。
- Python 全量测试通过：`38 passed in 4.66s`。
- Windows wheel 已刷新：
  - `D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
  - 大小约 `9512321` bytes。
- Python source-tree native import smoke 通过，确认 `Simulation.input_conv_frame_source_status` 可见。
- 离线包已用最新 wheel 刷新：
  - `D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`
  - 大小约 `52049747` bytes。

当前设计判断：

- 这次修改把 dynamic visual input 变成了 Simulation 的可观测通用能力，而不是示例专属能力。
- REQ/REP 和 PUB/SUB 协议不需要为状态查询改线；状态来自 C++ driver 内部真实统计。
- `last_error` 是最近一次错误，不是历史错误日志；需要长期错误追踪时，应在 Python helper 或后续 C++ ring buffer 中扩展。

下一步建议：

1. 用 `input_conv_frame_source_status()` 增强 Python `InputConvFramePublisher.wait_ready()` 或新增 `wait_until_received()` helper，降低 PUB/SUB warmup/repeat 依赖。
2. 补同步 `InputConvFrameClient`/server runner 的更正式封装，让 REQ/REP 在 Python 侧也有和 async publisher 同级的用户 API。
3. 设计 ROS image / spike adapter 时，直接复用当前 frame source stats 做健康检查、超时判断和数据链路诊断。

## 16. 2026-09-08 Async 状态等待与 Sync Client/Server 适配封装

本轮目标：落实两个下一步任务：

1. 基于 `Simulation.input_conv_frame_source_status()` 增强 Python async publisher 的 ready/wait 机制。
2. 补同步 ZMQ 的正式 Python client/server 适配封装。

已完成：

- `InputConvFramePublisher` 新增 `wait_for_native_receiver(...)`：
  - 轮询 `simulation.input_conv_frame_source_status(source_name_or_index)`。
  - 可按 `previous_received_frames` 等待 received count 增量。
  - 可按 `min_received_frames` 等待绝对 received count。
  - 超时会抛出 `TimeoutError`，并携带最后一次 native source status，便于诊断。
- `InputConvFramePublisher` 新增 `publish_frame_until_received(...)`：
  - 发布 frame 后轮询 native async source status。
  - 如果 PUB/SUB 订阅握手导致早期消息丢失，会在 `max_attempts` 内重发。
  - 目标是把示例和测试从固定 `sleep + 大 repeat` 迁移到“以 C++ 真实接收状态为准”的等待。
- 新增 `InputConvFrameClient`：
  - 对应同步 ZMQ `REQ` client。
  - 使用 `InputConvFrameRequest` 发起请求，接收 `InputConvFrame` 响应。
  - 提供 `summary()`，统计 requested/received/failed、最后请求/响应时间步和错误。
  - 用于 Python 到 Python、Python 到外部 server、协议互操作测试。
- `InputConvFrameServer` 保持作为同步 ZMQ `REP` server：
  - 继续服务 native C++ sync driver 的按需拉帧。
  - 和新增 client 构成完整 Python 侧 REQ/REP 协议适配层。
- `neuronbridge.__init__` 新增导出：
  - `InputConvFrameClient`
- `Simulation` 高层新增 `add_input_conv_frame_server_source(...)`：
  - 接收 Python `InputConvFrameServer`。
  - 自动调用 `add_zmq_input_conv_frame_source(...)`。
  - 可选 `bind_to` 直接绑定到 InputConv index/name。
  - 让用户不必手动拼 address/port/source/bind 四步。

测试更新：

- 新增 `test_input_conv_frame_client_server_roundtrip`：
  - Python `InputConvFrameClient` 请求 Python `InputConvFrameServer`。
  - 验证 request/response 字段和双方统计。
- `test_input_conv_zmq_frame_source_native`：
  - 改用 `Simulation.add_input_conv_frame_server_source(...)` 注册 Python server。
- `test_input_conv_zmq_async_frame_source_native`：
  - 改用 `InputConvFramePublisher.publish_frame_until_received(...)`。
  - 不再依赖固定 `wait_ready(0.2) + repeat=40` 作为主要同步机制。

验证结果：

- 新封装目标测试子集通过：`5 passed, 34 deselected in 1.37s`。
- Python 全量测试通过：`39 passed in 4.12s`。
- Windows wheel 已刷新：
  - `D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
  - 大小约 `9513325` bytes。
- 离线包已刷新：
  - `D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`
  - 大小约 `52050961` bytes。
- source-tree import smoke 已确认：
  - `InputConvFrameClient`
  - `InputConvFramePublisher`
  - `InputConvFrameServer`
  - `Simulation.add_input_conv_frame_server_source`

当前设计判断：

- async PUB/SUB 仍保留 ZeroMQ 原生语义，不引入协议级 ack；Python 层通过 native received stats 形成工程上可验证的 ready/wait。
- sync REQ/REP 已有 Python client + server 两端封装，适合未来做 ROS bridge 或其他外部系统的 clock-driven pull 模式互操作测试。
- 对用户来说，推荐优先使用 Python helper 和 Simulation helper，不直接手写 ZMQ multipart。

下一步建议：

1. 把 `publish_frame_until_received(...)` 接入 `pattern_motion.py --input-source zmq-async`，进一步减少示例里的 warmup/repeat 参数。
2. 为 ROS image adapter 草拟第一版 API：ROS topic -> `InputConvFramePublisher` / `InputConvFrameServer`。
3. 为 spike interaction 设计类似的 Python helper，统一 spike batch/stream 与 frame source 的诊断字段。

## 17. 2026-09-08 pattern_motion 接入 Async Native Status Wait

本轮目标：把 `InputConvFramePublisher.publish_frame_until_received(...)` 接入实际示例迁移路径，优先处理 `pattern_motion.py --input-source zmq-async`。

已完成：

- `python/examples/pattern_motion.py` 的 `zmq-async` 动态视觉输入路径已改为：
  - 先注册 native async source。
  - 先绑定 InputConv source。
  - 对每个动态视觉 frame 调用 `publish_frame_until_received(frame, sim, source_name, ...)`。
  - 以 C++ native source 的 `received_frames` 增量作为确认依据。
- 示例 summary/CLI 新增 native source 状态输出：
  - `source_received_frames`
  - `source_consumed_frames`
  - `source_queued_frames`
- 示例不再使用固定 `wait_ready(0.3) + publish_frames(... repeat=12) + wait_ready(0.1)` 作为主要同步逻辑。

验证结果：

- 相关测试子集通过：`2 passed, 37 deselected in 0.89s`。
- CLI smoke 通过：
  - `pattern_motion.py grating 0 --steps 2 --input-source zmq-async --frame-topic pattern_motion_status_wait_smoke`
  - 输出包含 `source_received_frames=4`、`source_consumed_frames=1`、`source_queued_frames=3`。
  - `published_frames=6`、`published_messages=6`，说明 helper 在 PUB/SUB 初期丢帧时按 native 状态进行了少量重发。
- Python 全量测试通过：`39 passed in 3.67s`。
- wheel 已刷新：
  - `D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
  - 大小约 `9513325` bytes。
- 离线包构建脚本已重新执行：
  - `D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`

当前判断：

- `publish_frame_until_received(...)` 已从单元/协议测试进入真实迁移示例路径。
- 现在 async dynamic visual input 的可靠性判断以 native C++ driver 真实接收为准，比固定 sleep 更适合后续 ROS/相机输入。
- PUB/SUB 本身仍无 ack；这里的确认是 producer 侧通过同进程 Simulation status 查询 C++ subscriber 状态，适合 neuronbridge Python 驱动仿真场景。

下一步建议：

1. 为 ROS image adapter 设计 `RosImageFramePublisher` / `RosImageFrameServer` 方案。
2. 为 spike interaction 设计同级 Python helper 与 status/stats。
3. 如未来外部 producer 不在同一 Python 进程，可考虑增加可选 ack/control socket，而不是依赖本地 `Simulation` status。

## 18. 2026-09-11 模型代码生成器收敛为通用 emitter

本轮纠正的架构问题：首批闭环曾使用 `lif_runtime.py` 和 `rstdp_runtime.py`，容易形成“每增加一个模型就复制一个 Python 生成脚本”的维护方式。该结构已经移除，不能作为后续开发模板。

已完成：

- 删除 `tools/model_codegen/lif_runtime.py` 与 `tools/model_codegen/rstdp_runtime.py`。
- 新增按模型种类组织的通用 `neuron_emitter.py` 与 `learning_rule_emitter.py`；不包含当前具体生成模型名称、批准模型白名单或硬编码 model id。
- 将 class name、variant id、Dense factory id、参数 API/default、Catalog category 和 backend 开关下沉到 `.nbmodel.json`。
- CLI 遍历全部 spec，统一生成每个 concrete C++/CUDA 类型以及固定聚合文件：
  - `GeneratedNeuronModels.cpp`
  - `GeneratedLearningRuleModels.cpp`
- CMake 只维护上述固定聚合单元，不再逐个列出 `CustomLifConductanceV1.cpp`、`CustomRStdpV1.cpp` 等具体模型源文件。
- 全量生成前只清理 generator 自己拥有的 `generated/model_codegen/include/neuronbridge_codegen` 与 `src`，避免删除 spec 后残留旧模型被继续编译。
- 新增结构测试：具体模型名称不得出现在通用 emitter/CMake 中；任意合法新名称和 ID 可由同一 emitter 生成；两个 neuron spec 可在一次 CLI 运行中同时进入聚合源、Factory registry 与 Dense model list。

验证结果：

- generator unittest：`24/24`（加入双 spec 测试后的目标值）。
- codegen ON Release 核心与 CUDA/Dense 构建通过；CTest `6/6`。
- `NR_ENABLE_MODEL_CODEGEN=OFF` 下 `nr_snn_core_gpuaware_cpp` 与 `nr_dense_runtime_gpuaware` 构建通过。
- Python 3.12 wheel 构建通过，SHA-256：`6DAC9F0DC6727F53D2F46CBB617E8956B2A4F36DB0EA5859F3BE9572D15BA8B0`。
- 新建隔离 venv 安装 wheel 后，native extension、CUDA、Dense generated LIF 和三种 generated learning rule runtime smoke 全部通过。

当前边界与维护规则：

1. 新增现有 IR/后端能力可表达的模型，只新增 spec、baseline fixture 和必要发布说明，不新增 `<model>.py`。
2. 只有新增语言能力或运行时 primitive 时才修改通用 parser/IR/emitter，并且实现不得判断具体模型名称。
3. 当前 neuron emitter 覆盖受控 conductance 模型结构；learning-rule emitter 覆盖二 trace pair 和三状态 trigger/eligibility 结构。尚不能宣称任意方程、分支、函数和状态类型均可生成。
4. 生成后的 C++/CUDA 热路径保持静态编译，不执行 Python、JSON parser 或 AST，因此本次架构修正不引入运行时解释开销。

## 19. 2026-09-12 模型级硬编码消除与通用 IR 收口

本轮目标：把上一轮“通用 emitter”从文件组织层面的通用，推进为真正由 spec、方程 AST 和语义角色驱动；禁止在生成器中通过具体模型名称、固定参数名或固定事件表达式选择实现。

已完成：

- 新增通用 parameter schema 编译：参数内部名、公开 API 名、aliases、存储类型、默认值和数值约束均来自 spec。
- neuron IR 可编译任意数量的状态，并显式要求每个状态声明 `forward_euler`、`exact_exponential` 或 `hold` 更新策略；状态 slot、初值、输入绑定、阈值 comparison 和 reset expression 均由 IR 持有。
- learning-rule IR 可编译任意数量的 CPU 状态及有序 `on_pre`、`on_post`、`on_trigger` 语句；不再按 `CustomPairStdpV1` 或 `CustomRStdpV1` 名称选择生成函数。
- Dense neuron lowering 通过 `conductance_lif_v1` capability 的 role map 将 spec 符号映射到固定高性能 ABI；阈值运算符与复位表达式直接由 AST 生成，不再固定为 `next_v > threshold` 与 `next_v = reset`。
- Dense learning lowering 通过 `pair_stdp_v1`、`reward_stdp_v1` capability role map 工作；state/parameter 可以重命名，事件 CUDA 语句来自 AST。
- Dense learning host wrapper 会把 spec 的公开 API/alias 归一到既有 Dense ABI key，并注入 spec 默认值；不再静默回落到手写父类的固定默认值。
- CLI 不再声明尚未实现的 `custom_dense_*` kind；Concrete class、Catalog、Factory registry 与 aggregate translation unit 均由实际 spec 集合确定。
- 新增结构与变异测试，证明以下修改会真实改变生成结果：状态/参数重命名、双 Euler 状态、comparison `>=`、reset 算术表达式、parameter API alias/default、学习事件顺序。

验证结果：

- generator unittest：`29/29` 通过。
- `NR_ENABLE_MODEL_CODEGEN=ON` Release MSVC/CUDA 12.6 全量构建通过。
- CTest：`6/6` 通过，覆盖 Catalog、CPU LIF、CPU R-STDP、Dense LIF GPU、Pair-STDP GPU 和 R-STDP GPU 轨迹。
- `NR_ENABLE_MODEL_CODEGEN=OFF` 的 `nr_snn_core_gpuaware_cpp` 与 `nr_dense_runtime_gpuaware` 构建通过。
- Python 3.12 wheel：
  - `D:\code_optimized\network_release\build_model_codegen_hardcoding_removal\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`
  - 大小 `9568976` bytes。
  - SHA-256 `9E6CD34057756343BA15803C52A6A7BF6AA005ED5F34245F98EA456427ABB37F`。
- 独立 venv `--no-deps` 安装后：native extension/CUDA/Dense runtime 可用；CPU wheel smoke `193` 个事件完全匹配；三种生成学习规则均在真实 CUDA backend 上产生权重变化。

长期维护约束：

1. 新增可由现有 IR 与 capability 表达的 concrete 模型时，只新增 `.nbmodel.json`、baseline fixture 和说明，不新增 `<model>.py`，也不在 emitter 中增加模型名分支。
2. `conductance_lif_v1`、`pair_stdp_v1`、`reward_stdp_v1` 是静态 Dense ABI capability，不是 concrete 模型白名单。只有新增状态布局、数学 primitive 或执行时序时才新增/扩展 capability。
3. CPU IR 比当前 Dense capability 更通用。不能宣称 Dense 已支持任意状态布局、任意函数或条件分支；不满足 capability 的 spec 必须在生成期失败，禁止静默生成语义不一致代码。
4. 生成期可以解析 JSON/AST；仿真热路径必须继续是静态 C++/CUDA，不允许引入 Python callback、字符串查找或运行时方程解释。
5. 每次扩展生成语言必须同时增加：parser/IR 负例、变异测试、真实 C++/CUDA 编译、CPU/GPU 数值轨迹和 wheel runtime smoke。

详细证据见：`reports/model_codegen_hardcoding_removal_20260912.md`。

## 20. 2026-09-12 受控数学函数 AST

本轮为 QIF、ExpIF/EIF、AdEx 等模型扩展通用表达式语言，不增加任何具体模型 emitter。

已完成：

- 新增通用 `FunctionCall(name, arguments)` AST 节点，支持函数嵌套和逗号分隔参数。
- 新增维护者所有的 `math_functions.py` 注册表；spec 不能注入 C++/CUDA symbol。
- 当前白名单：`exp`、`expm1`、`log`、`log1p`、`sqrt`、`abs`、`min`、`max`、`pow`。
- 固定参数数量：前六个为一元函数，后三个为二元函数；未知函数、零参数、少参数和多参数均在生成期拒绝。
- CPU 映射到 `std::exp/std::expm1/std::log/std::log1p/std::sqrt/std::fabs/std::fmin/std::fmax/std::pow`。
- CUDA 映射到对应 float device function：`expf/expm1f/logf/log1pf/sqrtf/fabsf/fminf/fmaxf/powf`。
- `identifiers_in` 已递归遍历函数参数，因此未知变量检查继续适用于 ODE、初值、spike comparison、reset 和 learning event。
- 修复科学计数法 literal 的 float suffix：`1e-3` 现在发射为合法的 `1e-3f`。
- CPU generated header 显式包含 `<cmath>`；Dense neuron 与 Dense learning event 使用 CUDA target lowering。

验证结果：

- generator unittest：`32/32` 通过。
- 新增 `model_codegen_math_functions` MSVC/NVCC 编译与真实 CUDA runtime 探针，九个函数的 CPU/GPU 聚合结果误差不超过 `1e-4`。
- CTest：`7/7` 通过；原有 LIF、Pair-STDP、R-STDP CPU/GPU baseline 无回归。
- Release 全量 `build_model_codegen_scaffold_check` 构建通过。

边界：

1. 当前没有完整单位/维度推导；`exp/log` 无量纲要求以及 `log/sqrt/pow` 定义域由 spec 参数约束和 baseline 负责。
2. 不在热路径自动插入 clamp、NaN/Inf 检查或异常处理，避免改变方程和性能。
3. `pow` 目前映射通用库函数；QIF 的平方项优先写成乘法，后续实现整数 `Power` 节点后可做静态乘法展开。
4. 函数 AST 已具备生成 CPU ExpIF/AdEx 方程的语言基础，但 Dense AdEx 仍需要新的结构化 adaptive-neuron capability 和双状态写回布局。

详细证据见：`reports/model_codegen_math_functions_20260912.md`。

## 21. 2026-09-12 主网 Legacy GPU 神经元生成闭环

本轮填补此前仅有主网 CPU 与 Dense GPU generated neuron、没有主网 Legacy GPU generated neuron 的验证空白。

已完成：

- `custom_lif_conductance_v1.nbmodel.json` 新增 `legacy_gpu` backend，并以 `conductance_lif_v1` role map 声明状态与参数语义；具体实现名为 `CustomLifConductanceV1_GPU`。
- 新增通用 `legacy_gpu_neuron_emitter.py`。它从受控方程 AST 生成直接 CUDA kernel、主网 `TimeDrivenNeuronModelGPU_Interface` concrete class、参数映射、输入路由、状态同步与 spike/reset 逻辑；生成器不判断具体模型名称。
- 生成固定聚合单元 `GeneratedLegacyGpuNeuronModels.cu`、头文件聚合和独立 GPU Factory registry；CMake 将聚合单元加入 `nr_snn_core_gpuaware_cuda`。
- Catalog 为同一 canonical model 同时声明 `LegacyCpu`、`LegacyGpu` 与 `DenseGpu`，并把 GPU concrete name 作为 alias。
- 新增 `model_codegen_legacy_gpu_lif_baseline`：检查 Catalog/Factory concrete type，并逐 tick 比较生成 CPU 与生成主网 GPU 的四状态轨迹和 spike。

验证结果：

- MSVC/CUDA 12.6 编译 `nr_snn_core_gpuaware_cpp`、`nr_snn_core_gpuaware_cuda` 和新测试目标通过，NVCC 明确编译了 `GeneratedLegacyGpuNeuronModels.cu`。
- 主网 GPU 基线：`1000` 个逐状态 tick、`115` 次 spike，四状态最大绝对误差 `3.8147e-06`，阈值 `2e-5`；完整 `Simulation` 调度下，同一 spec 的生成 CPU 与生成 GPU 的 `134` 个输出事件完全一致。
- generator unittest `33/33`、codegen native CTest `8/8` 通过。
- 该路径使用编译期专用 CUDA kernel，不在仿真热路径解释 JSON/AST，也不调用 Python。
- 修复 GPU 接口基类的未初始化析构缺陷：主网模型分组会销毁尚未调用 `InitStateVector()` 的临时 concrete model；旧生成析构读取未初始化 `sync_event` 时会触发 `0xC0000005`。现在 `TimeDrivenNeuronModelGPU_Interface` 构造器统一初始化 event、streams、device id/property，并由基类析构统一释放 event 与 streams；fixture 固定验证 CPU->GPU 与 GPU->CPU 两种 Simulation 构造顺序。

当前边界：首个主网 GPU capability 为 `conductance_lif_v1` 与 Forward Euler。新增不同状态布局（如 AdEx 的适应变量）需要新增结构化 capability 和对应轨迹基线，不能把当前 LIF role map 冒充为任意 GPU 方程支持。

范围澄清（2026-09-12）：当前验收不要求生成 QIF、ExpIF 或 AdEx 具体模型；数学函数节点通过 parser/IR、CPU/CUDA emitter 和独立 CUDA runtime 测试即可视为当前阶段完成。主网不存在独立的“GPU 学习规则”体系：GPU 神经元仍使用主网统一 LearningRule 语义和注册路径，因此不得把“主网 GPU learning-rule emitter”列为缺失项。后续只有在主网学习规则本身增加新状态布局或事件语义时，才扩展统一 learning-rule generator。

## 22. 2026-09-12 Neuron Catalog 参数查询接口

本轮采用最小元数据方案，不在 Catalog 或 spec 中重复维护单位、说明、别名、约束或参数分类。`NeuronModelCatalog` 继续作为 canonical model 与 backend binding 的唯一来源，模型参数和值继续由既有虚接口 `NeuronModel::getParameters()` 提供。

已完成：

- `NeuronModelCatalog::Entries()` 提供只读模型枚举。
- `NeuronModelFactory::QueryParameters()` 根据 Catalog 解析 backend implementation，构造轻量模型并调用真实 `getParameters()`；`QueryDefaultParameters()` 是空配置包装。
- Python 新增 `neuronbridge.catalog.list_models()` 与 `describe_model()`；结果包含 canonical name、选中 backend、implementation、supported backends 和原生参数字典。
- 对必须配置后才能构造的模型，`describe_model(parameters=...)` 统一传入构造参数；`HandwritingTimeDrivenModel` 通过 `role` 验证，没有模型专用查询分支。
- pybind 补齐嵌套 `ModelDescription` 以及 `std::array<float, 2/3/4/6>` 的返回转换。

验证结果：MSVC 2022/CUDA 12.6/Python 3.12 下核心、Catalog 测试和 `_core` 编译通过；C++ CTest `1/1`、Python Catalog 测试 `6/6` 通过。实际遍历查询全部 `12` 个 Catalog 模型，无 `<unsupported boost::any value>`。

长期约束：不得在 Python 或 Catalog 中复制手写模型参数默认值。新增/修改模型参数时，手写模型维护其 `getParameters()`，生成模型由通用 emitter 继续生成同一接口；Catalog 查询层只负责解析和转发。
