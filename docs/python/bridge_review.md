# neuronbridge 桥接实现审批说明

## 本轮实现范围

本轮覆盖 `neuronbridge` 第一阶段桥接实现，并完成第二阶段描述对象绑定。目标是先建立稳定的工程边界和 Python 用户 API 骨架，再把 Python 友好的网络描述对象转换到现有 C++ description 结构，而不是为每个 C++ 示例程序单独开访问通道。

已实现内容：

- 顶层 CMake 增加 `NR_ENABLE_PYTHON` 开关，默认 `ON`，使 Python 发布路径在日常构建中更早暴露依赖问题。
- 新增 `network_release/python` 子工程，用于构建 `neuronbridge._core` 原生扩展模块。
- 原生扩展采用 pybind11 维护路线，避免长期维护手写 CPython C API 引用计数和类型转换代码。
- 原生扩展已绑定 `npgr::DebugMonitorConfig`，并暴露 `get_build_info()` 与 DebugMonitor component kind 常量。
- Python 包名确定为 `neuronbridge`。
- 新增 `DebugMonitorResult`，用于懒读取 C++ DebugMonitor 输出目录中的 CSV/JSON 文件。
- 新增 `plot_spike_raster()` 和 `plot_state_trace()` 两个可视化入口。
- 新增 `examples/dense_debug_monitor.py`，作为从 C++ 示例输出迁移到 Python 可视化的第一个示例。
- 新增第二阶段 Python 描述对象：
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
- 新增 pybind11 native bridge 容器：
  - `neuronbridge._core.NetworkDescription`
  - `neuronbridge._core.SimulationConfig`
- 已支持 Python 描述对象转换为 C++ `NeuronLayerDescription`、`ConnectionDescription`、`LearningRuleDescription`、`OuterDynamicDescription`、`OuterDynamicConnectionDescription`、`InputConvDescription` list。
- 已支持 Python 参数字典转换到 C++ `std::map<std::string, boost::any>`，当前支持 bool、int、float、str 及其同类型 list。
- 已新增 `int32()`、`float32()`、`float64()`、`int32_list()`、`float32_list()`、`float64_list()`，用于需要精确 C++ `boost::any` 类型的模型参数。
- 已补充模型专用便捷构造器：`OuterDynamic.spike_counter()`、`OuterDynamic.planar_arm_2dof()`、`InputConv.v1_bar()`、`InputConv.v1_grating()`、`InputConv.v1_plaid()`、`InputConv.v1_file()`。
- 已新增 `examples/phase2_network_description.py`，用于展示第二阶段统一 Python API 如何表达普通网络、OuterDynamic、InputConv 和 typed parameters。
- 新增第三阶段首批真实 Simulation 生命周期绑定：
  - `Simulation.init()`
  - `Simulation.run()`
  - `Simulation.reset()`
  - `Simulation.add_external_spikes()`
  - `Simulation.add_external_currents()`
  - `Simulation.neuron_state()`
  - `Simulation.neuron_states()`
  - `Simulation.output_spikes()`
  - `Simulation.enable_debug_monitor()`
  - `Simulation.disable_debug_monitor()`
  - `Simulation.flush()`
  - `Simulation.result()`
  - `Simulation.get_connection_weight()`
  - `Simulation.set_connection_weight()`
  - `Simulation.save_weights()`
  - `Simulation.load_weights()`
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
  - `Simulation.outer_dynamic_spike_counter_snapshot()`
  - `Simulation.clear_outer_dynamic_spike_counter()`
  - `Simulation.clear_outer_dynamic_spike_counter_slot()`
- 已新增 `float32_array3()`、`float32_array4()`，用于桥接 `random_sigma` 等 C++ `std::array<float,N>` 参数。
- 已新增 `NeuronLayer.poisson_rate()`，用于通过统一 Python 网络描述表达 dense PoissonRate 子网络。
- `DebugMonitorConfig` 已补齐 `dense_local_neuron_ids`，可选择 dense subnetwork 内部 local neuron。
- `python/CMakeLists.txt` 已为 Python 扩展补入 `NR_ZMQ_CORE_SOURCES` 和 `NR_OUTER_DYNAMIC_SOURCES`，满足真实 `Simulation` 链接依赖。
- 已新增 `examples/phase3_run_simulation.py`，用于展示最小真实仿真运行、权重读写和 DebugMonitor flush。
- 已新增 `examples/dense_subnetwork_export.py`，用于展示 dense subnetwork snapshot/reset 的统一 Python API。
- 已新增 `examples/inputconv_poisson_dense.py`，用于展示 InputConvV1 输出到 dense PoissonRate 子网络并通过 DebugMonitor 可视化读取的统一 Python API。
- 已新增 `examples/_dense_baseline_builders.py`，沉淀 dense baseline 示例的统一 Python 网络构造逻辑。
- 已新增 `examples/dense_mixed.py` 和 `examples/dense_mixed_current.py`，继续按统一 `Simulation` API 迁移 C++ dense 示例。
- 已新增 `examples/dense_run_no_debug.py`，迁移无 DebugMonitor 的 dense baseline。
- 已新增 `examples/dense_subnetwork_smoke.py`，迁移 C++ smoke 中可通过公开 Python API 表达的 dense ownership 和 dense-to-dense 构图检查。
- 已新增 `examples/_ei_baseline_builders.py`、`examples/ei_program_dense_subnetwork.py` 和 `examples/ei_program_main.py`，迁移 dense/main EI benchmark 的 Python 构造与运行路径。
- 已新增 `examples/_pattern_motion_builder.py` 和 `examples/pattern_motion.py`，迁移 pattern-motion 的 Python 构造、刺激生成、运行与结果汇总路径。
- 已新增 `examples/my_first_app.py`，迁移 C++ `my_first_app` 最小仿真入口，并作为推荐的 Python 入门级示例。
- 已新增 `examples/_handwriting_stage1.py` 和 `examples/handwriting_stage1.py`，迁移 handwriting stage-1 的数值重放流程。
- 已新增 `examples/_handwriting_stage2.py` 和 `examples/handwriting_stage2.py`，迁移 handwriting stage-2 的 binary 输入、CM forward、population decode 和 replay 流程。
- 已新增 `examples/_handwriting_stage3.py` 和 `examples/handwriting_stage3.py`，迁移 handwriting stage-3 的输入加载、模式解析、目标路径和 `test/test_d/train_g/train_d` NumPy runtime。
- 已新增 `examples/handwriting_stage4_different_position.py`，迁移 handwriting stage-4 的 9 位置偏移重放流程。
- 已新增 `examples/handwriting_stage3_framework.py`，迁移 C++ framework 入口的命令行形态和输出契约，复用已对齐的 stage-3 shared runtime，不新增单示例 native 通道。
- 已新增 `examples/cerebellum_framework.py`，通过统一 `Network` / `LearningRule` / `OuterDynamic.strict_matlab_planar_arm_2dof()` / `OuterDynamicConnection` API 迁移 cerebellum framework 的可缩小规模真实仿真 smoke，并已按 C++ 样本循环注入 arm actual/desired state。
- 已新增 `examples/zmqcommunication_client.py`，迁移 ZMQ client 示例的本地仿真网络、协议 pack/unpack helper，并通过通用 `Simulation.add_zmq_async_input_output_spike_driver()` 支持 optional async ZMQ driver 注册和 Python `pyzmq` server 端到端验证。
- 已新增 `Simulation.outer_dynamic_state()`，用于读取 C++ `GetLatestOuterDynamicState` 的 latest joint state。
- 已新增 `Simulation.reset_outer_dynamic_state()` 和 `Simulation.set_outer_dynamic_desired_state()`，用于控制支持该语义的 2-DOF OuterDynamic 模型。
- 已新增 `Simulation.add_zmq_async_input_output_spike_driver()`，用于统一注册 C++ async ZMQ spike driver。
- 已新增 `Simulation.publish_output()`，用于显式触发 buffered output drivers 发布/刷出，ZMQ async driver 端到端测试依赖该通用入口。
- 已新增 `OuterDynamic.strict_matlab_planar_arm_2dof()`，用于表达 `StrictMatlabPlanarArm2DOFOuterDynamic`。
- 已完成 2026-09-07 新增 InputConv communication driver 的静态评估：C++ 侧已加入 `InputConvFrameDriver`、`InputConvFrameQueue`、`AsyncInputConvFrameDriver`、`ZMQInputConvFrameDriver`、`ZMQAsyncInputConvFrameDriver` 和 `Simulation` named frame-source/binding 入口；Python 侧下一步应以统一 `Simulation` API 暴露这些能力，不为 `inputconv_*_frame_smoke` 单独开 native 入口。
- 已完成 `simulation_weight_smoke` 的 Python baseline 对齐。
- 已完成 `outer_dynamic_spike_counter_smoke` 的 Python baseline 对齐。
- 已完成 `gpu_dense_subnetwork_export_real` 相关核心 API 的 Python baseline 对齐，覆盖 dense count/name/find、snapshot、GPU backend flags、权重向量、输入进入 dense runtime 和单 dense reset。
- 已完成 `inputconv_poisson_dense_real` 相关 Python baseline 对齐，采用统一 `Simulation` + `InputConvDescription` + `DebugMonitor` 路径，不为该示例新增专用 native 入口；当前 8x8x1 `InputConvV1` 配置实际输出数为 512。
- 已完成 `dense_mixed_current` 的 C++/Python baseline 对齐：C++ 与 Python 均输出 2 个 buffered spikes，分别为 `cell=2,t=4` 和 `cell=3,t=4`。
- `dense_mixed` Python 迁移版已运行通过；原 C++ baseline 的过时 dense layer `isOutput=true` 标记已修复，C++ exe 当前构建并运行通过。
- 已完成 `dense_no_debug` 的 C++/Python baseline 对齐：两侧均输出 `buffered_output_spike_count=4`，readout neuron 8/9 的 g 值一致。
- 已修复 `ei_program_dense_subnetwork` 共享构造中的过时 dense layer `isOutput=true` 标记；C++ baseline 与 Python 示例均可运行。该示例 C++ 自身重复运行时 spike totals 会变动，因此不作为固定 spike-count 单元基准。
- 已完成 `ei_program_main` 的 C++/Python baseline 对齐：两侧完整 2000-step 运行均输出 `total_e_spikes=7545`、`total_i_spikes=1861`。
- 已完成 `pattern_motion` 的 quick native smoke：通过统一 `Network`、dense Poisson/LIF layer、`InputConv.v1_file()` 和 `Simulation` API 完成 InputConvV1 文件刺激到 dense subnetwork 的初始化、运行和 snapshot 统计；未新增单示例 native 入口。
- 已完成 `pattern_motion --full` 的新版 C++ unified baseline 与 Python 长跑对比：两侧均为 grating dir0、32x32x8、2000 steps，方向 winner 指标全部对齐：`v1_spike_winner=2`、`v1_rate_winner=2`、`v1_current_update_mean_winner=2`、`cds_winner=2`、`pds_winner=2`、`pds_fs_winner=6`、`lip_winner=2`。

## 暂未实现内容

`Simulation` 高级对象已在 Python API 中接入真实 C++ 构造器和首批生命周期方法。dense debug snapshot、dense reset、InputConv monitor、output spike 读取、主网 neuron state 单个/批量读取、OuterDynamic latest/control state、ZMQ async spike driver 注册入口和 buffered output 发布入口已经完成。当前新增缺口来自 C++ 新扩展的 InputConv communication driver：Python 侧尚未暴露动态 frame source 注册、绑定、推帧、队列清空和 ZMQ frame source helper。

仍未完成的第三阶段扩展内容：

- `handwriting_stage3` 训练模式主仿真输出已完成 full C++ baseline 对齐；剩余 `w_mmbg1_final.tsv` 差异已定位到隐藏 BG/MM spike 轨迹的 update-count 分叉，若要逐权重完全一致需要新增 C++/Numba parity kernel
- `cerebellum_framework` 已完成约定缩小规模 analytic C++ baseline 对齐；默认 Pinocchio desired 轨迹生成仍可继续迁移
- ZMQ spike optional `pyzmq`/外部 peer 端到端 pub/sub 联调已通过；InputConv frame communication driver 已补 Python 绑定和协议 helper，本地 queued frame source native smoke 与同步 ZMQ InputConv native e2e 均已通过
- `pattern_motion` 连续值逐值数值级对齐仍可作为更严格验收继续推进
- Python 3.12 Windows wheel、基础离线 wheelhouse 包和同机离线 venv 验证已完成；后续仍需嵌入式 Python runtime 与无开发工具干净机器验证。

## 2026-09-07 InputConv communication driver 评估

C++ 本次新增的 InputConv 动态输入通道分为三层：

- 帧数据层：`InputConvPixelFormat`、`InputConvFrame`、`InputConvFrameRequest` 和 `InputConvFrameSourceBinding`，用于表达时间步、camera index、尺寸、通道数、像素格式和 payload。
- 本地/异步队列层：`InputConvFrameDriver` 是同步抽象；`InputConvFrameQueue` / `QueuedInputConvFrameDriver` 支持按 camera 缓冲并消费最新可用帧；`AsyncInputConvFrameDriver` 使用后台线程轮询并推入队列。
- ZMQ 通信层：`ZMQInputConvFrameDriver` 使用 request/response 模式按仿真请求拉帧；`ZMQAsyncInputConvFrameDriver` 使用 pub/sub 模式接收外部视觉帧流。

`Simulation` C++ 侧新增统一入口：

- `AddInputConvFrameSource(source_name, driver)`
- `AddAsyncInputConvFrameSource(source_name, driver, start_immediately)`
- `BindInputConvFrameSource(inputconv_index/name, source_name, source_camera_index)`
- `HasInputConvFrameSourceBinding(inputconv_index)`
- `LoadInputConvFrame(inputconv_index, time_step, frame)`
- `AddExternalInputConvFrames(source_name, frames)`
- `ClearInputConvFrameQueue(source_name)`
- `AddZMQInputConvFrameSource(source_name, address, port, max_payload_bytes)`
- `AddZMQAsyncInputConvFrameSource(source_name, subscribe_address, subscribe_port, topic, max_payload_bytes, max_buffered_frames_per_camera)`

`InputConvV1` 行为变化：

- 若当前 InputConv 已绑定 dynamic frame source，则 `Update()` 优先调用 `Simulation::LoadInputConvFrame()`。
- 拉帧成功后调用 `AcceptFrame()` 写入 `host_stimulus_`。
- 拉帧失败时按 `input_frame_missing_policy` 处理：`hold_last`、`fallback` 或默认 zero。
- 当前实际接收格式为 `UInt8Gray`、`UInt8RGB`、`UInt8BGR`；`Float32Gray/Float32HWC/Float32CHW` 只在枚举层存在，尚未被 `InputConvV1::AcceptFrame()` 实现。

Python 侧已新增 API：

- `InputConvPixelFormat`：Python 枚举，暴露 `uint8_gray`、`uint8_rgb`、`uint8_bgr`、`float32_gray`、`float32_hwc`、`float32_chw` 的协议表达；当前 `InputConvV1` native 实际接收仍以 UInt8 三类为准。
- `InputConvFrame`：Python dataclass，对齐 C++ frame 字段并校验 payload 长度。
- `InputConvFrameRequest`：Python dataclass，用于协议测试和外部 peer 请求解析。
- `Simulation.add_input_conv_frames(source_name, frames)`：将 Python 图像帧推入 C++ queued frame source。
- `Simulation.clear_input_conv_frame_queue(source_name)`：清空指定 frame source 队列。
- `Simulation.bind_input_conv_frame_source(index_or_name, source_name, source_camera_index=0)`：绑定某个 InputConv 到 source/camera。
- `Simulation.has_input_conv_frame_source_binding(index)`：查询某个 InputConv 是否已有动态帧源绑定。
- `Simulation.add_zmq_input_conv_frame_source(source_name, address, port, max_payload_bytes=0)`：同步 ZMQ request/response 拉帧。
- `Simulation.add_zmq_async_input_conv_frame_source(source_name, subscribe_address, subscribe_port, topic, max_payload_bytes=0, max_buffered_frames_per_camera=8)`：异步 ZMQ pub/sub 收帧。
- `pack_input_conv_frame()` / `unpack_input_conv_frame()` / `pack_input_conv_frame_request()` / `unpack_input_conv_frame_request()`：Python 测试和外部 peer 对接 helper，避免协议结构体在示例里重复手写。
- `InputConv.v1_bar()` / `v1_grating()` / `v1_plaid()` / `v1_file()` 已增加 `input_frame_missing_policy` 和 `input_frame_max_lag_steps` 便捷参数。
- `Simulation.add_input_conv_frames()` 在首次推帧前会自动 `init()`，避免 C++ `InitSimulation()` / `ResetForNextRound()` 清空预先写入的 frame queue。

验证状态：

- C++ 新增 smoke target 已在顶层工程中出现：`inputconv_dynamic_frame_smoke`、`inputconv_async_frame_smoke`、`inputconv_zmq_frame_smoke`。
- Python 现有测试在实现后通过：`pytest python\tests -q` 输出 `32 passed`。
- 新增 `test_input_conv_dynamic_frame_protocol_helpers`，覆盖同步 response、异步 frame 和 request 的 Python pack/unpack round-trip。
- 新增 `test_input_conv_dynamic_frame_queue_native`，覆盖 Python `InputConvFrame` 推入 C++ queued frame source，并经 `InputConvV1` 被 `Simulation.input_conv_input()` 读回。
- 新增 `test_input_conv_zmq_frame_source_native`，通过 Python REP server 连续响应 C++ `ZMQInputConvFrameDriver` request，并验证同步 ZMQ 帧进入 `InputConvV1` 输入缓冲。该测试已进入默认测试集；关键修正是 server 必须用 poll loop 连续响应，因为 `sim.run(1)` 会触发不止一次 InputConv request。
- `build_neuronbridge_wheel.ps1` 已重新构建成功，说明 `neuronbridge_core` 与新增 `ZMQInputConvFrameDriver.cpp` / `ZMQAsyncInputConvFrameDriver.cpp` 链接路径实际可用。
- Windows wheel 已刷新：`D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`，大小约 `9506274` bytes；`validate_neuronbridge_wheel.ps1` 在 `D:\code_optimized\network_release\build_neuronbridge_wheel_validate\venv_inputconv_frames` 中通过 native import，新增公开 API 可见性检查也通过。
- 基础离线包已用新 wheel 刷新：`D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`。

## 对现有 C++ 代码的影响

本轮新增代码遵循低侵入原则：

- 不修改仿真算法。
- 不修改现有 C++ 示例源码。
- 默认 CMake 构建会启用 Python 桥接，以对齐最终 Python 发布形态；如只验证传统 C++ 目标，可显式传入 `-DNR_ENABLE_PYTHON=OFF`。
- 不改变 DebugMonitor 当前输出格式。
- 复用已有 `nr_dense_runtime_gpuaware` 运行时目标。
- C++ 新增 InputConv frame-source 机制本身与 Python 统一桥接路线一致，但 `OutputSpikeDriver` 删除 `WriteState/IsWritePotentialCapable`、`OutputWeightDriver::WriteWeight()` 从 `Network*` 改为 `Simulation*`，对旧 monitor/output driver 属于潜在兼容性变化，需要 C++ 回归测试覆盖。

顶层 CMake 默认启用 Python 桥接。只需要传统 C++ 构建时，应显式关闭：`-DNR_ENABLE_PYTHON=OFF`。

## 审批建议

建议先审批以下设计决策：

- Python 包名采用 `neuronbridge`，C++ 内部命名空间 `npgr` 保持不变。
- 桥接目标名为 `neuronbridge._core`，Python 用户默认使用 `import neuronbridge as nb`。
- `NR_ENABLE_PYTHON` 默认开启，使 CI 和开发构建尽早发现 Python 3.12、pybind11、Python include/lib、native DLL 打包等问题。
- 第一阶段先支持 DebugMonitor 结果读取和可视化，作为已有 C++ 示例迁移到 Python 的低风险入口。
- 第二阶段描述对象绑定已完成，第三阶段首批真实 `Simulation` 生命周期已接入。
- 大文件结果读取默认采用 lazy iterator，只有用户显式调用 `to_pandas()` 时才整体加载表格。
- C++/Python 绑定维护基座采用 pybind11；CPython C API 不作为后续维护路线。

## 依赖接入方式

当前工程会在 `NR_ENABLE_PYTHON=ON` 时执行 `find_package(pybind11 CONFIG REQUIRED)`。推荐接入方式有三种，按优先级排序：

- 使用 vcpkg/conan 或系统包管理器安装 pybind11，并通过 `CMAKE_PREFIX_PATH` 让 CMake 找到 `pybind11Config.cmake`。
- 将 pybind11 作为受控 vendor 子模块放到 `dependency/vendor/pybind11`，再在 CMake 中增加 fallback `add_subdirectory()`。
- Python wheel 构建时由 `pyproject.toml` 的 build-system 依赖安装 `pybind11>=2.12`，供 scikit-build-core 配置阶段发现。
- 发布和验证基线采用 Python 3.12，`pyproject.toml` 约束为 `>=3.12,<3.13`。

## 发布版本依赖集成策略

开发验证环境可以使用本地 venv 安装 `pybind11`、`pytest`、`scikit-build-core` 等工具，但正式发布版本不能要求用户额外安装 C++ 编译器、pybind11、Pinocchio、ZeroMQ 或项目内部 native DLL。

建议发布形态分为两层：

- Python wheel：面向已有 Python 3.12 环境的用户。wheel 内应包含 `neuronbridge._core` 以及运行时所需的非系统 DLL，例如 Pinocchio、ZeroMQ、Boost、sdformat、coal、assimp、cudart 等。Windows wheel 建议在构建后使用 DLL 修复/打包步骤收集 native 依赖，使用户只需 `pip install neuronbridge-*.whl`。
- 完整离线包：面向没有 Python/CUDA 开发环境的用户。该包应包含 Python 运行时、`neuronbridge` wheel、项目 native DLL、示例脚本、可视化依赖和启动脚本。用户不应需要安装 MSVC、CMake、pybind11 或手工配置 PATH。

需要单独声明的外部条件：

- NVIDIA GPU 驱动属于系统级依赖，不能可靠地随 wheel 分发，应在安装文档中列为运行前条件。
- CUDA runtime DLL 可随包集成，但 CUDA driver 仍由显卡驱动提供。
- Python wheel 通常不打包 `python312.dll`；如果目标是完全免安装，应使用完整离线包或嵌入式 Python 发行方式，而不是普通 wheel。

建议后续补充自动化发布流水线：

- `build_wheel.ps1`：在 VS Developer 环境中配置并构建 wheel。
- `repair_wheel.ps1`：收集并修复 Windows native DLL 依赖。
- `package_portable.ps1`：生成包含 Python 运行时和示例的完整离线包。
- `validate_release.ps1`：在干净环境中执行 `import neuronbridge`、DebugMonitor 读取、示例可视化 smoke test。

## 已执行检查

- Python 模块语法检查通过。
- `import neuronbridge` 通过。
- `DebugMonitorConfig.to_dict()` 通过。
- `open_debug_monitor()` 可打开当前已有 `dense_debug_monitor_real_out` 目录。
- 已创建 Python 3.12 本地验证环境：`D:\code_optimized\network_release\.conda-neuronbridge-py312`。
- Python 3.12 构建工具已安装到：`D:\code_optimized\network_release\.conda-neuronbridge-py312\Lib\site-packages`。
- 已安装工具版本：`pybind11 3.1.0`、`pytest 9.1.1`、`scikit-build-core 1.0.3`。
- 已安装 NumPy：`numpy 2.5.2`，位置为 `D:\code_optimized\network_release\.conda-neuronbridge-py312\Lib\site-packages`。
- 使用 VS2022 Community Developer 环境、CUDA 12.6、Python 3.12、pybind11 完成 `neuronbridge_core` Release 构建。
- 生成 native 扩展：`D:\code_optimized\network_release\python\src\neuronbridge\_core.cp312-win_amd64.pyd`。
- native 导入测试通过，`backend_info()` 显示 `binding=pybind11`、`native_extension_loaded=True`。
- venv 下 `pytest network_release\python\tests -q` 通过，结果为 `29 passed`。
- `Network.to_native()` 验证通过，可将 Python 网络描述转换为 `neuronbridge._core.NetworkDescription`。
- `NetworkDescription.to_dict()` 验证通过，可读回 layer、connection、learning rule 的关键字段。
- `SimulationConfig.to_native()` 验证通过，可转换为 `neuronbridge._core.SimulationConfig`。
- `OuterDynamic`、`OuterDynamicConnection`、`InputConv` native 往返验证通过。
- `float64_list()` 到 C++ `std::vector<double>` 的类型保留验证通过。
- `examples/phase2_network_description.py` 运行通过，并可输出 native bridge summary。
- `examples/phase3_run_simulation.py` 运行通过，可构造真实 C++ `Simulation`、读写连接权重、注入 spike、运行仿真并 flush DebugMonitor。
- `examples/dense_subnetwork_export.py` 运行通过，可输出 dense 子网络名称、snapshot 步数、神经元数、权重数、非零 `gexc` 数和 reset 后 firing 数。
- `examples/inputconv_poisson_dense.py` 运行通过，可输出 InputConv 数量、输出数量 512、dense snapshot 神经元数和 DebugMonitor 文件大小。
- `examples/dense_mixed.py` 运行通过，可输出 dense 子网络名称、snapshot 步数、神经元数、synapse 数和 DebugMonitor 文件大小。
- `examples/dense_mixed_current.py` 运行通过，可输出 `buffered_spike_count 2`，且两个 spike 为 `cell=2,t=4` 与 `cell=3,t=4`。
- `examples/dense_run_no_debug.py` 运行通过，可输出 `buffered_output_spike_count=4` 和 readout neuron 8/9 的 g 值。
- `examples/dense_subnetwork_smoke.py` 运行通过，可输出 dense ownership reject、dense output reject、TriggerRelay 构图和 dense-to-dense 构图 smoke 结果。
- `examples/ei_program_dense_subnetwork.py` 运行通过，可读取 EI connectivity 数据并运行 2000-step dense EI benchmark。
- `examples/ei_program_main.py` 运行通过，可读取 EI connectivity 数据并运行 2000-step main-network EI benchmark。
- `examples/pattern_motion.py grating 0 --steps 2` 运行通过，可生成 `selected_stimulus.dat`、TSV spike counts 和 `summary.txt`，并完成 dense/InputConv snapshot 路径验证。
- `examples/pattern_motion.py grating 0 --full` 运行通过，可生成 `v1_rate_sums.tsv`、`v1_current_update_mean.tsv`、`inputconv_rate_heatmap.bmp` 和 `summary.txt`；如果运行环境安装了 matplotlib，还会额外尝试生成 PNG。
- `examples/my_first_app.py` 运行通过，可构造 4 输入到 4 LIF 输出神经元的 all-to-all 最小网络，完成 100-step native simulation，并读取 `buffered_output_spike_count=40`。
- `examples/handwriting_stage1.py` 运行通过，可读取项目自带 stage-1 X/Y stroke 序列，输出 A/Q/desired/replay/summary TSV。
- `examples/handwriting_stage2.py` 完整 400-window、3-stroke 运行通过，可输出 CM window counts、QQ、summary 和 replay TSV。
- `examples/handwriting_stage3.py test --run` NumPy full runtime 运行通过，可读取 stage-3 test 数据并输出 `simulated_steps=90000`、`window_count=900`、`total_population_spikes=30797`、`summary.tsv`，并已完成 C++ full baseline 对齐。
- `examples/handwriting_stage4_different_position.py` 运行通过，可输出 9 个 shift 的 target/replay path、combined path 和 summary TSV。
- `simulation_weight_smoke` C++ exe 在 `build_neuronbridge_baseline_smokes` 中构建并运行通过；Python baseline 覆盖相同权重数值和 boundary write rejection。
- `outer_dynamic_spike_counter_smoke` C++ exe 在 `build_neuronbridge_baseline_smokes` 中构建并运行通过；Python baseline 覆盖相同 slot/type count 和 weighted sum 数值。
- `gpu_dense_subnetwork_export_real` 和 `inputconv_poisson_dense_real` 的 Python baseline 已通过。当前顶层 `NR_ALL_PROJECTS` 尚未收录这两个 C++ 测试 target，因此本轮没有通过普通 `NR_PROJECTS` target 名直接构建到对应 exe；Python 侧以同源 C++ 测试代码行为为参照，走统一 `Simulation` API 完成可观测行为对齐。
- `dense_mixed_current` C++ exe 在 `build_neuronbridge_baseline_migration` 中构建并运行通过；关键输出为 `buffered_spike_count=2`、`output_spike[0] cell=2 t=4`、`output_spike[1] cell=3 t=4`、`readout_g neuron9=0.505726 neuron10=0.505726`。
- `dense_mixed` C++ exe 在 `build_neuronbridge_baseline_migration` 中构建并运行通过；关键输出为 `relay_present=1 dense1_removed=1 dense2_removed=1 readout_present=1`、`readout_g_with_relay neuron10=3.65232 neuron11=3.65232`、`stage=done`。Python 迁移版采用同样的 dense->main route 表达并运行通过。
- `dense_no_debug` C++ exe 在 `build_neuronbridge_baseline_migration` 中构建并运行通过；Python 迁移版对齐 `buffered_output_spike_count=4` 和 readout g 值。
- `ei_program_dense_subnetwork` C++ exe 在 `build_neuronbridge_baseline_migration` 中构建并运行通过；共享 EI dense 构造已修复 `isOutput=true` 冲突。该示例 C++ 自身 spike totals 非确定，不写入固定数值 pytest。
- `ei_program_main` C++ exe 在 `build_neuronbridge_baseline_migration` 中构建并运行通过；Python 迁移版完整 2000-step 对齐 `total_e_spikes=7545`、`total_i_spikes=1861`。
- `pattern_motion` C++ baseline 已在 `build_neuronbridge_pattern_motion_baseline` 中构建并运行通过；新版 C++ grating dir0 输出 `dense_build_path=unified_simulation`，并与 Python full 的所有方向 winner 指标一致。
- `my_first_app` C++ baseline 已在 `build_neuronbridge_my_first_app_baseline` 中构建并运行通过；Python 迁移版基础输出字段与 C++ 对齐，并通过统一 API 额外读取 output spike 缓冲。
- `handwriting_stage1` C++ baseline 已通过既有 `build_gpu_probe` 运行；Python 迁移版三个 stroke 的 summary MSE 与 C++ 输出在 `1e-13` 容差内对齐。
- `handwriting_stage2` C++ baseline 已通过既有 `build_gpu_probe` 运行；Python 完整迁移版三个 stroke 的 summary MSE 与 C++ 输出最大差异约 `3.6e-12`。
- `handwriting_stage4_different_position` C++ baseline 已通过既有 `build_gpu_probe` 运行；Python 迁移版 stroke 1 九个 shift 的 summary MSE 与 C++ 输出最大差异约 `2.1e-10`。
- `handwriting_stage3` NumPy full runtime 已完成 `test` 模式 90,000-step 长跑并生成 `QQ.tsv`、`pop_spk.tsv`、`replay_path.tsv`、`target_path.tsv`、`summary.tsv`；C++ full baseline 已完整跑完并完成对比：`pop_spk.tsv` 逐 cell 完全一致，双方 `total_population_spikes=30797`；`QQ.tsv` 最大差异约 `5e-8`，`summary.tsv` 最大差异约 `1.4e-11`，`replay_path.tsv` / `target_path.tsv` 最大差异约 `5e-7`。
- `handwriting_stage3` 训练模式已补 Hebbian 更新路径和 MSVC `std::mt19937` 随机序列复现：`train_g` 与 C++ baseline 的 `pop_spk.tsv` 逐 cell 完全一致，`total_population_spikes=30614`；`train_d` 与 C++ baseline 的 `pop_spk.tsv` 逐 cell 完全一致，`total_population_spikes=23471`，且 `w_mmbg2_final.tsv` 完全一致。当前剩余差异已通过 `HANDWRITING_STAGE3_DIAG_DIR` 诊断定位到隐藏 BG/MM spike 轨迹的 update-count 分叉。
- `cerebellum_framework` 通过新增 OuterDynamic control API 对齐约定缩小规模 analytic C++ baseline：`sample_count=4`、`nn=2`、`nv=2`、`n_pc/n_cf/n_dcn=4`、CPU neurons、无 dense subnetwork、`CEREBELLUM_USE_PINOCCHIO_DESIRED=0`；Python/C++ `joint_state.tsv` 最大差异约 `4.9e-4`，对应 C++ 默认文本输出精度。
- ZMQ optional 端到端验证已完成：`pyzmq 27.2.0` 安装在 `D:\code_optimized\network_release\.conda-neuronbridge-py312\Lib\site-packages`；Python server + native async ZMQ driver 联调中，server 收到 51 个 SNN output batch，client `--enable-zmq` 输出 `buffered_output_spikes=10`。
- Windows wheel 已生成：`D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`。新增 `build_neuronbridge_wheel.ps1`、`validate_neuronbridge_wheel.ps1`、`check_neuronbridge_runtime_deps.ps1`；最终 wheel 已排除历史 `neuronbridge/Release/**` 目录，并在 `D:\code_optimized\network_release\build_neuronbridge_wheel_validate\venv_nodeps_final` 中通过 native import smoke。
- Python 3.12 基础离线包已生成：`D:\code_optimized\network_release\release\neuronbridge_offline_py312`，压缩文件为 `D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`。新增 `build_neuronbridge_offline_bundle.ps1`，已收集 `neuronbridge` wheel、`numpy/matplotlib/pandas/pyzmq` wheelhouse、迁移示例、文档、离线安装脚本和 smoke-test 脚本；同机干净 venv `D:\code_optimized\network_release\release\neuronbridge_offline_py312\.venv-offline-validate` 已通过 `--no-index --find-links` 离线安装和 native import。
- `pattern_motion` Python full 已运行通过；Python grating dir0 输出目录为 `D:\code_optimized\network_release\pattern_motion_output\python_full_grating_dir0_unified_after_fix`，包含 `summary.txt`、V1 rate/current TSV 和 `inputconv_rate_heatmap.bmp`。

未完成检查：

- 尚未在无开发工具的干净机器上做离线安装验证；当前是同机干净 venv 验证。
- 尚未把离线包升级为完全自带 Python runtime 的便携包；当前包已经收拢 wheel 和 Python 依赖，但仍需要用户提供 Python 3.12 或后续嵌入 Python 3.12 runtime。
- 尚未完成 `pattern_motion --full` 的连续值逐值数值级对齐；当前方向 winner baseline 已收敛。

## 2026-09-07 InputConv Communication Driver 闭环更新

本轮把 InputConv dynamic frame communication 的 C++/Python 双侧验证推进到可验收状态：

- Python 高层 API 已修正通信源注册生命周期：`add_zmq_input_conv_frame_source()` / `add_zmq_async_input_conv_frame_source()` 现在会先 `init()`，避免首次 `run()` 时 native 初始化清掉早到 frame；`add_input_conv_frames()` 继续保持同样策略。
- Python 新增 `test_input_conv_zmq_async_frame_source_native`，覆盖异步 ZMQ `PUB/SUB` 的 `topic + 36-byte async header + payload` 三段消息协议；C++ 后台 worker 入队后，`Simulation.run()` 可消费并通过 `input_conv_input()` 读回发布帧。
- Python 全量测试通过：`33 passed in 2.77s`。
- C++ 三个 smoke target 已构建并运行通过：
  - `inputconv_dynamic_frame_smoke`
  - `inputconv_async_frame_smoke`
  - `inputconv_zmq_frame_smoke`
- C++ `inputconv_dynamic_frame_smoke` baseline 修复为先 `InitSimulation()` 再注入本地 frame queue，和 Python facade 的使用语义一致。
- Windows wheel 已刷新并通过干净 venv 验证：`D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`，大小 `9506279` bytes；验证环境为 `D:\code_optimized\network_release\build_neuronbridge_wheel_validate\venv_inputconv_comm`。
- 离线包中的 `neuronbridge` wheel 已替换为最新版本，并重新生成 `D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`，大小约 `134433932` bytes。
- 离线 wheelhouse 安装验证已通过：`D:\code_optimized\network_release\release\neuronbridge_offline_py312\.venv-inputconv-comm` 从本地 wheels 安装 `neuronbridge[communication]`，并成功加载 native extension。

本轮发现并记录的工程问题：

- MSBuild 仍会在部分 target 上遇到 `MSB6001`，根因是 Codex/PowerShell 进程环境中 `PATH` 与 `Path` 重复；新增 `D:\code_optimized\network_release\tools\run_clean_env_passthrough.ps1` 作为本地验证辅助脚本，清理重复环境变量且直接透传 MSBuild 输出。
- 原 clean-env wrapper 使用同步 `ReadToEnd()` 读取 stdout/stderr，在 MSBuild 大输出场景下可能无输出卡住；后续应修复为并发读取或无重定向透传。

## 2026-09-07 协议负例与实际示例迁移更新

本轮在 InputConv communication driver 正向闭环基础上补了负例和真实示例路径：

- 协议 helper 负例已覆盖 request/response/async header magic 错误，以及 payload 截断。
- native 负例已覆盖 queued frame 尺寸不匹配、max lag 过期、同步 ZMQ 坏 response magic、异步 ZMQ wrong topic 忽略后 good topic 恢复。
- `pattern_motion.py` 新增 `--input-source file|queue|zmq-async`：
  - `file` 保持原始 stimulus file 路径。
  - `queue` 使用统一 `Simulation.add_input_conv_frames()` 注入动态视觉帧。
  - `zmq-async` 使用 Python `PUB` + C++ `ZMQAsyncInputConvFrameDriver` 的真实通信路径。
- `test_pattern_motion_dynamic_input_sources` 已覆盖 queue 与 zmq-async 两条实际示例迁移路径，并验证关键 winner 指标一致。

验证结果：

- 目标测试子集：`7 passed, 30 deselected in 2.71s`。
- Python 全量测试：`37 passed in 4.42s`。
- CLI 示例 smoke：
  - `pattern_motion.py grating 0 --steps 2 --input-source queue`
  - `pattern_motion.py grating 0 --steps 2 --input-source zmq-async --frame-topic cli_pattern_motion_inputconv`
  两者输出一致，`v1_rate_winner=2`、`v1_current_update_mean_winner=0`、`dense_subnetwork_count=1`。
- wheel 已刷新并在 `D:\code_optimized\network_release\build_neuronbridge_wheel_validate\venv_inputconv_negative_examples` 通过 native import。
- 离线包已用 `-SkipDownload` 重新生成，当前 `D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip` 大小约 `52037528` bytes，并在 `D:\code_optimized\network_release\release\neuronbridge_offline_py312\.venv-inputconv-negative-examples` 通过本地 wheelhouse 安装验证。

## 2026-09-08 InputConv Communication Python API 封装更新

本轮按“REQ/REP 保留、PUB/SUB 保留、Python 侧正式封装”的方案完成实现：

- 新增 `python/src/neuronbridge/communication.py`。
- 新增并导出 `InputConvFramePublisher`：
  - 对应异步 ZMQ `PUB/SUB`。
  - 自动打包 `topic + async header + payload`。
  - 支持随机端口、context manager、`publish_frame()`、`publish_frames()`、`summary()` 和发布统计。
- 新增并导出 `InputConvFrameServer`：
  - 对应同步 ZMQ `REQ/REP`。
  - 后台线程处理 C++ `InputConvFrameRequest`。
  - 通过 Python `frame_provider(request)` 返回 `InputConvFrame`。
  - provider 错误时返回零帧，避免 REQ/REP 状态机死锁。
- `pattern_motion.py --input-source zmq-async` 已改用 `nb.InputConvFramePublisher`，示例中不再手写裸 ZMQ multipart 发送；summary/CLI 新增 `frame_topic`、`published_frames`、`published_messages`。
- 同步正向测试切到 `InputConvFrameServer`；异步正向和 wrong-topic 恢复测试切到 `InputConvFramePublisher`；保留少量裸 ZMQ 仅用于故意发送坏协议负例。

验证结果：

- 目标测试子集：`5 passed, 33 deselected in 4.74s`。
- Python 全量测试：`38 passed in 8.23s`。
- CLI smoke：
  - queue 模式：`published_frames=4`、`published_messages=4`。
  - zmq-async 模式：`published_frames=4`、`published_messages=48`。
  两者 `v1_rate_winner=2`、`v1_current_update_mean_winner=0`。
- wheel 已刷新并通过干净 venv 验证：`D:\code_optimized\network_release\build_neuronbridge_wheel_validate\venv_inputconv_comm_api`。
- 离线包已用 `-SkipDownload` 重打，并在 `D:\code_optimized\network_release\release\neuronbridge_offline_py312\.venv-inputconv-comm-api` 通过本地 wheelhouse 安装验证；新 API 可 import。

## 2026-09-08 Async InputConv Driver 与 Watchdog 解耦判断

本轮确认了 async ZMQ InputConv driver 的运行模式：

- `AsyncInputConvFrameDriver::Start()` 自己启动 `std::thread(&AsyncInputConvFrameDriver::WorkerLoop, this)`。
- `Simulation::AddZMQAsyncInputConvFrameSource(...)` 会调用 `AddAsyncInputConvFrameSource(..., true)`，因此注册 async frame source 时就会启动 communication worker。
- 普通 `Simulation.run()` / `RunSimulationStep()` 中，`InputConvV1::Update()` 通过 `simulation->LoadInputConvFrame(...)` 从 async driver 队列取帧。
- watchdog / `SyncThread` / `RealTimeRestriction` 属于实时 pacing 和同步事件机制，不是 async InputConv frame driver 的必要启动条件。

结论：

- async ZMQ InputConv driver 可以在普通仿真模式下运行，不要求开启 watchdog 同步线程。
- 当前普通模式验证已经覆盖：
  - `test_input_conv_zmq_async_frame_source_native`
  - `test_pattern_motion_dynamic_input_sources`
  - `pattern_motion.py --input-source zmq-async`
- 当前 `PUB/SUB` 需要 warmup/repeat 的原因是 ZeroMQ 订阅握手可能丢早期消息，不是 watchdog 未开启。

后续应把这个判断转化为代码能力：

- 补 C++ frame source status/stats。
- 绑定 Python `Simulation.input_conv_frame_source_status(...)`。
- 用真实 received/queued 状态增强 `InputConvFramePublisher.wait_ready()`。
- ROS 耦合默认可以走普通 `sim.run()` 消费 async image/spike 输入；只有需要 wall-clock pacing 时才启用 realtime watchdog。

## 2026-09-08 InputConv Frame Source Status/Stats 落地

本轮已把上一节的 status/stats 目标落成代码：

- C++ 通用状态结构：`InputConvFrameSourceStats`。
- C++ 通用入口：`InputConvFrameDriver::GetStats()`。
- queued source 统计：received、consumed、dropped、queued、last received/consumed timestep。
- async source 统计：worker running、received、consumed、failed_requests、dropped、queued、last_error。
- sync ZMQ source 统计：requested、received、failed_requests、last_error。
- `Simulation` 暴露按 source name / source index 查询状态的 C++ API。
- pybind/Python 暴露 `Simulation.input_conv_frame_source_status(source_name_or_index)`。
- 同步 ZMQ 坏 magic 负例已修复 multipart payload 排空，避免 REQ socket 状态机被坏帧破坏。

验证：

- InputConv 通信状态测试子集：`5 passed, 33 deselected in 1.87s`。
- Python 全量测试：`38 passed in 4.66s`。
- 最新 wheel：`D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`，大小约 `9512321` bytes。
- source-tree native import smoke 已确认新 API 可见。
- 离线发布包已刷新：`D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`，大小约 `52049747` bytes。

下一步：

- 把 Python publisher 的 ready/warmup 从固定 sleep 逐步升级为基于 native source status 的等待。
- 补同步 REQ/REP 的 Python client/server 适配层，使 ROS 或其他外部系统接入时可以选择 clock-driven pull 或 async push。
- 给 ROS adapter 方案预留 image encoding、timestamp/frame_id、camera_index/topic 映射，以及 spike topic 的 batch/stream 双模式。

## 2026-09-08 Async Wait 与 Sync REQ/REP 适配封装落地

本轮完成上一节的两个封装目标：

- `InputConvFramePublisher.wait_for_native_receiver(...)`：基于 `Simulation.input_conv_frame_source_status(...)` 等待 native source 的 `received_frames` 达标。
- `InputConvFramePublisher.publish_frame_until_received(...)`：发布 frame 并按 native received stats 自动重试，降低 PUB/SUB warmup 固定 sleep 的依赖。
- `InputConvFrameClient`：新增 Python 同步 REQ client，可向符合协议的 server 请求 `InputConvFrame`。
- `InputConvFrameServer`：继续作为 Python REP server，与新增 client 构成 Python 双端协议适配。
- `Simulation.add_input_conv_frame_server_source(...)`：把 Python server 直接注册为 native sync InputConv frame source，并可选自动绑定 InputConv。
- `neuronbridge` 顶层已导出 `InputConvFrameClient`。

验证：

- 新封装目标测试：`5 passed, 34 deselected in 1.37s`。
- Python 全量测试：`39 passed in 4.12s`。
- 最新 wheel：`D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`，大小约 `9513325` bytes。
- 离线发布包已刷新：`D:\code_optimized\network_release\release\neuronbridge_offline_py312.zip`，大小约 `52050961` bytes。
- import smoke 已确认 `InputConvFrameClient` 与 `Simulation.add_input_conv_frame_server_source` 可见。

后续重点：

- 把新 async wait helper 接入实际迁移示例，优先是 `pattern_motion.py --input-source zmq-async`。
- 开始 ROS image/spike adapter API 设计，把当前 frame status 作为连接健康检查基础。

## 2026-09-08 pattern_motion 使用 Native Status Wait

本轮已把 `publish_frame_until_received(...)` 接入 `pattern_motion.py --input-source zmq-async`：

- 示例注册 native async source 后立即绑定 InputConv。
- 每个动态视觉 frame 都通过 native `received_frames` 增量确认送达。
- summary/CLI 新增 `source_received_frames`、`source_consumed_frames`、`source_queued_frames`。
- 移除该路径中作为主同步方式的固定 warmup sleep 和大 repeat。

验证：

- 相关测试：`2 passed, 37 deselected in 0.89s`。
- Python 全量测试：`39 passed in 3.67s`。
- CLI smoke 输出 `source_received_frames=4`、`source_consumed_frames=1`、`source_queued_frames=3`。
- wheel 已刷新：`D:\code_optimized\network_release\python\dist\neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`，大小约 `9513325` bytes。
- 离线包构建脚本已重新执行。

后续重点：

- ROS image/spike adapter API 设计。
- 外部进程 producer 场景下，可评估是否需要协议级 ack/control socket。
