# NeuronBridge 自定义方程模型：详细代码修改清单

**状态：实施中；通用 neuron/learning-rule emitter、固定聚合编译单元、Catalog/Factory/Dense CUDA 注册及首批 baseline 已落地。**
**目标：在不改变既有手写模型默认行为的前提下，使 `.nbmodel.json` 可生成主网/Dense neuron 与 learning rule，并完整接入 Catalog、Factory、事件流、Python 和发布构建。**

相关文档：

- [总体维护架构](model_codegen_maintenance_architecture.md)
- [维护者方程规格](neuronbridge_custom_equation_format.md)
- [具体实现设计](custom_model_codegen_implementation_design.md)
- [代码改动蓝图](custom_model_codegen_patch_blueprint.md)

## 1. 实施边界与不可变约束

### 1.1 不改动的稳定资产

以下文件中的现有模型和名称保持权威、默认且兼容：

```text
source_file/legacy/source_file_realtime_v1_async/NeuralModel/inc/TimeDriven/
source_file/legacy/source_file_realtime_v1_async/NeuralModel/src/TimeDriven/
source_file/legacy/source_file_realtime_v1_async/LearningRule/inc/Rule/
source_file/legacy/source_file_realtime_v1_async/LearningRule/src/Rule/
source_file/gpu/include/dense_subnetwork/model/Dense*Model.h
source_file/gpu/include/dense_subnetwork/learning/Dense*LearningModel.h
```

不得重命名、替换或改变既有 `TimeDrivenLIF_Exponential_double`、`PoissonRate`、`STDP`、`R_STDP` 等 canonical model name 的默认解析。

### 1.2 生成模型的身份规则

每个生成模型必须具有永久唯一且彼此不同的 `variant_id`、`implementation_name`、`canonical_name`。首版 custom variant 使用独立名称，例如 `CustomLifConductanceV1`；不得将既有模型名用作 alias、canonical name 或 implementation name。除非完成公开审批，`public_user_model` 必须为 `false`。

### 1.3 热路径禁止项

JSON、字符串、`std::map`、AST、单位对象、Python callback、动态内存分配和锁只允许发生在 configure/build/init 阶段，禁止进入 per-neuron、per-synapse 或 CUDA device 热路径。

### 1.4 通用生成器约束

生成器按模型种类和目标后端组织，不按具体模型组织。最终结构必须收敛为通用 `NeuronIR`、`LearningRuleIR`、Legacy CPU emitter、Dense CUDA emitter、Catalog/Factory emitter 和固定聚合产物；不得保留“新增一个模型就新增一个 `<model>_runtime.py`”的维护方式。

当 DSL 已能表达一个新模型时，新增模型只允许修改：

```text
model_specs/<model>.nbmodel.json
tests/model_codegen/<model>_baseline.*
必要的发布说明
```

不得为该模型修改 Python generator、Factory switch、Catalog 表、CUDA dispatch 或 CMake 源文件清单。只有引入新的 AST 节点、积分算法、事件语义、状态存储类型或后端 primitive 时，才允许扩展通用生成器，并且扩展必须同时服务于所有规格而不是检查具体模型名称。

阶段性的 `lif_runtime.py` 和 `rstdp_runtime.py` 已于 2026-09-11 删除，由 `neuron_emitter.py` 和 `learning_rule_emitter.py` 取代。CLI 只导入这两个按模型种类组织的通用 emitter；具体类名、variant id、Dense factory id、参数声明、Catalog 分类和后端开关均来自 spec。CMake 只编译固定的 `GeneratedNeuronModels.cpp` 与 `GeneratedLearningRuleModels.cpp` 聚合单元，不再列举具体生成模型。

当前通用性的准确边界：新增一个已被现有 IR 支持的 conductance neuron、二 trace pair rule 或三状态 trigger/eligibility rule，不需要新增或复制 Python 文件；如果新模型需要新的 AST 节点、积分算法、状态类型、事件语义或 CUDA primitive，才扩展对应通用 emitter。这里的“通用”指 spec 驱动且不绑定模型身份，不等于已支持任意数学语言。

## 2. 变更总览

| 工作包 | 新增/修改范围 | 依赖 | 首个交付物 |
|---|---|---|---|
| W0 | 规格、manifest、维护者 generator | 无 | CI 可执行的 spec validation |
| W1 | Catalog 可组合注册 | W0 | custom name 可解析 |
| W2 | 主网 CPU custom neuron | W1 | Custom LIF 可跑 |
| W3 | 主网 custom learning rule | W1 | Custom R-STDP 可跑 |
| W4 | Dense custom neuron | W1, W2 | Dense custom LIF CUDA update |
| W5 | Dense custom learning rule | W1, W3 | Dense custom R-STDP CUDA hooks |
| W6 | 官方 wheel、Python schema/API | W0-W2 | 官方发布可安装的 CPU wheel |
| W7 | baseline、性能、打包门禁 | W2-W6 | 发布级验证 |

## 3. W0：规格、解析与生成器

### 3.1 新增目录与文件

```text
tools/model_codegen/
  __init__.py
  cli.py
  schema.py
  json_loader.py
  equation_lexer.py
  equation_parser.py
  expression_ast.py
  unit_checker.py
  model_validator.py
  manifest.py
  emit_cpp.py
  emit_cuda.py
  emit_catalog.py
  emit_python.py
  emit_tests.py

model_specs/
  custom_neurons/
  custom_learning_rules/
  custom_dense_neurons/
  custom_dense_learning_rules/

tests/model_codegen/
  test_json_loader.py
  test_equation_parser.py
  test_unit_checker.py
  test_name_collision.py
  fixtures/
```

### 3.2 生成器 CLI

新增仅供维护者和 CI 在 validate/build 阶段运行的入口：

```text
python -m tools.model_codegen.cli
  --spec-root <source>/model_specs
  --output-root <build>/generated/model_codegen
  --manifest <build>/generated/model_codegen/manifest.json
```

必须提供 `--check` 模式：只解析、校验并输出诊断，不写 C++/CUDA 文件。

该命令不安装为 `neuronbridge` 用户 CLI，也不接受最终用户运行目录中的 spec。维护者通过源码工作树、CMake target 和 CI 调用它；生成模型只随官方 wheel 发布。

### 3.3 必须实现的验证

- JSON schema、`schema_version`、kind、base class contract digest。
- `variant_id`、canonical name、implementation name、alias 的全局唯一性。
- 禁止与手写 Catalog 条目冲突。
- 方程 AST、变量依赖环、未声明变量、只读变量赋值。
- 单位、参数范围、event assignment 类型。
- 主网模型的 `synaptic_inputs` 与 `Interconnections::type` 映射。
- Dense 模型的 input channel、spike effect、field slot 和 CUDA 支持限制。
- `on_pre`、`on_post`、`on_trigger` 与 rule kind 的一致性。
- manifest 中记录 spec 哈希、generator 版本、base contract digest、生成文件清单。

### 3.4 生成产物

```text
${build}/generated/model_codegen/
  include/neuronbridge_codegen/
    CustomNeuronCatalog.inc
    CustomLearningRuleCatalog.inc
    CustomLegacyNeuronRegistry.inc
    CustomLegacyLearningRuleRegistry.inc
    CustomDenseNeuronRegistry.inc
    CustomDenseLearningRegistry.inc
    <GeneratedModel>.h
    <GeneratedModel>Device.cuh
  src/
    GeneratedCatalogEntries.cpp
    <GeneratedModel>.cpp
    <GeneratedModel>GPU_Interface.cu
  python/neuronbridge/generated/
    custom_models.py
    custom_models.pyi
    custom_model_schema.json
  tests/model_codegen/
  manifest.json
```

生成目录必须位于 build tree，禁止把 generated C++/CUDA 直接写回源码树。

## 4. W1：Catalog 注册与 Factory 创建

### 4.1 Neuron Catalog

**修改：**

```text
source_file/shared/include/neuron_model/NeuronModelCatalog.h
source_file/shared/src/neuron_model/NeuronModelCatalog.cpp
```

新增受控内部注册接口：

```cpp
void RegisterGeneratedEntries(
    const std::vector<NeuronModelCatalogEntry>& entries);
```

实现顺序必须是：先初始化手写 builtin entries，再显式调用生成函数 `RegisterGeneratedNeuronCatalogEntries(&entries_)`，随后校验 canonical name、alias、backend implementation name 冲突。构造完成后 Catalog 只读；不使用 C++ 全局静态 constructor 注册。

新增稳定声明壳：

```text
source_file/shared/include/neuron_model/GeneratedNeuronCatalog.h
```

生成器提供 `GeneratedCatalogEntries.cpp` 实现。没有 custom spec 时也必须生成空实现，保证 CMake 配置稳定。

### 4.2 Learning Rule Catalog

**修改：**

```text
source_file/shared/include/learning_rule/LearningRuleCatalog.h
source_file/shared/src/learning_rule/LearningRuleCatalog.cpp   # 新增
```

当前 `LearningRuleCatalog` 是 header-only。为支持生成条目，应将构造函数和 `Instance()` 移入 `.cpp`，并增加同样的 `RegisterGeneratedEntries(...)`。

`LearningRuleCatalogEntry` 应扩展为包含 backend bindings 和 `public_user_model`；新增 `LearningRuleBackend::LegacyCpu`、`LearningRuleBackend::DenseGpu`。旧的 `IsDenseGpuSupported()` 作为兼容包装保留，避免破坏既有调用者。

### 4.3 主网 neuron Factory

**修改：**

```text
source_file/legacy/source_file_realtime_v1_async/ModelFactory/NeuronModelFactory.cpp
source_file/legacy/source_file_realtime_v1_async/ModelFactory/NeuronModelFactory.h
```

在 `BuildRegistry()` 中保留 `LegacyNeuronModelList.inc`，然后追加生成条目：

```cpp
#define NPGR_CUSTOM_LEGACY_NEURON(implementation_name, class_name) \
    registry[implementation_name] = &CreateTimedModel<class_name>;
#include "neuronbridge_codegen/CustomLegacyNeuronRegistry.inc"
#undef NPGR_CUSTOM_LEGACY_NEURON
```

新增 Catalog/Factory 一致性验证，确保每个 `LegacyCpu` supported binding 都可由 registry 构造。

### 4.4 主网 learning rule Factory

**修改：**

```text
source_file/legacy/source_file_realtime_v1_async/ModelFactory/LearningRuleModelFactory.h
source_file/legacy/source_file_realtime_v1_async/ModelFactory/LearningRuleModelFactory.cpp   # 新增
```

将当前 if/else 工厂迁移为 registry，但保留 `createLearningRuleModel(...)` 签名。先注册手写 STDP/R-STDP/Cerebellar 等模型，再 include `CustomLegacyLearningRuleRegistry.inc`。rule 名称先经 `LearningRuleCatalog` resolve，再按 implementation name 创建。

### 4.5 Dense factories

**修改：**

```text
source_file/gpu/include/dense_subnetwork/model/DenseNeuronModelList.inc
source_file/gpu/include/dense_subnetwork/learning/DenseLearningModelList.inc
source_file/gpu/src/dense_subnetwork/model/DenseNeuronModelFactory.cpp
source_file/gpu/src/dense_subnetwork/learning/DenseLearningRuleFactory.cpp
```

保留 builtin list，再追加生成 list。生成器负责分配稳定、不复用的 Dense factory id，并记录到 manifest。验证必须满足：

```text
Catalog DenseGpu supported
<=> Dense host model 已注册
<=> device update entry 已注册
<=> field slot 完整绑定
```

## 5. W2：主网 CPU custom neuron

### 5.1 新增稳定基类

```text
source_file/legacy/source_file_realtime_v1_async/NeuralModel/inc/Custom/CustomEquationDescriptor.h
source_file/legacy/source_file_realtime_v1_async/NeuralModel/inc/Custom/CustomTimeDrivenNeuronModel.h
source_file/legacy/source_file_realtime_v1_async/NeuralModel/src/Custom/CustomTimeDrivenNeuronModel.cpp
```

`CustomEquationDescriptor` 仅保存初始化所需 state、parameter、input 和 slot metadata。`CustomTimeDrivenNeuronModel` 负责 `InitStateVector`、`InitState`、`ProcessSpike`、`ProcessCurrent`、`InitializeInputCurrentSynapseStructure`、`CheckType` 和通用参数读取。禁止在该基类定义 `virtual EvaluateDerivatives(...)`。

### 5.2 `ProcessSpike` 生成 contract

每个 spec 必须有 `synaptic_inputs`。对每个 `Interconnections::type`，生成子类提供静态、可内联的到达处理逻辑：

```cpp
InternalSpike* ProcessSpike(Interconnections* inter, int arrival_time) override;
```

首版支持 `conductance_jump`、`current_accumulator`、`state_assignment`。不得在 `ProcessSpike` 中直接调用 learning rule；现有 `PropogatedSpike` 在同一抵达事件中负责 pre/trigger learning。

### 5.3 具体生成 neuron

生成 `<Model>.h/.cpp` 必须继承 `CustomTimeDrivenNeuronModel`，定义 constexpr slots 和 `N_DifferentialStates`，并生成 `SetParameters`、`UpdateState`、`CaculateDifferentialEquation`、`CaculateTimeDependentEquation`、`CaculateSpike`。`CaculateSpike` 使用 `previous_v/current_v` 实现 `upward_crossing`，按 spec 固定顺序执行 reset 与 refractory assignment。

### 5.4 积分器接入

**不修改：**

```text
source_file/legacy/source_file_realtime_v1_async/Intergration/inc/FixStep/ForwardEulerMethod.h
```

生成模型必须实例化：

```cpp
ForwardEulerMethod<GeneratedConcreteModel>
```

不得使用 `ForwardEulerMethod<CustomTimeDrivenNeuronModel>`，以避免每个 neuron derivative 退化为通用基类调用。

### 5.5 主网 GPU custom neuron

后续子阶段新增：

```text
NeuralModel/inc/TimeDrivenGPU/Custom/CustomTimeDrivenNeuronModelGPU.cuh
NeuralModel/inc/TimeDrivenGPU/Custom/CustomTimeDrivenNeuronModelGPU_Interface.cuh
NeuralModel/src/TimeDrivenGPU/Custom/CustomTimeDrivenNeuronModelGPU_Interface.cu
```

每个模型生成 host interface、device state update 和 kernel wrapper。只有 CPU model 已完成 baseline 的模型才能启用主网 GPU emitter。

## 6. W3：主网 custom learning rule

### 6.1 新增稳定基类和状态

```text
source_file/legacy/source_file_realtime_v1_async/LearningRule/inc/Custom/CustomLearningRuleModel.h
source_file/legacy/source_file_realtime_v1_async/LearningRule/inc/Custom/CustomEquationSynapseState.h
source_file/legacy/source_file_realtime_v1_async/LearningRule/src/Custom/CustomLearningRuleModel.cpp
source_file/legacy/source_file_realtime_v1_async/LearningRule/src/Custom/CustomEquationSynapseState.cpp
```

`CustomEquationSynapseState` 继续使用 `StateValue[connection_id * state_count + slot]` 和 `LastUpdate[connection_id]`。禁止在 pre/post/trigger 热路径按 state name 查询 `std::map`。

### 6.2 生成 rule 接口

生成 `<Rule>.h/.cpp` 必须实现：

```cpp
InitState(...)
ApplyPreSynaticSpike(...)
ApplyPostSynaticSpike(...)
ImplementPostSynaptic()
ImplementTriggerSynaptic()
GetParameters()
```

`on_pre/on_post/on_trigger` 的固定顺序是：更新 event-driven trace 到 event time，按 spec 顺序执行 assignment，更新 weight，应用 clamp policy。规则不得绕过 `Interconnections::weight`、事件队列或既有 post/trigger routing。

## 7. W4：Dense custom neuron

### 7.1 新增稳定扩展壳

```text
source_file/gpu/include/dense_subnetwork/model/CustomDenseNeuronModel.h
source_file/gpu/include/dense_subnetwork/model/CustomDenseNeuronDeviceRuntime.cuh
```

生成 Dense host model 继承 `DenseNeuronModelBase`，负责 fields、field slots、initial values、derived fields、reset、`InputChannels()` 与 `SpikeEffects()`。

### 7.2 输入和输出映射

主网 `ProcessSpike` 的 Dense lowering 必须是：

```text
spec synaptic_inputs
-> DenseSpikeEffectBinding(synapse_type, PendingChannel, scale)
-> DrainIsetToPendingKernel
-> pending_channels[channel][target] += weight * scale
-> generated device update ReadAndClearPendingChannel
```

生成 device update 必须写 `did_fire`、执行 reset/refractory，并按现有 Dense firing-table 协议导出 firing id；不得建立 per-synapse device object。

### 7.3 CUDA dispatch

生成器追加 `CustomDenseNeuronRegistry.inc`，提供 factory model id、host class 和 device update entry，沿用现有 `ResolveDenseDeviceModelUpdate(...)` 调度模式。不得在 kernel 内解析方程、字符串或 AST。

## 8. W5：Dense custom learning rule

### 8.1 新增稳定扩展壳

```text
source_file/gpu/include/dense_subnetwork/learning/CustomDenseLearningRuleModel.h
source_file/gpu/include/dense_subnetwork/learning/CustomDenseLearningDeviceRuntime.cuh
```

生成 host rule 继承 `DenseLearningRuleModelBase`，声明 `Flags`、`RuleFields`、`SynapseFields`、`FieldSlots`、`FillRuleFields`、`FillSynapseFields` 和 `SpikeBufferWindowMs`。

### 8.2 事件 hook 生成

对 spec 中存在的 block 生成对应 device entry：

```text
on_pre     -> Apply<Rule>PreDeviceEntry
on_post    -> Apply<Rule>PostDeviceEntry
on_trigger -> Apply<Rule>TriggerDeviceEntry
```

不存在的事件使用现有 no-op wrapper。每个 entry 仅做数组读写、数学表达式和权重裁剪。

### 8.3 调度与路由要求

- pre 由 Iset/delay 抵达事件触发。
- post 由 `current_did_fire[post_neuron]` 触发。
- trigger 复用 `DenseLearningTriggerRouteTable`。
- 需要历史 spike 的 rule 必须声明 `SpikeBufferWindowMs` 并满足 ring-buffer 访问约束。

## 9. W6：官方 wheel、Python 与发布

### 9.1 pybind 绑定

**修改：**

```text
python/bindings/neuronbridge_core.cpp
python/src/neuronbridge/__init__.py
python/src/neuronbridge/<新增 models 模块>.py
```

新增只读查询 API：

```text
available_neuron_models(include_experimental=False, backend=None)
available_learning_rules(include_experimental=False, backend=None)
describe_model(name)
describe_learning_rule(name)
```

查询结果来自 C++ Catalog；Python 不维护第二份名称真相。

### 9.2 生成 Python facade

生成 `custom_models.py`、`custom_models.pyi`、`custom_model_schema.json`。facade 只提供参数校验与用户友好构造器，将名称和参数送入既有网络描述路径；不得逐步解释方程或参与 simulation step。

### 9.3 wheel 内容

官方构建必须产出包含 native extension、生成模型 C++/CUDA 实现、schema、manifest、所需 runtime DLL/so 和许可证信息的 wheel。用户安装官方 wheel 后不得需要 C++ compiler、CUDA compiler 或 generator，也不能在安装环境中添加自定义 spec。GPU/Dense 官方模型在相应 emitter 与 CUDA baseline 完成前不得发布。

## 10. W7：CMake、测试与性能门禁

### 10.1 CMake

**修改：**

```text
CMakeLists.txt
cmake/NetworkReleaseShared.cmake
python/CMakeLists.txt
```

新增：

```cmake
option(NR_ENABLE_MODEL_CODEGEN "Generate and compile custom equation models" ON)
set(NR_MODEL_SPEC_ROOT "${NR_ROOT}/model_specs" CACHE PATH "Custom model specification root")
```

配置阶段生成空 registry stub；构建阶段以 `add_custom_command(OUTPUT ...)` 运行生成器；所有生成 `.cpp/.cu` 通过 `target_sources(...)` 显式加入目标。不得依赖 `file(GLOB_RECURSE ...)` 发现 build tree 文件。

依赖关系：

```text
model_codegen_generate
  -> nr_snn_core_gpuaware_cpp
  -> nr_snn_core_gpuaware_cuda
  -> nr_dense_runtime_gpuaware
  -> neuronbridge_core
```

### 10.2 单元与协议测试

新增：

```text
tests/model_codegen/test_catalog_registration.cpp
tests/model_codegen/test_catalog_factory_consistency.cpp
tests/model_codegen/test_custom_lif_cpu.cpp
tests/model_codegen/test_custom_lif_event_delivery.cpp
tests/model_codegen/test_custom_rstdp_event_trace.cpp
tests/model_codegen/test_custom_dense_lif_cuda.cpp
tests/model_codegen/test_custom_dense_rstdp_cuda.cpp
python/tests/test_custom_model_catalog.py
python/tests/test_custom_model_constructors.py
```

neuron 测试至少覆盖兴奋/抑制/current arrival、未知连接拒绝、ODE update、threshold crossing、reset、refractory 和延迟传播。learning rule 测试至少覆盖 pre、post、trigger、trace decay、weight clamp 和同时间步优先级。

### 10.3 baseline 与性能门禁

对于与手写模型等价的 custom variant，固定 seed、输入、dt、连接和初值，比较 spike time、state trace、weight trace 和最终权重。CPU/Dense 各自与同后端手写 baseline 对齐。

性能门禁：

```text
warm run 中零动态分配
无 parser/AST/Python 进入热路径
基准吞吐 >= 对应手写 baseline 的 98%
超过 2% 回归必须保留 profile 证据并获审批
```

## 11. 推荐实施顺序与提交粒度

### Phase A：维护者工具与基础设施

**进度（2026-09-10）：部分完成。** 已落地 W0 的标准库 parser/validator、确定性 manifest、空规格生成物和内部 `--check`；CMake 已将 `nr_model_codegen` 接入 `nr_snn_core_gpuaware_cpp`，并在 `NR_ENABLE_MODEL_CODEGEN=OFF` 时编译空注册回退实现。Neuron Catalog 已提供受校验的 generated entry 合并入口。主网 CPU 的 `CustomEquationDescriptor` 与 `CustomTimeDrivenNeuronModel` 已实现并编入核心：前者只保存静态 state/event layout，后者只处理 spike/current 输入、状态向量生命周期和 spike 标记，具体方程仍由后续生成的最终类与 `ForwardEulerMethod<ConcreteModel>` 承担。受限 Equation AST 已实现 ODE、标识符、数值、括号、一元符号和四则运算的解析及确定性 C++ 表达式发射；函数调用和未批准语法会在 spec validation 阶段拒绝。尚未完成 learning rule Catalog 拆分、Catalog/factory 一致性运行时测试，以及任何 custom neuron/rule emitter。

1. W0 parser、validator、空生成产物、manifest 和内部 `--check`。
2. W1 neuron Catalog 可组合注册。
3. W1 learning rule Catalog 从 header-only 拆出。
4. CMake target、空 stub、Catalog/factory consistency tests。

**完成条件：** 无 custom spec 时既有 build 与 Python import 不变；维护者可在不编译 runtime 的情况下验证和检查 spec；故意注册冲突时 configure/test 失败。

### Phase B：主网 CPU 首个模型

**更新（2026-09-10）：首个模型生成/注册/数值 baseline 已通过。** `lif_runtime.py` 已实现受控 LIF profile emitter，CMake 显式接入生成 class、descriptor、registry 和 Catalog。实际 Factory 构造测试通过；12 个场景、288000 次状态比较最大误差为 0，2197 次 spike 完全匹配。ON/OFF 核心构建通过，CTest 1/1 和生成器 unittest 9/9 通过。详见 `reports/model_codegen_lif_baseline_20260910.md`。上文“任何 custom emitter 尚未完成”为此前脚手架状态；学习规则 Catalog、通用 DSL、完整传播事件测试、性能和官方 wheel 验收仍待后续实施。

1. W2 `CustomTimeDrivenNeuronModel`。
2. 生成 `CustomLifConductanceV1`。
3. 生成主网 neuron registry、Catalog entry、CPU baseline。
4. 验证 `ProcessSpike` 输入映射和 `CaculateSpike` 输出映射。
5. W6 的最小官方 CPU wheel：在隔离 build tree 中构建，并在干净环境安装验证。

**完成条件：** custom LIF 可由名称构建、可接收突触输入、可产生并传播 spike，且与指定手写 baseline 对齐；维护者可构建官方 wheel，干净 Python 环境中的用户可安装并运行该 wheel。

### Phase C：主网学习规则

**更新（2026-09-11）：** 前置 W1 的 LearningRuleCatalog 已由 header-only 拆为 shared 实现，新增附加条目构造校验及 9 类名称冲突测试；保持四个手写条目与原有别名/Dense 列表。ON/OFF 核心构建及 CTest 2/2 通过。当前 singleton 尚未接入生成式 rule provider，Factory、custom rule base/state 和 R-STDP emitter 尚未完成。本批仅完成注册基础，不代表 Phase C 完成。下文同 tick 差异已由 scheduling 报告定位，不能再作为本阶段未开始的阻断原因。

**更新（2026-09-11，首个规则闭环）：** `CustomLearningRuleModel`、`CustomRStdpV1` spec/emitter、生成 synapse state、Catalog provider、Factory registry 和完整轨迹 fixture 已落地。直接事件测试的 48 场景、13152 事件、79200 行快照和 361800 次数值比较均为零误差；Simulation 的 6 场景、45000 样本和 467 spike 一致，权重上下界及 eligibility clear 均覆盖。codegen ON/OFF 核心构建、15 项 generator unittest、3 项 CTest 通过。Phase C 的首个 Legacy CPU R-STDP shadow 可判定完成，但 emitter 仍是批准的单一 profile，不应宣称任意学习规则、多规则、GPU/Dense 或 Python 发布能力已完成。

**进入后续发布验收前的新增事项（2026-09-10）：** 相同模型分组下的三组完整 Simulation 传播测试已通过（537 个事件匹配），积分计时未发现当前场景减速；但替换模型造成合并/拆分变化时观察到输出不一致，需先调查同 tick 调度。详见 `reports/model_codegen_simulation_20260910.md`。

1. W3 custom synapse state 与 custom learning base。
2. 生成 `CustomRStdpV1` 或 `CustomStdpV1`。
3. 验证 pre/post/trigger 全事件流和权重轨迹。

### Phase D：Dense 后端

**更新（2026-09-11，R-STDP shadow）：** 学习规则 emitter 已支持同批输出 `CustomRStdpV1` 和 `CustomRStdpPersistentV1`，并生成 Dense host wrapper 与 static model list。两个规则以稳定 model id 4/5 注册到 Dense factory/device switch；独立 MSVC/CUDA 12.6 构建中 NVCC 已编译新增 dispatch，真实 `dense_subnetwork_smoke` 通过，并与内建 Dense `R_STDP` 的逐 tick 权重采样一致。CPU 多规则 723600 次比较、Dense GPU smoke、codegen OFF 核心/Dense 构建均通过。此阶段只完成“同方程 profile 复用既有 CUDA device entry”的 shadow，不是 AST-to-CUDA emitter；Dense custom neuron、异构学习规则和生成 CUDA 数学函数仍待实现。

**更新（2026-09-11，受控 AST-to-CUDA）：** 两个批准的 R-STDP spec 现在各自生成独立 CUDA pre/post/trigger entry，事件算术直接来自受限表达式 AST，generated device switch 不再复用手写 R-STDP entry。独立真实 GPU fixture 导出实际事件流和权重轨迹，并向生成 CPU 规则重放同一事件顺序；22 个跨后端权重样本最大误差 `1.90734863e-06`，阈值 `5e-6`，CTest 4/4 与 generator unittest 17/17 通过。W5 对当前批准 profile 判定完成；任意方程 CUDA lowering、Dense custom neuron 和第二种异构学习方程仍未完成。

**更新（2026-09-11，Dense LIF + 异构 Pair-STDP）：** W4 首个完整实现已落地：`CustomLifConductanceV1` 的 Dense host fields、input/spike mapping、reset/derived fields 与 CUDA update 均由相同 spec/ODE AST 生成，160 个 GPU 样本和 20 个 spike 对手写 Dense LIF 最大误差为 0。W5 新增 `CustomPairStdpV1`，采用二状态 `pre_trace/post_trace`、无 eligibility/trigger、pre 立即 LTD、post 立即 LTP；17 个 CPU/GPU 权重样本、8 个变化步，对手写 STDP及跨后端最大误差均为 0。异构布局同时暴露并修复 Dense learning field span 未按 factory id 排序的问题。generator unittest 20/20、CTest 6/6、codegen OFF 核心/Dense 构建通过。W4 与第二种异构 W5 profile 可判定完成；任意方程/分支/函数 CUDA lowering和正式吞吐门禁仍未完成。

1. W4 Dense custom LIF host metadata 与 CUDA update。
2. W5 Dense custom rule CUDA hooks。
3. 主网-Dense boundary、input pending channel 和 output spike 路径测试。

### Phase E：Python 与发布

**更新（2026-09-11）：** 分组差异调查已完成，实际差异仅为测试场景的同刻记录排序；六组 Simulation 事件和状态对比均通过。候选 wheel 已构建并通过独立 Python 3.12 venv 的隔离导入和 generated LIF 仿真验收，193 个网络事件匹配。详见 `reports/model_codegen_scheduling_20260911.md` 和 `reports/model_codegen_wheel_20260911.md`。图形依赖、完整离线包及独立机器测试仍待执行，尚未公开发布。

**更新（2026-09-11，CUDA wheel）：** wheel 验收脚本默认运行 installed-wheel CUDA smoke。最新 cp312 wheel 在隔离 venv 中成功加载 native extension，并通过公开 API 为两个生成 R-STDP 规则初始化和运行真实 Dense CUDA backend；CUDA runtime 报告已持久化。wheel 内已包含 `cudart64_12.dll` 及当前非系统运行 DLL。底层独立 fixture 已验证生成 CUDA 权重更新；Python 黑盒 trigger 拓扑尚未形成可观察的权重变化，故 wheel 端暂不把权重变化作为通过条件。完整离线 bundle 和无开发工具独立机器仍待验收。

**更新（2026-09-11，wheel 权重验收）：** wheel CUDA smoke 使用现有 `Simulation.get_connection_weight()` 做单连接测试断言，并以一个二神经元生成 Dense LIF 层运行三种生成规则。`CustomRStdpV1`、`CustomRStdpPersistentV1`、`CustomPairStdpV1` 均产生发放与可观察权重变化，最终权重分别为 `13.3458414`、`15.5036612`、`15.1958809`（初值均为 8.0）。候选 cp312 wheel SHA-256 为 `3045B7C6A0D82CB2405E57160D58AA0DCDF75CFD3F98C0F7E9CE6F79D62E9783`。不新增 Dense 实时权重批量访问器；正式用户输出、baseline 留档和离线分析使用 `Simulation.save_weights()` 生成的权重文件。独立无开发工具机器与完整离线 bundle 验收仍待完成。

**更新（2026-09-12，模型级硬编码消除）：** emitter 已从固定 LIF/STDP 字段表升级为 parameter schema + neuron/learning-rule IR。CPU 状态槽、状态更新、初值、输入、阈值、复位和学习事件均由 spec/AST 生成；Dense 使用显式 capability role map，不依赖 concrete 模型名。Dense LIF 的 comparison/reset 以及 Dense learning 的公开参数 alias/default 已改为 spec 驱动，避免 CPU/Dense 静默分叉。generator unittest `29/29`、CTest `6/6`、codegen ON/OFF 构建、Python 3.12 wheel CPU smoke 和三规则 CUDA runtime smoke 均通过。固定 Dense field slots/调度步骤仍作为 ABI capability 保留，它们不是模型级硬编码；任意 Dense 状态布局、函数和分支仍须后续新增受控 capability/AST primitive。详见 `reports/model_codegen_hardcoding_removal_20260912.md`。

1. W6 Catalog 查询、facade、stub typing。
2. Wheel 内 manifest/schema/native artifact 完整性测试。
3. 无编译器干净 Python 环境安装并运行发布 wheel。

## 12. 首次实现批准范围

当前批准的第一批代码变更限定为：

```text
W0 parser/validator/empty generator/internal validation
W1 Catalog 注册基础设施
CMake generation skeleton
Generator 协议测试与 CMake on/off 编译验证
```

以下项目明确延后，不能与首次实现混合提交：

```text
主网 GPU emitter
Dense CUDA emitter
custom learning rule runtime
GPU/Dense 官方 wheel 发布
Python facade 自动生成
公开稳定 API
```

下一批审批包才应验证生成定义、Catalog 可见性、Factory 创建、`ProcessSpike` 输入和 `CaculateSpike` 输出五个基础闭环。
