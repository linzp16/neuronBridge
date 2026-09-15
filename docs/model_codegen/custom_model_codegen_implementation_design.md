# NeuronBridge 自定义方程模型：具体实现设计

**状态：实施设计，尚未改动 runtime。**
**目标：让 `.nbmodel.json` 中的确定方程生成主网/Dense neuron 与 learning rule，同时不降低现有热路径性能。**

相关格式定义见 [用户自定义方程格式规范](neuronbridge_custom_equation_format.md)。逐文件的实施顺序、Catalog/Factory 注册边界和验收条件见[详细代码修改清单](custom_model_codegen_change_checklist.md)。

## 1. 不可妥协的性能规则

解析 JSON、方程字符串、单位、名称、参数别名和模型选择只能发生在 configure/build/init 阶段。每次 neuron 或 synapse 更新中禁止出现：

```text
JSON/字符串解析
std::map / unordered_map / std::string 查找
AST 遍历或解释执行
std::function、lambda type erasure
per-neuron virtual derivative dispatch
动态内存分配、锁或 Python 回调
```

生成后的连续方程必须与现有手写模型同样成为编译器可见的 C++/CUDA 表达式。目标热路径如下：

```text
主网 CPU: ForwardEulerMethod<GeneratedModel>::CaculateIncreament
             -> GeneratedModel::CaculateDifferentialEquation (static type, inlineable)

Dense GPU: unified kernel -> factory_model_id switch
             -> GeneratedUpdateDeviceEntry (static __device__ inline expression)

主网 rule: event dispatch -> GeneratedCustomRule::ApplyPre/Post/Trigger
             -> direct SynapseState array access and inline equation

Dense rule: pre/post/trigger switch -> generated __device__ inline entry
```

`Custom*` base class提供状态管理和构建期契约，**不**在每个 state derivative 上做虚分派。

## 2. 新增代码边界

### 2.1 手写且长期维护的通用 runtime

```text
source_file/legacy/source_file_realtime_v1_async/
  NeuralModel/inc/Custom/CustomEquationDescriptor.h
  NeuralModel/inc/Custom/CustomTimeDrivenNeuronModel.h
  NeuralModel/src/Custom/CustomTimeDrivenNeuronModel.cpp
  NeuralModel/inc/Custom/CustomForwardEulerMethod.h
  NeuralModel/inc/TimeDrivenGPU/Custom/CustomTimeDrivenNeuronModelGPU.cuh
  NeuralModel/inc/TimeDrivenGPU/Custom/CustomTimeDrivenNeuronModelGPU_Interface.cuh
  NeuralModel/src/TimeDrivenGPU/Custom/CustomTimeDrivenNeuronModelGPU_Interface.cu
  LearningRule/inc/Custom/CustomLearningRuleModel.h
  LearningRule/inc/Custom/CustomEquationSynapseState.h
  LearningRule/src/Custom/CustomLearningRuleModel.cpp
  LearningRule/src/Custom/CustomEquationSynapseState.cpp
  ModelFactory/CustomModelRegistry.h
  ModelFactory/CustomModelRegistry.cpp

source_file/gpu/include/dense_subnetwork/
  model/CustomDenseNeuronModel.h
  model/CustomDenseNeuronDeviceRuntime.cuh
  learning/CustomDenseLearningRuleModel.h
  learning/CustomDenseLearningDeviceRuntime.cuh
```

这些文件只包含稳定的 storage、event contract、field access helper、registry 壳和数值 integration helper；它们不知道任何用户模型名称或方程文本。

### 2.2 构建期生成文件

```text
${build}/generated/model_codegen/
  include/neuronbridge_codegen/
    CustomModelCatalog.inc
    CustomLegacyNeuronRegistry.inc
    CustomLegacyLearningRuleRegistry.inc
    CustomDenseNeuronRegistry.inc
    CustomDenseLearningRegistry.inc
    CustomLifConductanceV1.h
    CustomLifConductanceV1Device.cuh
    CustomRStdpV1.h
    CustomRStdpV1Device.cuh
  src/
    CustomLifConductanceV1.cpp
    CustomLifConductanceV1GPU_Interface.cu
    CustomRStdpV1.cpp
  python/neuronbridge/generated/
    custom_models.py
    custom_models.pyi
    custom_model_schema.json
  tests/model_codegen/
    custom_lif_conductance_v1_reference.cpp
    custom_rstdp_v1_event_trace.cpp
    fixtures/*.json
  manifest.json
```

generated C++/CUDA 永不提交源码树。现有手写 LIF/STDP 文件、`LegacyNeuronModelList.inc`、`DenseNeuronModelList.inc`、`DenseLearningModelList.inc` 保持不变。

## 3. 主网 custom neuron runtime

### 3.1 描述符只在初始化期使用

```cpp
enum class CustomFieldRole : unsigned char { State, Parameter, Input, Derived };

struct CustomFieldDescriptor {
    const char* name;
    CustomFieldRole role;
    float default_value;
    unsigned char storage;  // Float32, Int32, UInt8; initial implementation is Float32.
};

struct CustomInputBinding {
    int connection_type;    // existing Interconnections::type
    int state_slot;         // target slot in Neuron_State_Vector
    bool clear_on_step;
};

struct CustomEquationDescriptor {
    const char* implementation_name;
    int state_count;
    int differential_state_count;
    int voltage_state_slot;
    const CustomFieldDescriptor* fields;
    int field_count;
    const CustomInputBinding* inputs;
    int input_count;
};
```

约束：所有连续 ODE state 必须排在 `Neuron_State_Vector` 的前 `differential_state_count` 个 slot。这样可以直接复用现有 `ForwardEulerMethod` 的 `for (j < N_DifferentialStates)` loop，不引入索引映射或额外间接访问。

### 3.2 通用基类接口

`CustomTimeDrivenNeuronModel` 继承现有 `TimeDrivenModel`，负责 descriptor 驱动的状态初始化、输入通道绑定、`CurrentSynapse` 生命周期、`ProcessSpike`、`ProcessCurrent`、`CheckType` 和通用参数读取。

它**不**定义 virtual `EvaluateDerivatives`。核心接口是给生成子类和模板 integrator 使用的普通成员/静态约定：

```cpp
class CustomTimeDrivenNeuronModel : public TimeDrivenModel {
public:
    explicit CustomTimeDrivenNeuronModel(const CustomEquationDescriptor& descriptor,
                                         int timestep_size);
    ~CustomTimeDrivenNeuronModel() override;

    void InitStateVector(int neuron_count, int gpu_index) override;
    Neuron_State_Vector* InitState() override;
    InternalSpike* ProcessSpike(Interconnections* inter, int time) override;
    void ProcessCurrent(Interconnections* inter, Neuron* target, float current) override;
    void InitializeInputCurrentSynapseStructure() override;
    void CheckType(Interconnections* inter) override;
    int getV_index() override;
    int get_NumberOfState() override;
    NeuronModelType getNeuronModelType() override;

protected:
    void ConfigureCommonParameters(const std::map<std::string, boost::any>& parameters);
    void AdvanceRefractory(float* state, int neuron_index, float dt_ms);
    void EmitSpike(int neuron_index);
    const CustomEquationDescriptor& descriptor() const;
};
```

生成模型覆盖 `UpdateState`，但其实现只在每个 population step 调一次既有 `IntegerationMethod`。每个 neuron 的导数调用仍通过具体模板类型完成。

### 3.3 生成的具体 neuron

对于 `CustomLifConductanceV1`，生成器输出：

```cpp
class CustomLifConductanceV1 final : public CustomTimeDrivenNeuronModel {
public:
    static constexpr int N_DifferentialStates = 3;
    static constexpr int kV = 0;
    static constexpr int kGExc = 1;
    static constexpr int kGInh = 2;

    explicit CustomLifConductanceV1(int timestep_size);
    void SetParameters(std::map<std::string, boost::any> parameters, float dt_ms);
    void UpdateState(int index, int time, Simulation* simulation) override;

    // Called from ForwardEulerMethod<CustomLifConductanceV1>; no virtual dispatch.
    void CaculateDifferentialEquation(float* state, float* derivative, int neuron_index);
    void CaculateTimeDependentEquation(float* state, int neuron_index, float dt_ms);
    void CaculateSpike(float previous_v, float* state, int neuron_index);
};
```

生成的方程体直接展开为：

```cpp
derivative[kV] = (v_rest - state[kV] +
                  state[kGExc] * (e_exc - state[kV]) +
                  state[kGInh] * (e_inh - state[kV]) +
                  state[kInputCurrent]) * inv_tau_m;
derivative[kGExc] = -state[kGExc] * inv_tau_exc;
derivative[kGInh] = -state[kGInh] * inv_tau_inh;
```

时间常数倒数、常量参数和 slot 编号在 `SetParameters`/initialization 时准备。方程循环中没有名称查找、单位对象或 AST。

### 3.4 关键的模板积分接入

不能把 generated model 交给 `ForwardEulerMethod<CustomTimeDrivenNeuronModel>`，否则 template 只看得到基类，必须通过虚函数或回调执行导数，造成热循环开销。

必须生成并实例化：

```cpp
this->integrationMethod =
    ForwardEulerMethod<CustomLifConductanceV1>::CreateIntegerationMethod(
        integration_parameters, this);
```

因此 `ForwardEulerMethod<CustomLifConductanceV1>::CaculateIncreament` 对 `CaculateDifferentialEquation`、`CaculateTimeDependentEquation`、`CaculateSpike` 的调用可内联，布局与现有手写 `TimeDrivenLIF_Exponential_double` 相同。

首版强制 `integration.method = forward_euler`。RK2/RK4 只有在对应 `Custom*` template integrator、C++ reference 和 CUDA emitter 都实现并通过同一 fixture 后才能开放。

### 3.5 主网 GPU

主网 GPU 生成物采用既有 `TimeDrivenNeuronModelGPU_Interface` 生命周期模式：生成一个具体 host interface `.cu`、一个具体 device model `.cuh` 和一个具体 update kernel wrapper。host virtual dispatch 只发生在模型初始化/launch 边界；每个 device thread 内的方程是静态 `__device__ inline` 函数。

```text
GeneratedCustomLifGPU_Interface::UpdateState
  -> generated launch wrapper
     -> GeneratedCustomLifGPU::UpdateOneNeuron
        -> direct float state-array loads/stores + expanded ODE
```

首版只允许与主网 CPU custom model 完全相同的 state slot order、Forward Euler、connection type binding 和 spike ordering。不能满足该 profile 的 model 只生成 CPU custom model 或转入 Dense backend；不生成“通用 GPU interpreter”。

## 4. 主网 custom learning rule runtime

### 4.1 通用 state 与事件分发

`CustomEquationSynapseState : public SynapseState` 使用现有 contiguous `StateValue[connection * NumberOfState + slot]` 作为 storage。它新增 generated descriptor 提供的 slot count、initial values、last-update policy；不使用 `std::map` 查询 state。

`CustomLearningRuleModel : public LearningRule` 负责：

```text
InitState -> 一次分配 CustomEquationSynapseState
ApplyPreSynaticSpike -> 更新 trace 到 event time -> generated pre assignments
ApplyPostSynaticSpike -> 更新相关 connection trace -> generated post assignments
trigger dispatch -> generated trigger assignments
weight bounds / connection index validation / state cleanup
```

每条 rule instance 在初始化期接收一个 generated descriptor；每个 event 调用只接触 connection、state array、时间步和常量参数。

### 4.2 生成的 custom R-STDP

```cpp
class CustomRStdpV1 final : public CustomLearningRuleModel {
public:
    void InitState(int connection_count, int neuron_count, float dt_ms) override;
    void ApplyPreSynaticSpike(Interconnections*, int spike_time, Simulation*) override;
    void ApplyPostSynaticSpike(Neuron*, int spike_time, Simulation*) override;
    void ApplyTriggerSynapticSpike(Interconnections*, int spike_time, Simulation*);
};
```

生成的线性 trace 衰减使用预计算 `inv_tau` 和 `expf(-delta_t * inv_tau)`，并直接读写 `StateValue`。如果 `event_driven` ODE 不满足一维线性可解析条件，生成器拒绝 event-driven flag，改由固定步进 state update；不会在运行期尝试符号求解。

现有 `STDP`、`R_STDP`、`CerebullarLearningRule` 继续由当前 `LearningRuleModelFactory` 默认创建。`CustomRStdpV1` 使用新内部名注册；factory 的 name lookup 仅在 network build 时发生，不属于 spike event hot path。

## 5. Dense custom neuron 与 rule

### 5.1 Host metadata

`CustomDenseNeuronModel` 与 `CustomDenseLearningRuleModel` 只在 build/finalize 阶段使用 fields、slots、parameter aliases 和 input schema。它们直接复用当前 `DenseNeuronHostFieldTable`、`DenseLearningHostFieldTable`、`DenseFieldSchema` 和 slot compaction；field name -> id 的 map 查找在构建期完成。

生成器为每个 custom Dense model 输出固定 enum slots、`Fields()`、`FieldSlots()`、`InputChannels()`、`SpikeEffects()`、`FillInitialFieldValues()` 与 `BuildDerivedFields()`。由现有 factory 将字段名压缩为 `field_ids[]`。

### 5.2 CUDA entry

生成器为每个 custom Dense neuron 输出：

```cpp
__device__ inline unsigned char UpdateCustomLifConductanceV1DeviceEntry(
    DenseNeuronDeviceFieldTable fields,
    const int* __restrict__ field_ids,
    int neuron_index,
    int time_step,
    DensePendingChannelDeviceView pending,
    float dt_ms,
    int* fired_field_id) {
    // direct DeviceFloatField + fixed field_ids[kSlot] + expanded equations
}
```

它被加入独立 generated variant registry。现有 `DenseNeuronDeviceUpdate.cuh` 的 switch 壳不变；每个 neuron 现有的一次 `factory_model_id` switch 仍然存在，但没有新增 dispatch 层、函数表或解析成本。

Dense rule 以同样方式生成 pre/post/trigger static entry。weight、rule field、synapse field 和 spike history 均用现有 `DenseLearningDeviceFieldTable`/`DenseDeviceLearningSpikeBufferTable`。不引入 per-synapse object 或 new/delete。

## 6. Registry、catalog 与 Python

### 6.1 Registry 接入

新增四个 generated fragment，由稳定的手写 bridge include：

```text
CustomLegacyNeuronRegistry.inc
CustomLegacyLearningRuleRegistry.inc
CustomDenseNeuronRegistry.inc
CustomDenseLearningRegistry.inc
```

每项都是 compile-time class/entry symbol。现有手写 list 和默认名字不改；custom item 只以 `implementation_name` 进入 registry。factory 的查找发生在 `Simulation`/`Network` 初始化，运行期间只保存构造结果和 integer id。

### 6.2 Python

不新增“每个模型一个 pybind 通道”。现有：

```python
NeuronLayer(model_name, count, parameters)
LearningRule(rule_name, parameters)
```

仍是唯一 native construction 通道。generator 只生成：

```python
neuronbridge.generated.custom_lif_conductance_v1(...)
neuronbridge.generated.custom_rstdp_v1(...)
```

它们创建上述通用描述对象，检查参数、单位标记和 visibility；不执行方程。native extension 无需为单一模型增加新的 pybind 定义。

## 7. 生成器实现

```text
tools/model_codegen/
  generate.py            # CLI: validate/generate/check/emit-ir
  json_loader.py          # envelope, source locations and spec inventory
  equation_lexer.py       # tokenizes the equations field
  equation_parser.py      # ODE/subexpression/parameter/event grammar
  dimensions.py           # dimension vectors and compatibility checks
  dependency_graph.py     # topological ordering and event write checks
  equation_ir.py          # typed, backend-neutral Equation IR
  lower_legacy.py         # main CPU/GPU lowering
  lower_dense.py          # Dense field/slot lowering
  lower_learning.py       # main/Dense rule lowering
  emit_cpp.py
  emit_cuda.py
  emit_python.py
  emit_tests.py
  emit_manifest.py
  tests/
```

普通运行 wheel 不依赖 Brian2、SymPy、Jinja、Lark、JSON Schema 或 Python parsing。维护者开发环境和 CI 使用锁定的 parser/schema/template toolchain；这些依赖仅在内部 validate/build 阶段存在，绝不进入仿真热路径或发布 wheel 的运行依赖。

### 7.1 Generator command

```text
python tools/model_codegen/generate.py validate --spec-root model_specs
python tools/model_codegen/generate.py generate --spec-root model_specs --output-root <build>/generated/model_codegen
python tools/model_codegen/generate.py check --spec-root model_specs --output-root <build>/generated/model_codegen
```

`check` 必须验证：input digest、generator version、base contract digest、manifest、生成内容的确定性和 public schema snapshot。

## 8. CMake 具体接入

根 `CMakeLists.txt` 新增：

```cmake
option(NR_ENABLE_MODEL_CODEGEN "Generate custom equation model sources" ON)
option(NR_MODEL_CODEGEN_VERIFY "Verify generated custom model sources" ON)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)
```

新增 `cmake/NeuronBridgeModelCodegen.cmake`，定义 `nr_add_model_codegen()`：

1. 收集 `model_specs/**/*.nbmodel.json` 和 `tools/model_codegen/*.py`。
2. `add_custom_command(OUTPUT <all generated headers/sources/manifest>)` 运行 `generate.py generate`。
3. `add_custom_target(nr_model_codegen DEPENDS <outputs>)`。
4. 在 `nr_snn_core_gpuaware_cpp`、`nr_snn_core_gpuaware_cuda`、`nr_dense_runtime_gpuaware`、`neuronbridge_core` 和 custom model test target 上添加 `add_dependencies(... nr_model_codegen)`。
5. 用 `target_sources` 将 generated `.cpp` 加入 `nr_snn_core_gpuaware_cpp`，generated legacy GPU `.cu` 加入 `nr_snn_core_gpuaware_cuda`，generated Dense `.cu/.cpp` 加入 `nr_dense_runtime_gpuaware`。
6. 将 `${CMAKE_BINARY_DIR}/generated/model_codegen/include` 放在这些 target 的 `PRIVATE` include path 首位。

不能依赖 `file(GLOB_RECURSE NR_LEGACY_CPP_SOURCES ...)` 发现 build-directory generated source；必须用 `target_sources` 显式加入，确保 Visual Studio/Ninja 都能建立正确依赖。

`NR_ENABLE_MODEL_CODEGEN=ON` 必须独立于 `NR_ENABLE_PYTHON`。发布 wheel 和 offline bundle 只打包生成后的 native library、generated Python schema 和 manifest；最终用户不需要生成器。

## 9. 性能验收与回归门禁

### 9.1 静态门禁

- generated `.cpp/.cu` 不能 include parser、JSON、Python、`std::map`、`std::function`、`std::string` 或 filesystem header。
- generated device entry 不得包含 indirect function pointer、virtual call、dynamic allocation 或 field-name lookup。
- generated legacy equation slots 是 `constexpr int`；Dense slots 是 enum/`constexpr int`。
- generated manifest 记录 `hot_path_contract: static_direct`；CI grep/AST test 验证产物。

### 9.2 数值门禁

- Custom LIF CPU 与 generated reference 每步 state/spike 对齐。
- Custom LIF GPU 与 CPU 对齐指定 abs/rel tolerance，离散 spike/reset 顺序精确一致。
- Custom R-STDP 与 reference event trace 对齐：pre/post/trigger、weight、trace、eligibility、last-update。
- Dense custom model 与 CUDA reference 对齐，包含 input clear、refractory 和 32x32x8 规模输入。

### 9.3 吞吐门禁

每个 Release CI 记录固定机器/固定 build 的中位数与 p95：

```text
main custom neuron: 100k neurons, 10k steps, Forward Euler
main custom rule:   1M plastic synapses, fixed pre/post/trigger trace
dense custom:       1M neurons / 10M synapses where available
```

比较对象为同方程、同积分器、同输入 fixture 的手写 LIF/R-STDP 或基准 custom code。新增代码必须满足：

```text
CPU/GPU median throughput >= baseline * 0.98
GPU kernel launch count unchanged
allocation count after initialization = 0
```

若出现超过 2% 的稳定回归，PR 必须提供汇编/PTX/Nsight 或 profiler 证据并获得明确批准；默认不能合并。该阈值是噪声保护带，目标是无可测性能损失。

## 10. 实施顺序

1. 先实现 parser、AST、维度检查、manifest 和 reference emitter；不接 CMake runtime。
2. 新增 `CustomTimeDrivenNeuronModel`、`CustomForwardEulerMethod` 和 `CustomEquationDescriptor`，但不改变现有模型。
3. 生成并编译 CPU `CustomLifConductanceV1`；验证静态模板调用、数值和吞吐。
4. 接入主网 GPU custom LIF，完成 CPU/GPU baseline。
5. 新增 `CustomLearningRuleModel`、`CustomEquationSynapseState`，生成 `CustomRStdpV1`，完成 event trace 和吞吐验证。
6. 接入 `CustomDenseNeuronModel` 与 Dense CUDA entry；之后才接入 custom Dense learning。
7. 在 CPU custom model 纵向闭环中交付维护者 generator 验证、官方 wheel 构建与 wheel smoke；之后再扩展 Python convenience constructor、GPU/Dense emitter 与完整发布矩阵。

每一步只新增 custom variant。现有 LIF、STDP、R-STDP、Dense models、示例和默认 API 不修改行为。

## 11. 首次代码审批清单

第一批可审批代码应仅包含：

```text
CustomEquationDescriptor
CustomTimeDrivenNeuronModel (CPU only)
CustomForwardEulerMethod template
generator parser/AST/dimension unit tests
CustomLifConductanceV1 generated CPU source
CPU reference + throughput benchmark
CMake nr_model_codegen skeleton
```

不应在第一批中加入 GPU interface、Dense runtime、learning rule、pybind 变化或现有模型重构。CPU custom LIF 的数值与性能门禁通过后，再逐层增加后端。
