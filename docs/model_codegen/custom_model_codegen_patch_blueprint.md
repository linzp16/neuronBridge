# NeuronBridge 自定义方程模型：代码改动蓝图

**状态：实现前设计，非可直接合并补丁。**
**范围：把详细修改清单进一步落实到 C++ 签名、CMake target、生成文件 ABI、事件调用点与测试夹具。**

本文件只描述首个可审批实现包：维护者 generator 验证、Catalog/Factory 基础设施、generator skeleton、主网 CPU `CustomLifConductanceV1` 和官方 wheel 安装 smoke。主网 GPU、Dense custom model、custom learning rule 和 Python 自动 facade 保留为后续包。

前置文档：[详细代码修改清单](custom_model_codegen_change_checklist.md)。

## 1. 首包目录和文件图

```text
手写稳定源码
  source_file/shared/include/neuron_model/GeneratedNeuronCatalog.h       [新增]
  source_file/shared/include/neuron_model/NeuronModelCatalog.h          [修改]
  source_file/shared/src/neuron_model/NeuronModelCatalog.cpp            [修改]
  source_file/legacy/source_file_realtime_v1_async/
    NeuralModel/inc/Custom/CustomEquationDescriptor.h                    [新增]
    NeuralModel/inc/Custom/CustomTimeDrivenNeuronModel.h                 [新增]
    NeuralModel/src/Custom/CustomTimeDrivenNeuronModel.cpp               [新增]
    ModelFactory/NeuronModelFactory.cpp                                  [修改]
  cmake/NetworkReleaseShared.cmake                                       [修改]
  CMakeLists.txt                                                         [修改]

生成器源码
  tools/model_codegen/...                                                [新增]
  model_specs/custom_neurons/custom_lif_conductance_v1.nbmodel.json     [新增]

build tree 仅生成
  generated/model_codegen/include/neuronbridge_codegen/
    CustomLegacyNeuronRegistry.inc
    CustomNeuronCatalog.inc
    CustomLifConductanceV1.h
  generated/model_codegen/src/
    GeneratedNeuronCatalogEntries.cpp
    CustomLifConductanceV1.cpp
```

## 2. Catalog：精确修改蓝图

### 2.1 设计目的

`NeuronModelCatalog` 是名称语义层；它不创建对象。生成条目必须在 Catalog singleton 构造时一次性附加，随后冻结。Factory 才负责由 `implementation_name` 创建模型。

禁止的实现：

```text
Python import 时动态注册 C++ 模型
dlopen/plugin 方式加载不受版本控制的模型
全局静态对象的跨 translation unit 自注册
运行时读取 .nbmodel.json 后改变 Catalog
```

### 2.2 新增稳定声明壳

**新增文件：** `source_file/shared/include/neuron_model/GeneratedNeuronCatalog.h`

```cpp
#ifndef NPGR_GENERATED_NEURON_CATALOG_H
#define NPGR_GENERATED_NEURON_CATALOG_H

#include <vector>

namespace npgr {
struct NeuronModelCatalogEntry;

// GeneratedCatalogEntries.cpp always defines this symbol. With no custom
// specs it is an empty function, keeping every configuration linkable.
void RegisterGeneratedNeuronCatalogEntries(
    std::vector<NeuronModelCatalogEntry>* entries);
}  // namespace npgr

#endif
```

该文件是稳定 ABI 壳，不能由 generator 覆盖。

### 2.3 `NeuronModelCatalog.h` 拟增接口

在 private 区增加初始化期注册 API：

```cpp
class NeuronModelCatalog {
public:
    static NeuronModelCatalog& Instance();
    // Existing query APIs remain unchanged.

private:
    NeuronModelCatalog();
    void RegisterGeneratedEntries(
        const std::vector<NeuronModelCatalogEntry>& entries);
    void ValidateEntryOrThrow(const NeuronModelCatalogEntry& candidate) const;
    std::vector<NeuronModelCatalogEntry> entries_;
};
```

注册函数保持 private，避免任意 runtime 调用者改变 singleton 内容。

### 2.4 `NeuronModelCatalog.cpp` 构造顺序

新增 include：

```cpp
#include "neuron_model/GeneratedNeuronCatalog.h"
#include <stdexcept>
#include <unordered_set>
```

构造函数的末尾增加：

```cpp
NeuronModelCatalog::NeuronModelCatalog() {
    entries_ = {
        // Existing handwritten entries, byte-for-byte semantic equivalent.
    };
    std::vector<NeuronModelCatalogEntry> generated_entries;
    RegisterGeneratedNeuronCatalogEntries(&generated_entries);
    RegisterGeneratedEntries(generated_entries);
}
```

`ValidateEntryOrThrow` 必须检查：

```text
candidate.canonical_name 非空。
canonical name 不与现有 canonical/alias 冲突。
每个 alias 非空，且不相互冲突。
每种 NeuronBackend 至多出现一个 binding。
supported binding 必须拥有非空 implementation_name。
implementation_name 不得与任意同 backend 已注册 implementation_name 冲突。
至少一个 backend supported。
```

### 2.5 生成的 Catalog 实现格式

`GeneratedNeuronCatalogEntries.cpp` 只能生成静态 C++ 数据，例如：

```cpp
#include "neuron_model/GeneratedNeuronCatalog.h"
#include "neuron_model/NeuronModelCatalog.h"

namespace npgr {
void RegisterGeneratedNeuronCatalogEntries(
    std::vector<NeuronModelCatalogEntry>* entries) {
    if (entries == nullptr) return;
    entries->push_back({
        "CustomLifConductanceV1",
        {"custom_lif_conductance_v1"},
        {
            {NeuronBackend::LegacyCpu, "CustomLifConductanceV1", true, true},
            {NeuronBackend::LegacyGpu, "", false, false},
            {NeuronBackend::DenseGpu, "", false, false},
        },
        NeuronModelCategory::Core,
        NeuronModelRole::OrdinaryCore,
        false,
    });
}
}  // namespace npgr
```

首包中 `DenseGpu=false` 是刻意设计：生成器不得宣称未编译的后端可用。

## 3. Factory：精确修改蓝图

### 3.1 现有路径

当前 `NeuronModelFactory::createNeuronModel(...)` 已先调用 `NeuronModelCatalog::ResolveImplementationName(...)`。因此 custom model 不需要改变 `Network::CreateNeuronModel(...)`，只需让 Catalog 能解析名称、Factory registry 能创建类。

### 3.2 `NeuronModelFactory.cpp` 的拟修改

在现有 `BuildRegistry()` 的手写 list 后追加：

```cpp
#define NPGR_CUSTOM_LEGACY_NEURON(implementation_name, class_name) \
    registry[implementation_name] = &CreateTimedModel<class_name>;
#include "neuronbridge_codegen/CustomLegacyNeuronRegistry.inc"
#undef NPGR_CUSTOM_LEGACY_NEURON
```

首个生成 registry：

```cpp
NPGR_CUSTOM_LEGACY_NEURON("CustomLifConductanceV1", CustomLifConductanceV1)
```

`CustomLegacyNeuronRegistry.inc` 必须只含宏调用，不含 include、namespace、逻辑或名称解析。

### 3.3 Factory 校验

首包不扩大生产 API。Catalog/Factory 一致性测试遍历 `SupportedModelNames(LegacyCpu)`，以默认参数夹具实际调用 public `createNeuronModel(...)`。生成模型的参数夹具由 manifest 提供。

## 4. 主网 custom neuron 基类

### 4.1 `CustomEquationDescriptor.h`

首包 descriptor 限制为 float state、float parameter 和到达事件输入：

```cpp
namespace npgr {
enum class CustomInputDelivery : unsigned char {
    AddToState,
    AddToCurrentAccumulator,
};
struct CustomInputBinding {
    int connection_type;
    int target_state_slot;
    float scale;
    CustomInputDelivery delivery;
};
struct CustomEquationDescriptor {
    const char* implementation_name;
    int state_count;
    int differential_state_count;
    int voltage_state_slot;
    int refractory_state_slot;
    const CustomInputBinding* input_bindings;
    int input_binding_count;
};
}  // namespace npgr
```

不在 descriptor 中保存方程文本、AST、`std::string` 或参数 map。

### 4.2 `CustomTimeDrivenNeuronModel.h`

```cpp
class CustomTimeDrivenNeuronModel : public TimeDrivenModel {
public:
    explicit CustomTimeDrivenNeuronModel(
        const npgr::CustomEquationDescriptor& descriptor,
        int timestep_size);
    ~CustomTimeDrivenNeuronModel() override;
    void InitStateVector(int neuron_count, int gpu_index) override;
    Neuron_State_Vector* InitState() override;
    InternalSpike* ProcessSpike(Interconnections* inter, int arrival_time) override;
    void ProcessCurrent(Interconnections* inter, Neuron* target, float current) override;
    void InitializeInputCurrentSynapseStructure() override;
    void CheckType(Interconnections* inter) override;
    int getV_index() override;
    int get_NumberOfState() override;
    NeuronModelType getNeuronModelType() override;
protected:
    const npgr::CustomEquationDescriptor& descriptor() const;
    bool ApplyArrivalInput(Interconnections* inter);
    void MarkSpike(int neuron_index);
private:
    const npgr::CustomEquationDescriptor& descriptor_;
    CurrentSynapse* current_synapse_model_;
};
```

实际 include、`CurrentSynapse` 命名空间和 `NeuronModelType` 的具体值必须在实现前以编译器确认，不在生成器中硬编码。

### 4.3 基类 `ProcessSpike` 精确语义

```cpp
bool CustomTimeDrivenNeuronModel::ApplyArrivalInput(Interconnections* inter) {
    const int neuron_index = inter->TargetNeuronModelIndex;
    float* state = StateVector->GetNeuronState(neuron_index);
    for (int i = 0; i < descriptor_.input_binding_count; ++i) {
        const CustomInputBinding& binding = descriptor_.input_bindings[i];
        if (binding.connection_type != inter->type) continue;
        state[binding.target_state_slot] += inter->weight * binding.scale;
        return true;
    }
    return false;
}
```

首包的 `ProcessSpike` 在未声明输入 type 时抛出可诊断错误，成功时返回 `nullptr`。它仅改变目标 neuron 的输入状态；pre/trigger learning 仍由 `PropogatedSpike::ProcessEvent(...)` 在返回后调用，禁止重复调用 `LearningRule`。

`ProcessCurrent` 不能简单等同于普通 spike input。它必须复用现有 `CurrentSynapse` 语义：同一 current connection 更新最新值，多条 current connection 在更新步被求和。首包将 current input 限制为与 `TimeDrivenLIF_Exponential_double` 同构的行为，并由 dedicated test 锁定。

## 5. 生成的 `CustomLifConductanceV1`

### 5.1 生成 header 的目标形态

```cpp
class CustomLifConductanceV1 final : public CustomTimeDrivenNeuronModel {
public:
    static constexpr int kV = 0;
    static constexpr int kGExc = 1;
    static constexpr int kGInh = 2;
    static constexpr int kInputCurrent = 3;
    static constexpr int kRefractoryRemaining = 4;
    static constexpr int N_DifferentialStates = 3;
    explicit CustomLifConductanceV1(int timestep_size);
    void SetParameters(std::map<std::string, boost::any> parameters, float base_dt_ms);
    void UpdateState(int index, int time, Simulation* simulation) override;
    void CaculateDifferentialEquation(float* state, float* derivative, int neuron_index);
    void CaculateTimeDependentEquation(float* state, int neuron_index, float dt_ms);
    void CaculateSpike(float previous_v, float* state, int neuron_index);
    std::map<std::string, boost::any> getParameters() override;
private:
    float v_rest_, v_reset_, v_threshold_, e_exc_, e_inh_;
    float inv_tau_m_, inv_tau_exc_, inv_tau_inh_, refractory_ms_;
};
```

生成器必须保证所有微分状态位于 `[0, N_DifferentialStates)`，从而直接符合现有 `ForwardEulerMethod` 的增量循环。

### 5.2 生成的积分接入

```cpp
void CustomLifConductanceV1::UpdateState(int, int time, Simulation* simulation) {
    integrationMethod->CaculateIncreament(simulation, time);
}

integrationMethod =
    ForwardEulerMethod<CustomLifConductanceV1>::CreateIntegerationMethod(
        integration_parameters, this);
```

这里有意使用具体子类。现有 `ForwardEulerMethod` 会调用同名的 `int neuron_index` overload；它不是 `TimeDrivenModel` 的 `float dt` 虚函数 override，生成器必须维持当前手写模型的实际模板调用约定。

### 5.3 生成的状态更新与输出 spike

```cpp
void CustomLifConductanceV1::CaculateDifferentialEquation(
    float* state, float* derivative, int) {
    derivative[kV] = (v_rest_ - state[kV]
        + state[kGExc] * (e_exc_ - state[kV])
        + state[kGInh] * (e_inh_ - state[kV])
        + state[kInputCurrent]) * inv_tau_m_;
    derivative[kGExc] = -state[kGExc] * inv_tau_exc_;
    derivative[kGInh] = -state[kGInh] * inv_tau_inh_;
}

void CustomLifConductanceV1::CaculateSpike(
    float previous_v, float* state, int neuron_index) {
    if (state[kRefractoryRemaining] > 0.0f) return;
    if (previous_v < v_threshold_ && state[kV] >= v_threshold_) {
        MarkSpike(neuron_index);
        state[kV] = v_reset_;
        state[kRefractoryRemaining] = refractory_ms_;
    }
}
```

`CaculateTimeDependentEquation` 应按 `dt_ms` 衰减/清理每步输入并推进 refractory；具体顺序必须用 reference baseline 固定，不能凭直觉修改。

## 6. CMake：精确 target 结构

### 6.1 顶层开关

在 `CMakeLists.txt` 的现有 Python 开关后加入：

```cmake
option(NR_ENABLE_MODEL_CODEGEN
  "Generate and compile NeuronBridge custom equation models" ON)
set(NR_MODEL_SPEC_ROOT "${NR_ROOT}/model_specs" CACHE PATH
  "Root directory of NeuronBridge custom model specifications")
```

### 6.2 `NetworkReleaseShared.cmake` 新函数

新增 `nr_configure_model_codegen()`，输出 `NR_CODEGEN_OUTPUT_ROOT`、`NR_CODEGEN_INCLUDE_ROOT`、`NR_CODEGEN_CPP_SOURCES`、`NR_CODEGEN_CUDA_SOURCES`、`NR_CODEGEN_PYTHON_ROOT`、`NR_CODEGEN_MANIFEST`。

```cmake
add_custom_command(
  OUTPUT ${NR_CODEGEN_STAMP} ${NR_CODEGEN_CPP_SOURCES}
  COMMAND ${NR_CODEGEN_PYTHON_EXECUTABLE} -m tools.model_codegen.cli
          --spec-root ${NR_MODEL_SPEC_ROOT}
          --output-root ${NR_CODEGEN_OUTPUT_ROOT}
          --manifest ${NR_CODEGEN_MANIFEST}
  DEPENDS ${NR_CODEGEN_GENERATOR_SOURCES} ${NR_CODEGEN_SPEC_FILES}
  WORKING_DIRECTORY ${NR_ROOT}
  VERBATIM)
add_custom_target(model_codegen_generate DEPENDS ${NR_CODEGEN_STAMP})
```

生成器必须稳定输出预声明文件列表，或输出唯一 stamp 并配置 empty stubs；不能让 CMake 依赖运行时才出现、配置期未知的 source list。

### 6.3 目标接入

```cmake
add_dependencies(nr_snn_core_gpuaware_cpp model_codegen_generate)
target_sources(nr_snn_core_gpuaware_cpp PRIVATE ${NR_CODEGEN_CPP_SOURCES})
target_include_directories(nr_snn_core_gpuaware_cpp PRIVATE ${NR_CODEGEN_INCLUDE_ROOT})
```

后续有 CUDA emitter 后才追加到 `nr_snn_core_gpuaware_cuda`。不得将 build-tree generated files 混入 `NR_LEGACY_CPP_SOURCES` 的 glob。

## 7. 维护者构建与官方 wheel

维护者在源码工作树中运行 generator，再由 CMake target 构建 official wheel；不提供 `neuronbridge model` 终端用户命令、`model-build` extra、用户 spec 导入或用户自建 custom wheel。wheel manifest 至少含 `core_version`、`core_abi_version`、Python ABI、平台、generator version、spec hash、supported backends 和 artifact hashes。

### 7.1 运行用户 smoke

在新建的干净 Python 3.12 环境执行：

```text
pip install <official generated-model wheel>
python -c "import neuronbridge"
<minimal custom LIF simulation and Catalog query>
```

该 smoke 不允许调用 generator，也不允许依赖维护者机器的 CMake、源码目录或编译器路径。

## 8. 首包测试：逐例说明

### 8.1 Catalog 单元测试

`tests/model_codegen/test_catalog_registration.cpp`：

```text
Resolve("CustomLifConductanceV1") 返回 custom entry。
Resolve("custom_lif_conductance_v1") 解析为同一 canonical name。
LegacyCpu 为 supported，LegacyGpu/DenseGpu 为 unsupported。
ResolveImplementationName(... LegacyCpu) 返回 CustomLifConductanceV1。
PublicModelNames() 默认不包含 experimental custom model。
```

### 8.2 Factory 和输入事件测试

`tests/model_codegen/test_custom_lif_event_delivery.cpp`：

```text
用 custom canonical name 构建最小网络。
发送 excitatory arrival，断言 g_exc 增加 weight。
发送 inhibitory arrival，断言 g_inh 增加 weight。
发送未声明 type，断言建网或抵达时抛出可诊断错误。
验证 pre learning 不会由 ProcessSpike 重复执行。
```

### 8.3 输出 spike 测试

`tests/model_codegen/test_custom_lif_cpu.cpp`：

```text
设定可预测初始 v、输入和 dt。
验证阈值下方无 spike。
验证 previous_v < threshold 且 current_v >= threshold 时仅发放一次。
验证 reset 与 refractory state。
验证 refractory 期间持续高电压不重复 spike。
```

### 8.4 生成器负例

Python tests 至少拒绝：与 builtin canonical name 冲突、两个 spec 复用 implementation name、ODE state 不在前 `N_DifferentialStates` slots、输入引用不存在 state、spike block 写 parameter、声明 DenseGpu 但没有 CUDA emitter profile。

## 9. 后续包的精确接口预留

### 9.1 Learning Rule Catalog 和 Factory

后续将 `LearningRuleCatalog` 从 header-only 拆分，实现与 neuron Catalog 相同的 generated registration。`LearningRuleModelFactory` 从 header-only if/else 拆为 `.cpp` registry，但保留：

```cpp
static LearningRule* createLearningRuleModel(
    LearningRuleDescription description);
```

生成主网 rule 的注册宏：

```cpp
NPGR_CUSTOM_LEGACY_LEARNING_RULE("CustomRStdpV1", CustomRStdpV1)
```

### 9.2 Dense 生成条目

后续生成两个宏片段：

```cpp
NPGR_CUSTOM_DENSE_NEURON_MODEL(
    kCustomLifConductanceV1ModelId, 10001,
    CustomDenseLifConductanceV1Model,
    UpdateCustomLifConductanceV1DeviceEntry)

NPGR_CUSTOM_DENSE_LEARNING_MODEL(
    kCustomRStdpV1ModelId, 11001,
    CustomDenseRStdpV1Model,
    ApplyCustomRStdpV1PreDeviceEntry,
    ApplyCustomRStdpV1PostDeviceEntry,
    ApplyCustomRStdpV1TriggerDeviceEntry)
```

生成器必须维护 factory id lock file，例如 `model_specs/model_id_lock.json`；禁止根据目录遍历顺序重新编号。

## 10. 审批前检查表

- [ ] 首包不改动任何手写 LIF/STDP 公式、名称和默认 factory binding。
- [ ] 没有 custom spec 时，生成器仍产出可编译的空 registry 和空 catalog function。
- [ ] Catalog 条目与 Factory 创建项通过双向测试。
- [ ] 生成类使用 `ForwardEulerMethod<GeneratedConcreteModel>`。
- [ ] `ProcessSpike` 只处理接收输入，不调用学习规则。
- [ ] `CaculateSpike` 只标记输出 spike 和执行 reset；传播仍交给 `TimeDrivenInternalSpike`。
- [ ] 生成物不写回 source tree。
- [ ] CMake 通过显式 `target_sources` 编译生成源。
- [ ] 维护者可通过 CMake/CI 产出包含生成模型的官方 wheel。
- [ ] 干净环境中的用户无需 generator/编译器即可安装并运行官方 wheel。
- [ ] baseline 测试和性能测量在同一提交中可复现。
