# NeuronBridge 继承式模型代码生成维护架构

**状态：设计冻结前的实施蓝图**
**范围：主网（legacy）neuron model、主网 learning rule、Dense neuron model、Dense learning rule、统一 catalog 与 Python 友好 API**
**不在首个实施范围：任意 C++ 方程文本执行、自动生成 OuterDynamic 数值求解器、一次性重写历史事件队列与网络布局。**

## 1. 背景与问题定义

工程包含两个同等重要但运行契约不同的神经元路径，以及两套不能混同的学习规则路径。主网（现有目录名为 legacy）拥有 CPU time-driven model、GPU interface/device model、积分方法 factory、事件队列与按模型/queue 分组的状态布局；主网 learning rule 由 `LearningRuleModelFactory` 创建，并以 `LearningRule`/`WithPostSynaptic`/`WithTriggerAndPostSynaptic` 与 `SynapseState` 表达事件和突触状态。Dense 子网则采用 flat field table、统一 CUDA kernel 与 model-id dispatch，Dense learning 使用自己的 pre/post/trigger CUDA hook。`LegacyNeuronModelList.inc`、`DenseNeuronModelList.inc`、`DenseLearningModelList.inc` 已分别收敛了一部分注册关系，说明体系具备生成化的基础。

但新增一个模型仍需要维护多处手写代码：模型元数据、字段 slot、参数别名、初始化、派生量、CUDA update、注册清单、Python 友好构造器、文档和 baseline。任何一处遗漏都可能在运行很晚时才暴露为 CUDA 错误或 C++/Python baseline 偏差。

本方案把“模型语义”定义成可验证的声明式规格，并将其编译为主网或 Dense 所需的不同 C++/CUDA glue code。性能关键路径保持不变：Dense 仍是 model id `switch` 到 `__device__ inline`；主网仍保持既有 model interface、integration object 与 event scheduling，不把 Dense 的 flat-array 模型硬套到主网。

## 2. 目标、非目标与设计原则

### 2.1 目标

1. 一个派生语义规格同时驱动主网 neuron、主网 learning rule、Dense neuron、Dense learning 的 metadata、注册表、Python API 元数据、参考实现和测试夹具。
2. 新增常规主网或 Dense 神经元、主网或 Dense 学习规则时，作者主要新增一个规格文件和少量受限扩展，不修改各自统一 runtime 的主循环。
3. `canonical_name`、stable model id、字段名、slot 顺序、参数别名和 API 语义可静态验证。
4. C++、CUDA 与 Python 参考模型从同一个已验证的中间表示生成，降低三侧方程漂移。
5. 保持当前 C++ 描述对象、`neuronbridge.NeuronLayer(model, ...)`、`LearningRule(name, ...)` 的向后兼容。
6. NeuronBridge 维护者通过版本锁定的 Python 3.12 generator toolchain 维护官方模型；发布 wheel 不要求用户安装代码生成器、C++ 编译器或 CUDA compiler，也不提供用户提交 spec 的构建入口。

### 2.2 非目标

- 不把任意 C++、CUDA、Python 片段塞进 JSON/TOML 后原样拼接执行。
- 不在第一个版本自动重写主网的事件队列、时间积分框架、历史 GPU interface template 或 OuterDynamic 求解器。
- 不承诺跨 CPU/GPU、跨 CUDA 版本逐 bit 一致；承诺为每个模型定义可重复的数值容差基线。
- 不以运行时解释 DSL 的方式替代编译期 CUDA 代码。

### 2.3 不可违反的约束

```text
模型语义规格 -> 编译期生成主网 adapter 或 Dense entry
               -> 主网：既有 interface/integration/event runtime
               -> Dense：flat arrays -> unified kernel -> model-id switch -> inline device function
```

- Dense device entry 没有字符串、动态内存、虚函数、递归、循环语句或 host callback。主网 GPU 现有 integration object 的构造与生命周期保持其既有语义，生成器不得把字符串判断带入其 per-neuron update。
- 一个模型的 field slot 顺序是 ABI；一经发布不得随意重排。
- 既有 stable model id 是持久标识而不是数组位置；generated variant 使用独立 variant id，二者都只能弃用/tombstone，不能复用。
- 生成结果必须可再生：相同输入、生成器版本与配置得到字节稳定的逻辑产物。

### 2.4 核心决策：现有模型保持默认，生成模型继承通用方程基类

现有主网 neuron、主网 learning rule、Dense neuron 和 Dense learning rule 均是经过 C++/CUDA/Python baseline 验证的**权威默认实现**。生成器不取代它们，不重命名 public canonical model，也不把现有模型的计算方程迁走。

但生成模型**不继承某个具体 LIF/STDP 等旧模型**。工程新增通用、可扩展的手写基类：

```text
CustomTimeDrivenNeuronModel : public TimeDrivenModel
CustomLearningRuleModel     : public LearningRule
CustomDenseNeuronModel      : public DenseNeuronModelBase
CustomDenseLearningRuleModel: public DenseLearningRuleModelBase
```

生成规格描述的是一个基于确定方程的 `custom variant`：它声明 `extends.base_class` 为上述通用基类之一，以及 base contract version/digest。生成类型继承通用基类，基类负责状态存储、输入递送、积分调度、事件/突触状态生命周期与 registry contract；生成子类只提供方程、参数、状态变量、事件跳变、阈值/复位和 rule hook 的声明性实现。

```text
existing handwritten LIF/STDP/etc. (default, authoritative)

CustomTimeDrivenNeuronModel / CustomLearningRuleModel (generic extension base)
                  ^
                  | equation-driven derived implementation
                  |
generated custom model (opt-in variant, independently testable)
```

现有 canonical name 永远继续映射到既有实现。生成 variant 使用内部 implementation name（例如 `CustomLifConductanceV1`）和独立 variant id，只能由显式测试/实验配置选中。只有用户明确批准某个新模型作为公共模型时，才为该**新方程模型**分配新的 canonical name；不会覆盖已有模型。

### 2.5 方程驱动的通用基类契约

`CustomTimeDrivenNeuronModel` 不是某个 LIF 的重命名，而是 `TimeDrivenModel` 的通用实现骨架。它拥有 generic state table、parameter table、pending input、integrator 与 event dispatcher，并将模型差异限制在下列受保护方程接口：

```cpp
class CustomTimeDrivenNeuronModel : public TimeDrivenModel {
protected:
    virtual void EvaluateDerivatives(const EquationStateView& state,
                                     const EquationInputView& input,
                                     float time_ms,
                                     DerivativeView* d_state_dt) const = 0;
    virtual void ApplyDiscreteEvents(EquationStateView* state,
                                     float time_ms,
                                     SpikeDecision* decision) const = 0;
    virtual const EquationModelDescriptor& Descriptor() const = 0;
public:
    void UpdateState(int index, int time, Simulation* simulation) final;
};
```

`UpdateState` 由通用基类统一完成“读入输入 -> 积分 `dx/dt=f(x,u,t)` -> 应用 event jump -> 发放 spike -> 写回状态”。生成子类只实现由方程 IR 导出的 `EvaluateDerivatives` 与 `ApplyDiscreteEvents`。首版只提供 Forward Euler；RK2/RK4 等只在通用 base、reference 和 CUDA 三侧同时实现后才加入 `integration.method`。

`CustomLearningRuleModel` 同理派生 `LearningRule`：它统一管理 `SynapseState`、连接定位、pre/post/trigger 分发和 weight bounds；生成子类声明 trace ODE 与 event update，例如：

```text
dx_pre/dt  = -x_pre / tau_pre
dx_post/dt = -x_post / tau_post

on_pre:  weight <- clamp(weight + A_plus * x_post, w_min, w_max)
         x_pre  <- x_pre + 1
on_post: weight <- clamp(weight - A_minus * x_pre, w_min, w_max)
         x_post <- x_post + 1
```

这套表达可同时覆盖纯 ODE 神经元、ODE + threshold/reset 神经元、STDP trace、R-STDP eligibility trace 和 trigger-driven weight jump；每一条方程、赋值和事件顺序都会进入 manifest 与 differential test。

## 3. 当前工程映射

| 当前入口 | 当前职责 | 生成化后的归属 |
|---|---|---|
| `source_file/legacy/.../ModelFactory/LegacyNeuronModelList.inc` | 主网 CPU/GPU creator registry | 保持既有默认项；额外 include generated variant fragment |
| `source_file/legacy/.../ModelFactory/NeuronModelFactory.cpp` | 主网 alias resolve 后的 model 创建 | 保持 factory；为显式 variant selector include 生成 registry fragment |
| `source_file/legacy/.../ModelFactory/IntegrationMethodList.inc` | 主网 CPU/GPU integration method 选择 | 保持独立规格与 emitter；不得与 Dense integration 混用 |
| `source_file/legacy/.../NeuralModel/inc/TimeDriven*` | 主网 CPU/GPU state、event、interface 实现 | 保持默认实现，不作为 generated subclass 的父类 |
| `source_file/legacy/.../NeuralModel/inc/Custom/CustomTimeDrivenNeuronModel.h`（新增） | 通用 ODE state、输入、积分和 spike/reset contract | generated time-driven neuron 的唯一主网基类 |
| `source_file/legacy/.../ModelFactory/LearningRuleModelFactory.h` | 主网 learning rule 的 `STDP`/`R_STDP`/Cerebellar 创建 | 保持默认 factory；额外 include generated learning variant fragment |
| `source_file/legacy/.../LearningRule/inc/LearningRule.h`、`Rule/*`、`State/*` | 主网 rule hook、trigger 语义与 `SynapseState` 所有权 | 保持默认实现，不作为 generated subclass 的父类 |
| `source_file/legacy/.../LearningRule/inc/CustomLearningRuleModel.h`（新增） | 通用 trace/state、pre/post/trigger dispatch、`SynapseState` lifecycle | generated main learning rule 的唯一主网基类 |
| `source_file/shared/include/learning_rule/LearningRuleCatalog.h` | 主网/Dense 共用 rule name/trigger 语义 catalog | 保持 canonical rule；生成 variant metadata，禁止覆盖已有 name |
| `source_file/gpu/include/dense_subnetwork/model/IDenseNeuronModel.h` | Dense neuron host 扩展契约 | 保持稳定；`CustomDenseNeuronModel` 实现其通用 contract |
| `DenseNeuronModelList.inc` | id、factory、device entry 的共同清单 | 保持既有默认项；生成独立 variant list |
| `DenseNeuronDeviceUpdate.cuh` | 由宏清单生成 CUDA switch | 保持手写的统一 switch；include 生成的 list |
| `DenseLifExponentialDoubleModel.h` | fields、aliases、derived state、reset | 保持默认 LIF 实现；不作为 generated model 父类 |
| `DenseLifExponentialDoubleDeviceUpdate.cuh` | device update equation | 保持默认 LIF CUDA entry；不作为 generated equation 的来源 |
| `IDenseLearningRuleModel.h` | learning host 扩展契约 | 保持稳定；`CustomDenseLearningRuleModel` 实现其通用 contract |
| `DenseLearningModelList.inc` | learning id 与三阶段 hook entry | 保持既有默认项；生成独立 learning variant list |
| `neuronbridge.NeuronLayer` / `LearningRule` | Python 通用描述 API | 不要求用户侧迁移；生成可选友好构造器与 schema |

`NeuronModelCatalog` 仍是用户可见模型语义的总入口。每个 spec 必须显式说明 `legacy_cpu`、`legacy_gpu`、`dense_gpu` 的支持矩阵和对应 implementation name；代码生成器生成 catalog fragment，最终 catalog 聚合和 backend capability policy 仍由 C++ 主工程控制。

## 4. 方案总览

```text
model_specs/*.json
        |
        v
  schema validation + semantic validation + backend capability validation
        |
        v
  Derived Model IR / Derived Learning IR
     |              |                |               |
     v              v                v               v
 main neuron/rule adapter  Dense neuron/rule entry registry  Python schema/reference/tests/docs
     |              |                |               |
     +--------------+----------------+---------------+
                         |
                         v
             CMake custom command / generated build directory
                         |
                         v
             existing main-network runtime, Dense runtime and neuronbridge wheel
```

生成器是一个小型“模型编译器”，不是模板替换脚本：先解析受限表达式，校验类型、读写时序、integration/backend capability 和字段依赖，再由同一个 IR 分别生成主网 C++/CUDA adapter、Dense C++/CUDA entry 和参考实现。

## 5. 权威基类与单一派生源

推荐新增以下受控源目录：

```text
model_specs/
  schema_version.json
  neurons/
    lif_exponential_double.json
    poisson_rate.json
  learning_rules/
    legacy_stdp_variant.json
    stdp.json
  common/
    units_and_helpers.json

tools/model_codegen/
  generate.py                 # Python 3.12 CLI
  parser.py                   # expression parser
  ir.py                       # typed intermediate representation
  validators.py
  emit_cpp.py
  emit_cuda.py
  emit_python.py
  emit_docs.py
  emit_tests.py
  tests/

${build}/generated/model_codegen/
  include/dense_subnetwork/generated/
  python/neuronbridge/generated/
  tests/model_codegen/
  docs/model_catalog.generated.md
  manifest.json
```

现有手写模型是既有行为的权威默认源；通用 `Custom*` base 的公开/受保护 contract 是生成模型的稳定扩展源；equation spec 是每个生成模型的单一行为源。规格、生成器、generator unit tests 和测试输入向量必须提交版本控制。`${build}/generated` 一律不提交；它是 CMake 的确定性派生物。这样不会出现“提交了旧的 `.cuh`，但本地规格已经变了”的双事实源问题。

手写模型和生成 custom 模型长期并存。生成的 registry fragment 与手写 registry fragment 在 configure 时合并；baseline 通过后只证明 custom variant 可用，**不删除、不替换**既有手写模型或其默认注册。

## 6. 规格格式与 DSL

### 6.1 文件格式选择

第一版使用 UTF-8 JSON，而不是 YAML：Python 3.12 标准库即可解析、类型明确、无 YAML 隐式类型陷阱，也不向构建或 wheel 引入新依赖。JSON 只承载声明；表达式由独立受限语言解析。

一个 neuron spec 的骨架如下：

```json
{
  "schema_version": 1,
  "kind": "custom_time_driven_neuron",
  "variant_id": 10001,
  "extends": {
    "base_class": "CustomTimeDrivenNeuronModel",
    "base_contract_version": 1,
    "base_contract_digest": "sha256:declared-by-base-contract"
  },
  "symbol": "kCustomLifConductanceV1Id",
  "implementation_name": "CustomLifConductanceV1",
  "canonical_name": "CustomLIF_Conductance_v1",
  "aliases": [],
  "backends": {
    "legacy_cpu": {"enabled": true, "implementation_name": "TimeDrivenLIF_Exponential_double", "integration": "ForwardEuler"},
    "legacy_gpu": {"enabled": true, "implementation_name": "TimeDrivenLIF_Exponential_double_GPU", "integration": "ForwardEuler"},
    "dense_gpu": {"enabled": true, "implementation_name": "TimeDrivenLIF_Exponential_double"}
  },
  "api": {"constructor": "custom_lif_conductance_v1", "visibility": "experimental"},
  "fields": [
    {"name": "v", "storage": "float32", "role": "state", "default": -65.0},
    {"name": "tau_m_ms", "storage": "float32", "role": "parameter", "default": 20.0,
     "aliases": ["tau", "tau_m_ms"], "unit": "ms"}
  ],
  "inputs": [{"channel": "excitatory_conductance", "target": "gexc", "clear": true}],
  "derived": [{"target": "membrane_alpha", "expr": "dt_ms / tau_m_ms"}],
  "reset": [{"target": "v", "expr": "v_rest"}],
  "equations": {
    "continuous": [
      {"d_dt": "v", "expr": "(v_rest - v + gexc * (e_exc - v) + ginh * (e_inh - v) + input_current) / tau_m_ms"},
      {"d_dt": "gexc", "expr": "-gexc / tau_exc_ms"},
      {"d_dt": "ginh", "expr": "-ginh / tau_inh_ms"}
    ],
    "events": [
      {"when": "v >= v_threshold", "emit_spike": true,
       "assign": [{"target": "v", "expr": "v_reset"}, {"target": "refractory_ms", "expr": "t_ref_ms"}]}
    ]
  },
  "integration": {"method": "forward_euler", "dt_unit": "ms"},
  "validation": [{"assert": "tau_m_ms > 0", "message": "tau_m_ms must be positive"}]
}
```

### 6.2 受限表达式语言

规格的核心是方程而不是通用 update 伪代码。连续状态必须由 `d_dt` 微分方程定义；离散状态变化必须由 event jump/assignment 定义。支持标量 `float32`、`int32`、`uint8/bool` 与已声明字段。表达式支持：

```text
literal, field name, + - * /, unary -, comparisons, && || !,
exp, log, min, max, clamp, abs, select(condition, a, b), floor, ceil
```

语句只支持以下结构化操作：

```text
d_dt(state) = expression     # 连续状态的 ODE 右端项
on_event(condition): assign  # 阈值、外部触发或 spike 到达时的离散跳变
read_input(channel)          # 读取并清空对应 pending input
emit_spike(condition)        # 仅在明确 event 中定义
```

禁止指针、索引算术、循环、任意函数调用、文件/网络访问、随机隐式状态和原样 C++/CUDA 注入。随机模型必须显式依赖统一 runtime 传入的 RNG primitive，且要求记录 seed/sequence 契约。

### 6.3 时序语义

对每个 Dense step 固定执行阶段，避免同一方程在 C++ 与 CUDA 中因读写顺序不同而漂移：

```text
1. consume_inputs
2. compute_next                 # 只读本步开始状态及临时值
3. threshold_or_events
4. commit_state                 # 一次性写回 state
5. report_spike
```

`derived` 只在 host build/reset 时计算，必须标记其对 `dt_ms` 和 parameter fields 的依赖。生成器对 derived dependency graph 做拓扑排序并拒绝环。`state` 在 device step 里写入；`parameter` 在 device step 中只读；`debug` 不能影响数值结果。

主网执行阶段不强制改写成 Dense 的五阶段。spec 必须声明 `execution_contract: legacy_time_driven`，并附带 CPU/GPU state-vector layout、event emission timing、input/current delivery hook、integration method capability、GPU queue grouping 与 host/device ownership。生成器据此生成适配层和注册信息；只有通过主网 contract profile 验证的常规模型才可自动生成完整 CPU/GPU equation。复杂 event-driven 或依赖历史 template 的模型使用 `manual` backend，并仍享有自动 catalog/registry/schema/test 生成。

### 6.4 学习规则 DSL

learning rule 与 neuron model 使用不同顶层 kind，保留现有 `pre`、`post`、`trigger` 三 hook：

```json
{
  "kind": "dense_learning_rule",
  "stable_id": 0,
  "canonical_name": "STDP",
  "rule_fields": [],
  "synapse_fields": [],
  "hooks": {"pre": [], "post": [], "trigger": null},
  "spike_history": {"window_ms": 0.0}
}
```

每个 hook 只可访问其明确传入的 event time、synapse fields、rule-derived constants、weight 与声明的 spike history view。生成器据此计算 hook flags，并生成现有的 pre/post/trigger entry wrapper。不允许 hook 隐式改变未声明字段。

### 6.5 主网 learning rule 派生规格

主网 learning rule 使用独立 kind：`custom_learning_rule`。它继承新增通用 `CustomLearningRuleModel : public LearningRule`，而不是现有 `STDP`、`R_STDP`、`CerebullarLearningRule` 或 `IDenseLearningRuleModel`。其 contract 必须显式描述：

```text
base_rule_class = CustomLearningRuleModel / base_contract_digest
InitState 的 SynapseState 类型、状态维度与所有权
pre-spike、post-spike、trigger 的可覆写 hook 与调用时序
connection/neuron/simulation 可见性
权重读写、max-weight/clipping 语义
trigger route、eligibility 清除与 buffered activity 约束
```

生成的 host 类型形如 `class GeneratedCustomRStdpV1 : public CustomLearningRuleModel`。它由方程规格生成 trace 的微分方程、pre/post/trigger 跳变和 weight update；通用基类实现 `InitState`、`SynapseState` 生命周期和事件分发。`SynapseState` 的分配/释放和连接索引语义由通用基类 contract 控制。

主网 learning rule 不能假设有 Dense CUDA hook。若用户未来需要 GPU learning，它应作为独立 `CustomDenseLearningRuleModel` 方程模型，二者通过共享 canonical rule family、参数映射和 baseline trace 对齐，而不是通过继承相互替换。

### 6.6 借鉴 Brian2 的用户方程语言

Brian2 将模型写成一阶 ODE、子表达式和参数三类声明，并将 threshold/reset/refractory 与 synaptic event code 分离；这些声明会被解析和翻译为后端代码，而不是作为任意 Python 执行。NeuronBridge 应借用这个分层思想，而不依赖 Brian2 runtime 或复制其完整语法。[Brian2 equations documentation](https://brian2.readthedocs.io/en/2.7.1/user/equations.html)

完整用户格式、语法和示例见 [NeuronBridge 用户自定义方程格式规范](neuronbridge_custom_equation_format.md)。

建议在 JSON envelope 中使用中性的 `equations` 多行字段。它是 **NeuronBridge 自己的受限语言**，不是 Brian2 文件，也不承诺完整语法兼容：

```json
{
  "kind": "custom_time_driven_neuron",
  "extends": {"base_class": "CustomTimeDrivenNeuronModel", "base_contract_version": 1},
  "equations": "dv/dt = (v_rest - v + g_exc*(e_exc-v) + g_inh*(e_inh-v) + i_input) / tau_m : mV\ndg_exc/dt = -g_exc / tau_exc : uS\ndg_inh/dt = -g_inh / tau_inh : uS\ni_input = current_input : nA (constant_over_dt)\nv_rest : mV (parameter, constant)\ntau_m : ms (parameter, positive, constant)",
  "spike": {"when": "v >= v_threshold", "reset": "v = v_reset; refractory_ms = t_ref"},
  "refractory": "refractory_ms > 0",
  "integration": {"method": "forward_euler"}
}
```

语言首版只支持以下声明元素：

| 元素 | NeuronBridge 语义 | 首版状态 |
|---|---|---|
| `dx/dt = f : unit` | 一阶 ODE；生成 derivative 与 integrator 输入 | 支持 |
| `x = f : unit` | 只读子表达式，按需计算，不分配 state storage | 支持 |
| `x : unit (parameter, ...)` | 参数或初始状态声明 | 支持 |
| `constant` / `constant_over_dt` | 构建期常量 / 本 step 缓存 | 支持 |
| `shared` | population/rule 共享标量 | Phase 2 |
| `unless_refractory` | state 在 refractory 中冻结 | Phase 2 |
| `event_driven` | 仅 event 时解析更新 | 仅为可证明的一维线性 trace 开放 |
| `xi` / stochastic equation | 可复现实验 RNG 流 | 后续；先不开放 |

借鉴 Brian2 的同时必须保留四个 NeuronBridge 约束：

1. 方程字符串被自有 lexer/parser 解析为 AST，不能访问 Python/C++ namespace，不能调用任意函数。
2. 单位采用项目规范单位（目前以 `ms`、`mV`、`nA`、`uS` 为可读 canonical unit），并做严格维度检查；导出的 native storage 不因显示单位而改变。
3. 外部量必须出现在显式 `parameters`、`inputs` 或已注册函数表中，禁止从调用者 Python locals/globals 隐式捕获。
4. `threshold`、`reset`、`refractory`、`on_pre`、`on_post`、`on_trigger` 是独立 AST block，生成器固定其调度顺序并写入 manifest。

对 learning rule，采用同样的方程语法和事件块：

```text
dx_pre/dt  = -x_pre/tau_pre  : 1 (event_driven)
dx_post/dt = -x_post/tau_post : 1 (event_driven)

on_pre:  w = clamp(w + a_plus*x_post, w_min, w_max); x_pre += 1
on_post: w = clamp(w - a_minus*x_pre, w_min, w_max); x_post += 1
on_trigger: eligibility = 0
```

不过，自动 event-driven 积分只在 AST 可证明为一维线性衰减、且不依赖 continuous state 时启用；否则由 `CustomLearningRuleModel` 按选择的 fixed-step method 更新。这个限制与 Brian2 对 event-driven equation 的依赖限制同方向，但更保守，适合当前 C++ baseline 优先的工程阶段。[Brian2 event-driven equation constraints](https://brian2.readthedocs.io/en/2.7.1/user/equations.html)

## 7. 中间表示、类型与数值规则

### 7.1 Typed IR

解析后，IR 必须保存：`extends` 基类、base contract version/digest、字段 storage/role、slot、单位、别名、表达式 AST、每个 read/write、stage 顺序、variant id、生成器版本和 spec source digest。Emitter 只能接受已经通过 semantic validation 的 IR。

必须检查：

- variant id、generated implementation name 全局唯一；已有 canonical name 与 alias 不得被 generated variant 占用。
- `extends.base_contract_digest` 必须与基类导出的 contract manifest 一致；基类 contract 变更必须使所有派生 spec 在 configure 阶段失效并要求重新验证。
- field slot 无重复，所有 device 访问字段都已入 slot。
- `parameter` 不能在 update 中写；`derived` 不能被 input/event 写。
- input target 存在且 storage/role 合法；spike effect channel 合法。
- 类型可转换规则明确，禁止 float 到 int 的隐式截断。
- 对同一 field 的多次 write 是否由结构化 `if` 互斥；否则拒绝。
- model catalog capability 与 spec backend capability 不冲突。

### 7.2 float 与单位

- Dense runtime 的连续状态以 `float32` 为规范类型。主网 emitter 必须从 profile 读取并保留既有 CPU/GPU scalar type、state-vector storage 和 cast boundary，不能因为共享 spec 而擅自改成 Dense `float32` ABI。生成 CUDA 时使用目标 backend 的显式类型/常量；reference 在每个 store 和 profile 指定的临时边界复现对应精度语义。
- 时间统一以 `ms`，电位以 `mV`，导通量/电流沿用既有模型的 project unit，并在 field 中强制记录 `unit`。
- 生成器不自动做单位换算；不同单位的表达式在 schema validation 时报错。第一版单位校验覆盖 `ms`、`mV`、`nA`、`uS`、`dimensionless`，未知单位必须显式标成 `project_defined:<name>`。
- `exp`、阈值比较、refractory 边界和随机采样的求值顺序是模型契约的一部分，写入 generated manifest。

### 7.3 一致性定义

1. **结构一致性**：host schema、slots、CUDA field access、catalog 和 Python schema 由同一 IR 导出。
2. **单后端确定性**：给定 build、seed、input trace，在同一 backend 上结果可重复。
3. **跨后端数值一致性**：reference/CUDA 对状态、spike、weight 按模型声明的 absolute/relative tolerance 比较；不把 FMA 或数学库微差误判为回归。
4. **行为一致性**：reset、refractory、input clear、event ordering 等离散语义必须精确一致。

### 7.4 主网专属 IR 与 emitter

主网与 Dense 自定义方程模型共享 `canonical_name` family、参数、状态、单位、阈值/复位和随机契约，但**不共享运行时 ABI**。主网 emitter 的输入是 Equation Model IR 加 `LegacyTimeDrivenProfile`，输出至少包括：

```text
1. LegacyNeuronModelList registry fragment（CPU/GPU implementation binding）
2. NeuronModelCatalog backend binding fragment
3. CPU `class Generated... : public CustomTimeDrivenNeuronModel`（生成 `CaculateDifferentialEquation`、event jump、parameter schema）
4. GPU `CustomTimeDrivenNeuronModelGPU` interface/device pair（通用 base 处理 allocation/upload/object lifecycle，生成 device derivative/event entry）
5. IntegrationMethodList capability fragment 与不支持时的清晰诊断
6. 主网 C++ baseline case：state trace、spike timestamps、input/current delivery 与 queue partition
```

第一版自动生成只覆盖 `legacy_time_driven` + `CustomTimeDrivenNeuronModel` + 已声明 `ForwardEuler` profile 的普通核心模型。通用 base contract 固定并校验：state variable order、global/local index mapping、`UpdateState` 的 event 可见时刻、突触电流/导通量写入点、GPU queue partition 和资源释放职责。生成器不会生成未知的 GPU object ownership 或把历史 `Integration_method_GPU_Factory` 的字符串选择复制到热路径。

这意味着新模型有三种明确落点：

| 模型类型 | 生成程度 | 作者仍需手写 |
|---|---|---|
| 主网常规 time-driven（标准 profile） | 从通用 base 派生 CPU/GPU equation model、variant registry、schema、reference、test | 方程 spec；必要时在通用 base 新增经过审核的 helper |
| 主网特殊 time-driven（非标准 state/event） | variant catalog、registry、schema、test skeleton | 从 `CustomTimeDrivenNeuronModel` 扩展 profile，或说明明确的 `manual` backend |
| event-driven/Input/OuterDynamic interface | catalog、registry、Python schema 可生成 | 运行语义保持手写；不强行以 neuron equation DSL 表达 |

因此，主网神经元不是方案的“迁移尾项”，而是与 Dense 并列的第一类派生目标；区别仅在于它有更严格的 profile gate 和更保守的首批自动生成范围。现有主网模型继续是默认行为。

### 7.5 主网 learning rule 专属 IR 与 emitter

主网 rule emitter 的输入是 Equation Rule IR 加 `LegacyLearningRuleProfile`。它与 neuron profile 独立，因为 rule 的核心 ABI 是 `LearningRule`、`Interconnections`、`Neuron`、`Simulation` 和 `SynapseState`，而非 state-vector 或 Dense field table。输出包括：

```text
1. LegacyGeneratedLearningRuleList.inc：显式 variant name -> creator
2. Generated... : public CustomLearningRuleModel：生成 trace ODE、pre/post/trigger jump 和 weight equation
3. LearningRuleCatalog generated variant metadata 与 alias collision check
4. SynapseState schema/initialization validator 和 state ownership test
5. event-trace reference case：pre/post/trigger、weight、eligibility、buffered activity
```

首批方程模型建议以 `CustomRStdpV1` 为目标，并以既有 `R_STDP` 的 event trace 作为 oracle，而不是继承或替换 `R_STDP`。它同时包含 pre、post、trigger 和 eligibility 清除语义，最适合校验 rule contract；手写 `R_STDP` 仍保持默认 factory 结果。

## 8. 生成产物

| 产物 | 用途 | 是否进入版本控制 |
|---|---|---|
| `LegacyGeneratedNeuronModelList.inc` | 主网派生 variant creator registry fragment | 否，build 生成 |
| `LegacyGeneratedNeuronAdapters.h/.cpp/.cu` | 标准 `legacy_time_driven` 的 `Generated... : public CustomTimeDrivenNeuronModel` 方程模型 | 否 |
| `LegacyGeneratedIntegrationCapability.inc` | model-to-integration capability 与诊断 | 否 |
| `LegacyGeneratedLearningRuleList.inc` | 主网派生 learning variant 的显式 creator registry fragment | 否 |
| `LegacyGeneratedLearningRules.h/.cpp` | `Generated... : public CustomLearningRuleModel` 与方程生成的 rule hooks | 否 |
| `LegacyGeneratedSynapseStateSchema.json` | 主网 rule state dimension/ownership/trace 验证元数据 | 否 |
| `DenseGeneratedNeuronModelList.inc` | stable id、host class、device entry 的 registry fragment | 否，build 生成 |
| `DenseGeneratedNeuronModels.h` | `class Generated... : public CustomDenseNeuronModel`，生成 fields/ODE/events | 否 |
| `DenseGeneratedNeuronDeviceUpdates.cuh` | 无 CUDA 继承；生成 static `__device__` ODE/event entry，并组合通用 base helper | 否 |
| `DenseGeneratedLearningModelList.inc` | 派生 learning variant id 与 hook entry fragment | 否 |
| `DenseGeneratedLearningDeviceUpdates.cuh` | 无 CUDA 继承；生成 pre/post/trigger entry 并组合基类 device helper | 否 |
| `generated_model_catalog.inc` | catalog 条目 fragment | 否 |
| `neuronbridge/generated/models.py` | optional friendly constructors、schema introspection | 否，wheel 内包含 |
| `model_reference.py` / C++ reference | 基线与 differential test | 否 |
| `model_cases.generated.json` | deterministic vectors、tolerances、spec digest | 否 |
| `model_catalog.generated.md` | API 文档表 | 否，可发布到文档站 |

初期不直接替换 `DenseNeuronDeviceUpdate.cuh` 的 switch，也不直接替换主网 `NeuronModelFactory.cpp`、integration factory 或 GPU interface 生命周期。两个运行时都保留极小而稳定的手写壳，只 include 手写与生成 registry fragment。generated variant 默认不进入 canonical-name resolution；这样 generator 出错时问题被限制在 model artifacts，不影响主网事件控制流、Dense unified kernel 或已有用户模型。

## 9. CMake 与发布集成

### 9.1 独立于 Python binding 的开关

新增：

```cmake
option(NR_ENABLE_MODEL_CODEGEN "Generate declarative main-network and Dense model artifacts" ON)
option(NR_MODEL_CODEGEN_VERIFY "Run generator freshness checks in tests/CI" ON)
```

`NR_ENABLE_MODEL_CODEGEN` 与 `NR_ENABLE_PYTHON` 独立。即便 Python extension 关闭，主网和 GPU C++ 构建也必须生成模型代码；这符合“计算核心由规格维护”的目标，也能更早暴露 spec/CUDA/主网 adapter 错误。

configure 时通过 `find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter)` 找到生成器解释器。该依赖仅属于开发/构建机，**绝不**作为已发布 neuronbridge 用户的安装前置条件。

### 9.2 target 依赖

1. `add_custom_command(OUTPUT ...)` 声明所有 generated headers、includes、Python schema 与 manifest。
2. `add_custom_target(nr_model_codegen DEPENDS ...)` 聚合生成结果。
3. `nr_snn_core_gpuaware_cpp`、`nr_snn_core_gpuaware_cuda`、主网 neuron/learning-rule app and smoke target、dense test target、`neuronbridge_core` 均 `add_dependencies(... nr_model_codegen)`。
4. 将 `${CMAKE_BINARY_DIR}/generated/model_codegen/include` 放在 target include path 的前部，避免 source tree 旧 generated 文件被误 include。
5. Python package build 从 build output 复制 generated `models.py` 与 manifest，wheel 内是静态产物，不需要用户执行 generator。

生成命令必须带 `--spec-root`、`--output-root`、`--generator-version`、`--depfile`，并把所有 spec 和 generator Python 模块列为 `DEPENDS`。`--check` 模式不写文件，用于 CI 确认构建产物未过期并验证 generator 幂等性。

### 9.3 增量和 IDE 行为

- VS/CMake generator 只在 specs 或 generator 源改变时重跑；使用 depfile 或 configure-time spec list 更新依赖。
- 输出先写入 temporary directory，通过内容比较后原子替换，避免 IDE 编译同时读到半写文件。
- 生成失败时打印 spec path、JSON pointer、DSL source span 和建议修复方式；不把 CUDA 编译器错误作为首个诊断。

## 10. Python API 策略

现有通用 API 是长期主入口：

```python
layer = neuronbridge.NeuronLayer(
    "TimeDrivenLIF_Exponential_double", 128,
    {"tau": 20.0, "V_th": -50.0},
)
rule = neuronbridge.LearningRule("STDP", {"tau_plus": 20.0})
```

生成代码只增加便利层，绝不要求为每个示例增加专用 native 通道：

```python
layer = neuronbridge.models.lif_exponential_double(
    128, tau_m_ms=20.0, v_threshold=-50.0
)
print(neuronbridge.models.schema("TimeDrivenLIF_Exponential_double"))
```

生成的 Python 层负责：参数名/别名冲突诊断、默认值展示、单位与范围预检查、IDE type stub、文档链接。真实 model construction 与运行仍经由现有 binding 和 C++ catalog。这样 Python 不复制仿真核心逻辑。

## 11. 维护者新增模型工作流

1. 选择 `custom_time_driven_neuron`、`custom_learning_rule`、`custom_dense_neuron` 或 `custom_dense_learning_rule` 模板，选择对应的通用 `Custom*` base contract，并分配未使用的 variant id。
2. 写 fields、aliases、inputs、derived/reset/event/hook 语义、backend support matrix、主网 integration profile 或 `LegacyLearningRuleProfile`（如适用）和数值 tolerance。
3. 运行 `python tools/model_codegen/generate.py validate --spec ...`，先获得 spec-level 报错。
4. 运行 `cmake --build <build> --target nr_model_codegen`，查看 generated schema 与 C++/CUDA 产物。
5. 执行 generated unit/reference/differential test；再加入真实 Simulation C++/Python baseline。
6. 评审 spec、自动生成 manifest diff、测试报告与 public API 文档，不以大段 generated `.cuh` diff 为主要评审对象。
7. variant 发布后，variant id、通用 base contract digest、已公开方程参数和 state/slot layout 进入兼容性保护；已有 canonical name 仍归手写模型，弃用走 variant metadata。

对于 DSL 不足以表达的派生模型，允许一个非常窄的 escape hatch：spec 标记 `implementation: manual`，仍由 generator 生成 variant registry/catalog/Python schema/test skeleton，但派生类的 host/device equation 保持手写。新增 escape hatch 必须说明为什么不能扩展 DSL；不得用它绕过验证。

## 12. 验证与 CI

### 12.1 四层测试

| 层级 | 内容 | 失败定位 |
|---|---|---|
| generator unit | parser、type/unit checker、IR lowering、determinism | codegen implementation |
| spec validation | ids、slots、fields、aliases、phase writes、capability | 单个 spec |
| generated reference | reset/input/event/edge cases、fixed RNG vector | 模型语义 |
| compiled differential | 主网 CPU vs 主网 GPU、主网 rule event trace、Dense host reference vs CUDA、Simulation C++ vs Python baseline | 运行时或 binding |

每个模型至少有：默认参数、参数边界、refractory/threshold、每种 input channel、reset、长序列、固定 seed（如有随机性）六类测试。主网 neuron 额外覆盖 CPU/GPU state-vector order、event timestamp、queue partition 和 integration-method compatibility；主网 learning rule 额外覆盖 `SynapseState` 初始化/释放、connection index、pre/post/trigger 调用顺序、eligibility 与权重变化；Dense learning 额外覆盖 pre-only、post-only、trigger、weight clipping、spike-history window。

### 12.2 CI 阶段

```text
1. codegen --validate-all --check-deterministic
2. CMake configure with NR_ENABLE_MODEL_CODEGEN=ON, NR_ENABLE_PYTHON=ON
3. build nr_model_codegen + main-network runtime + dense runtime + neuronbridge_core
4. generated model unit/reference tests
5. C++ main-network smoke/regression + dense smoke/regression
6. Python test_results + model constructor/schema tests
7. wheel smoke in a clean Python 3.12 environment
```

CI 还必须验证 manifest 中 id 没有重排或复用，并比较 public catalog snapshot。对规格变更，报告显示 canonical name、public parameters、fields、slots、default/tolerance 的语义 diff。

## 13. 分阶段迁移计划

### Phase 0: 基线冻结（先做）

- 为主网 CPU/GPU neuron、主网 `STDP`/`R_STDP`/Cerebellar rule、7 个 Dense neuron、4 个 Dense learning rule 建立 inventory：canonical name、backend implementation name、基类、generation contract、aliases、fields/state-vector 或 `SynapseState` layout、slots、inputs、effects、hooks、integration/trigger capability、tests。
- 保存当前 C++/Python baseline 和 GPU architecture/CUDA version；这是生成模型的比较基线。
- 决定 stable id tombstone 表与 catalog owner。

**验收：** inventory 完整，`LegacyNeuronModelList.inc`、`Dense*ModelList.inc` 与 `NeuronModelCatalog` 的 backend binding 一致，并确定首个 `CustomTimeDrivenNeuronModel` contract。

### Phase 1: 编译器骨架，只生成非运行期产物

- 建立 JSON schema、parser、公共 Model IR、`LegacyTimeDrivenProfile`、Dense IR、validator、manifest、docs 和 Python schema emitter。
- 先不让生成的 CUDA/host code 被编译；用 `--check` 验证可重复生成。

**验收：** 一个 `CustomLifConductanceV1` equation spec 能同时生成主网/Dense variant schema、registry fragments、reference vector 与文档；无既有 C++ 行为变化。

### Phase 2: Custom LIF 方程模型双运行时 shadow generation

- 将确定方程的 `CustomLifConductanceV1` 规格化，并以现有 `TimeDrivenLIF_Exponential_double` 作为 oracle。
- 生成 `Generated... : public CustomTimeDrivenNeuronModel` 主网 CPU model、主网 GPU adapter pair 和 `CustomDenseNeuronModel` host model；CUDA 生成与通用 base helper 组合的 static entry 到独立 shadow target。
- 对主网 CPU、主网 GPU 和 Dense 既有 LIF 实现分别做逐 step state/spike differential baseline；variant 不进入正式 canonical registry。

**验收：** 默认、边界、长跑输入下满足定义的 tolerances；主网 event timing/queue partition，及 Dense slot/input clear/threshold/refractory 行为一致。

### Phase 3: 首个 custom 方程 variant 的受控注册

- 由生成 registry fragment 注册 `CustomLifConductanceV1` 的隐藏 generated variant，主网 CPU/GPU/Dense 均只能通过明确的 test/experimental selector 使用。
- 手写 LIF double 继续是主网与 Dense default implementation，保留为长期权威 oracle，不因 variant 通过 baseline 而移除。
- Python 自动生成 `models.custom_lif_conductance_v1()` 和 type stub。

**验收：** 现有主网与 Dense C++/Python examples、baselines、wheel smoke 完全不变；显式 variant baseline 通过；没有 Python 用户迁移要求。

### Phase 4: 扩展主网与 Dense 神经元家族

- 主网顺序建议：LIF triple、LIF decay、voltage jump、Izhikevich、Poisson；Dense 按相同语义模型跟进。
- `InputSpikeNeuronModel`、`InputCurrentNeuronModel`、`TriggerRelayNeuronModel` 与 `HandwritingTimeDrivenModel` 先只生成 registry/schema，待其事件契约形成 profile 后再自动生成本体。
- 先迁移确定性模型，再迁移有 RNG 的 Poisson；随机模型必须先锁定 RNG contract。

**验收：** 所有新派生普通主网/Dense neuron variant registry 条目由 specs 生成，两个运行时路径均无额外动态开销；既有 registry 继续保留。

### Phase 5: 主网与 Dense 学习规则派生

- 先为主网 `R_STDP` 生成隐藏 trigger-policy variant，再处理主网 `STDP` 和 Cerebellar rule；它们继续继承现有 `LearningRule` family。
- Dense 侧按 STDP、R-STDP、additive/cerebellar 的顺序生成独立 variant。
- 将主网 `SynapseState`/event trace，和 Dense pre/post/trigger phase、weight mutation/history window 分别放入各自 Rule IR。

**验收：** generated main learning-rule list 与 `LearningRuleModelFactory` 默认结果并存，generated Dense learning variant list 与既有 `DenseLearningModelList.inc` 并存；主网 rule event trace 与四种 Dense learning 权重曲线 differential baseline 通过。

### Phase 6: Catalog、主网特殊模型与 OuterDynamic 收敛

- 让 generator 输出 derived variant catalog fragments、主网 factory registry、integration capability 与 special-model skeleton。
- 主网特殊模型先生成元数据/注册，不自动生成未知 integration/event semantics；高性能 GPU 等式仍显式 `manual`，直到形成受测 profile。

**验收：** 新派生模型不再需要修改大型 `if/else` factory；不扩大主网事件、积分和 GPU 生命周期风险，也不替换既有模型。

## 14. 风险与缓解

| 风险 | 缓解 |
|---|---|
| DSL 不足导致作者反复绕过 | 先以 LIF/STDP pilot 校准；只在两个真实模型证明共性后扩展语法 |
| 生成代码难调试 | 每个 generated 行写 spec path/source span；保留 source map 与 `--emit-ir` |
| CUDA/CPU 浮点微差 | 明确 tolerance，固定操作次序，测试离散事件精确性而非盲目 bitwise |
| 主网 GPU interface/lifecycle 被错误生成 | 首版仅开放经过 `LegacyTimeDrivenProfile` 验证的 ForwardEuler 普通模型；其余保持 `manual` adapter |
| 主网与 Dense 被错误视为同一 ABI | 公共语义 IR 之后立即分流到 legacy/Dense emitter；CI 单独比较主网 CPU/GPU 与 Dense reference/CUDA |
| 基类私有实现不适合派生 | 基类只导出小型受保护 contract/helper；不改写既有默认路径，不能安全继承时保持 `manual` variant |
| generated variant 意外替换现有模型 | catalog 禁止 generated implementation 占用已有 canonical name；variant 只能通过显式 selector 注册/测试 |
| stable id 被误复用 | immutable id registry + CI tombstone 检查 |
| 构建系统依赖 Python | 仅开发构建依赖 Python 3.12；wheel 包含已生成产物；`NR_ENABLE_MODEL_CODEGEN` 默认开 |
| 大量 generated diff 难评审 | 审查 spec 和 semantic manifest；CI 重生生成物，生成文件不入库 |
| 迁移中断既有 baseline | shadow target、逐模型切换、保留手写 oracle、每一步 C++/Python dual baseline |

## 15. 首批实施任务

1. 创建 `model_specs/` inventory 与 immutable variant-id registry，先完整盘点 `LegacyNeuronModelList.inc`、`IntegrationMethodList.inc`、`LearningRuleModelFactory.h`、`LearningRule`/`SynapseState` family、Dense lists 和 catalog，并为候选基类写 generation contract manifest；不改变 runtime。
2. 实现 `tools/model_codegen` 的 JSON loader、restricted expression parser、Derived Model IR、`LegacyTimeDrivenProfile`/Dense derived validator 与 deterministic manifest emitter。
3. 在 CMake 增加 `NR_ENABLE_MODEL_CODEGEN=ON` 和 `nr_model_codegen`，先只生成 build-directory 文档、主网/Dense **variant** registry fragments 与 schema。
4. 为 `CustomLifConductanceV1` 写完整三 backend equation spec 与 reference vectors，建立主网 CPU/GPU、Dense host/CUDA shadow generation target；随后为 `CustomRStdpV1` 写主网 equation-rule spec 和 event-trace fixture。
5. 完成 custom LIF/custom R-STDP 与既有 LIF/R-STDP oracle 的 C++/Python differential test 后，才评审是否公开任何新 variant；既有 runtime 默认路径不切换。

## 16. 关键决议

- **决议 1：** 现有主网/Dense neuron、主网/Dense learning rule 是权威默认实现；`Custom*` 是生成模型的通用基类；方程 spec 是 custom variant 的唯一行为源，既有模型不因生成器落地而被替换。
- **决议 2：** generator 生成到 build directory，不提交 generated C++/CUDA/Python 文件。
- **决议 3：** `NR_ENABLE_MODEL_CODEGEN` 默认开启且独立于 `NR_ENABLE_PYTHON`，让依赖与编译错误尽早出现。
- **决议 4：** Python 是公共用户接口和可视化/工作流层，不复制 native simulation core；生成器只提供 schema、友好构造器和 reference/testing support。
- **决议 5：** 主网普通 time-driven、主网 custom R-STDP rule 与 Dense neuron 并列为第一类**方程生成**目标：先用 `CustomLifConductanceV1` 做神经元双运行时 shadow、再用 `CustomRStdpV1` 做主网 rule event-trace shadow；Dense learning 随后跟进，主网特殊模型/OuterDynamic 最后按 profile 收敛；不冒险用一个 DSL 重写或替换全部历史运行时。
