# NeuronBridge 维护者方程规格规范

**状态：部分实现；以本节标注的当前子集为准。**
**版本：0.2-draft**

## 1. 目的

本规范定义 NeuronBridge 维护者新增神经元和学习规则的声明格式。它借鉴 Brian2 的思想：使用可读的一阶微分方程、参数、子表达式和事件代码描述模型；但它是 NeuronBridge 自己的受限语言，不读取 Python 调用者 namespace，不执行任意 Python/C++，也不依赖 Brian2 runtime。该规格位于源码仓库，仅由维护者和 CI 使用，不是最终用户运行时 API。

生成模型从通用基类派生，而不是从现有某个具体 LIF/STDP 模型派生：

```text
CustomTimeDrivenNeuronModel : TimeDrivenModel
CustomLearningRuleModel     : LearningRule
CustomDenseNeuronModel      : DenseNeuronModelBase
CustomDenseLearningRuleModel: DenseLearningRuleModelBase
```

现有手写模型继续是默认实现。维护者方程模型是具有独立 `implementation_name` 和 `variant_id` 的 custom variant，必须显式选用，不能占用已有 canonical model name。

## 2. 文件结构

一个模型由 UTF-8 JSON 文件定义，推荐扩展名为 `.nbmodel.json`：

```text
model_specs/
  custom_neurons/
    custom_lif_conductance_v1.nbmodel.json
  custom_learning_rules/
    custom_rstdp_v1.nbmodel.json
```

JSON 负责描述模型身份、后端、参数和事件；`equations` 承载人类可读的方程块。该字符串由 NeuronBridge lexer/parser 处理，不属于 Python 代码。

## 3. 顶层 Envelope

所有模型都使用以下顶层字段：

| 字段 | 必需 | 说明 |
|---|---:|---|
| `schema_version` | 是 | 当前为 `1` |
| `kind` | 是 | 见第 4 节 |
| `variant_id` | 是 | 永久唯一的 custom variant id，不可复用 |
| `implementation_name` | 是 | C++ registry 使用的唯一内部名称 |
| `canonical_name` | 否 | 仅批准公开的新模型可填写；不得复用既有名称 |
| `extends` | 是 | 通用 `Custom*` base contract |
| `api` | 是 | Python 构造器名、可见性和文档摘要 |
| `equations` | 是 | 方程字符串 |
| `events` | 否 | spike、pre、post、trigger 等事件块 |
| `integration` | 是 | 数值积分方法与时间单位 |
| `parameters` | 否 | 参数范围、默认值和别名补充 |
| `validation` | 否 | 初始化时断言 |
| `reference` | 是 | baseline tolerance 和固定输入夹具 |

`extends` 必须显式引用对应通用基类：

```json
{
  "base_class": "CustomTimeDrivenNeuronModel",
  "base_contract_version": 1,
  "base_contract_digest": "sha256:<published-contract-digest>"
}
```

基类 contract digest 改变时，所有对应 spec 必须重新生成并完成 baseline 验证。

## 4. `kind` 与运行时目标

| `kind` | 通用 C++ 基类 | 目标 |
|---|---|---|
| `custom_time_driven_neuron` | `CustomTimeDrivenNeuronModel` | 主网 CPU time-driven neuron |
| `custom_dense_neuron` | `CustomDenseNeuronModel` | Dense GPU neuron |
| `custom_learning_rule` | `CustomLearningRuleModel` | 主网 event-driven learning rule |
| `custom_dense_learning_rule` | `CustomDenseLearningRuleModel` | Dense GPU learning rule |

主网和 Dense 可共享同一个“方程家族”名称、参数和 reference input，但必须各自声明符合自身 ABI 的 spec。不得让主网 learning rule 直接继承 Dense learning rule，反之亦然。

## 5. 方程块语法

### 5.1 三种声明行

方程字符串由多行组成；空行和以 `#` 开头的注释忽略。

```text
dx/dt = expression : unit (flags)  # 连续状态微分方程
x = expression : unit (flags)     # 子表达式，只读且按需计算
x : unit (flags)                  # 参数或未演化状态
```

示例：

```text
dv/dt     = (v_rest - v + i_total) / tau_m : mV
dg_exc/dt = -g_exc / tau_exc : uS
i_total   = current_input + g_exc*(e_exc - v) : nA (constant_over_dt)
tau_m     : ms (parameter, positive, constant)
v_rest    : mV (parameter, constant)
```

### 5.2 允许的表达式

当前已实现的表达式子集：

```text
literal, variable, +, -, *, /, unary -, parentheses
==, !=, <, <=, >, >=（用于单一 spike comparison）
exp, expm1, log, log1p, sqrt, abs, min, max, pow
```

当前函数参数数量固定为：`exp/expm1/log/log1p/sqrt/abs` 各 1 个，`min/max/pow` 各 2 个。CPU 分别发射 `std::exp` 等 `<cmath>` 函数，CUDA 分别发射 `expf`、`expm1f`、`logf`、`log1pf`、`sqrtf`、`fabsf`、`fminf`、`fmaxf`、`powf`。未知函数和错误参数数量在生成期拒绝。

以下是后续目标，当前 parser 尚未实现：

```text
**, and, or, not, clamp, floor, ceil, 子表达式声明
```

表达式是逐 neuron 或逐 synapse 求值；不支持 Python indexing、指针、循环、递归、动态分配、任意函数调用或 C++ 代码片段。

函数必须由 NeuronBridge 注册，并同时具有：

1. AST 类型签名和单位签名。
2. C++ reference emitter。
3. CUDA emitter（仅 Dense backend）。
4. Python reference emitter（后续阶段；当前不作为 runtime 生成前置条件）。

当前尚未落地完整维度图，因此生成器不会静态证明 `exp/log` 输入无量纲，也不会证明 `sqrt/log/pow` 的运行时定义域。spec 必须通过参数约束和模型 baseline 保证定义域；生成器不会为其在 neuron/synapse 热路径插入隐式 clamp、有限值检查或异常分支。

## 6. 单位与类型

### 6.1 Canonical units

用户输入使用项目可读单位；第一版支持：

```text
ms, mV, nA, uS, Hz, 1, boolean, integer
```

`1` 代表无量纲浮点变量。`boolean` 与 `integer` 是无量纲离散变量。内部 storage type 由字段声明决定，不因展示单位改变。

### 6.2 维度检查

生成器必须验证：

- `dx/dt` 的右侧维度等于 `unit(x) / ms`。
- 赋值、阈值比较、`clamp` 边界和 event assignment 的维度兼容。
- `exp`、`log` 的输入无量纲。
- `min`、`max`、条件分支的候选值同维。
- `integer`/`boolean` 不接受隐式 float 截断。

第一版不进行隐式单位换算。若方程要从 `mV` 转为 `V`，必须使用明确注册的换算函数或在参数层完成换算。

## 7. Flags

| Flag | 允许位置 | 语义 | 0.1 支持 |
|---|---|---|---:|
| `parameter` | 参数行 | 用户可配置的模型参数 | 是 |
| `constant` | 参数/子表达式 | 单次 run 中不能被事件修改 | 是 |
| `positive` | 参数 | 初始化时必须大于零 | 是 |
| `constant_over_dt` | 子表达式 | 每个时间步只计算一次 | 是 |
| `shared` | 参数/子表达式 | 层或 rule 共享标量 | 否，Phase 2 |
| `unless_refractory` | ODE 行 | refractory 内冻结该状态 | 否，Phase 2 |
| `event_driven` | learning trace ODE | 只在事件时解析更新 | 受限支持 |

`event_driven` 只允许一维、线性、解析可积分的 trace，例如：

```text
dx_pre/dt = -x_pre / tau_pre : 1 (event_driven)
```

它不能依赖 continuous state、非线性子表达式或另一个非 event-driven ODE。不能证明安全时，生成器拒绝该 flag，并要求使用 fixed-step integration。

## 8. 神经元事件块

神经元方程只负责连续演化。spike、reset、refractory 和外部输入通过显式事件块描述。

```json
{
  "events": {
    "spike": {
      "when": "v >= v_threshold and not_refractory",
      "emit": "spike",
      "assign": [
        "v = v_reset",
        "refractory_remaining_ms = t_ref_ms"
      ]
    }
  },
  "refractory": {
    "active_when": "refractory_remaining_ms > 0",
    "countdown": "refractory_remaining_ms -= dt_ms"
  }
}
```

固定执行顺序：

```text
1. 清空并读取 pending synaptic/current input
2. 求值 subexpression
3. 按指定 method 积分 ODE
4. 处理 refractory countdown
5. 检查 spike condition
6. 应用 reset/event assignments
7. 写回 state 并发放 spike
```

一个事件块内的 assignment 从上到下执行；不同事件的优先级必须通过 `priority` 明确声明，默认拒绝同一 step 多个未排序写入。

## 9. 主网神经元完整示例

```json
{
  "schema_version": 1,
  "kind": "custom_time_driven_neuron",
  "variant_id": 10001,
  "implementation_name": "CustomLifConductanceV1",
  "extends": {
    "base_class": "CustomTimeDrivenNeuronModel",
    "base_contract_version": 1,
    "base_contract_digest": "sha256:example"
  },
  "api": {
    "constructor": "custom_lif_conductance_v1",
    "visibility": "experimental",
    "summary": "Conductance-based LIF neuron defined by ODEs."
  },
  "equations": "dv/dt = (v_rest - v + g_exc*(e_exc-v) + g_inh*(e_inh-v) + i_input) / tau_m : mV\ndg_exc/dt = -g_exc / tau_exc : uS\ndg_inh/dt = -g_inh / tau_inh : uS\ntau_m : ms (parameter, positive, constant)\ntau_exc : ms (parameter, positive, constant)\ntau_inh : ms (parameter, positive, constant)\nv_rest : mV (parameter, constant)\nv_reset : mV (parameter, constant)\nv_threshold : mV (parameter, constant)\ne_exc : mV (parameter, constant)\ne_inh : mV (parameter, constant)\nt_ref_ms : ms (parameter, positive, constant)",
  "inputs": [
    {"name": "g_exc", "kind": "excitatory_conductance", "clear_on_read": true},
    {"name": "g_inh", "kind": "inhibitory_conductance", "clear_on_read": true},
    {"name": "i_input", "kind": "current", "clear_on_read": true}
  ],
  "events": {
    "spike": {
      "when": "v >= v_threshold and not_refractory",
      "emit": "spike",
      "assign": ["v = v_reset", "refractory_remaining_ms = t_ref_ms"]
    }
  },
  "refractory": {"active_when": "refractory_remaining_ms > 0"},
  "integration": {"method": "forward_euler", "dt_unit": "ms"},
  "validation": [
    {"assert": "tau_m > 0 and tau_exc > 0 and tau_inh > 0", "message": "time constants must be positive"}
  ],
  "reference": {"abs_tolerance": 1e-5, "rel_tolerance": 1e-5, "fixture": "lif_conductance_default"}
}
```

## 10. 学习规则方程与事件

学习规则的连续 trace 仍使用 ODE，但权重更新通常发生在离散事件时。主网 rule 的事件块为 `on_pre`、`on_post`、`on_trigger`；Dense rule 使用同名语义，但由独立 Dense backend emitter 生成。

```json
{
  "schema_version": 1,
  "kind": "custom_learning_rule",
  "variant_id": 20001,
  "implementation_name": "CustomRStdpV1",
  "extends": {
    "base_class": "CustomLearningRuleModel",
    "base_contract_version": 1,
    "base_contract_digest": "sha256:example"
  },
  "equations": "dx_pre/dt = -x_pre/tau_pre : 1 (event_driven)\ndx_post/dt = -x_post/tau_post : 1 (event_driven)\ndeligibility/dt = -eligibility/tau_eligibility : 1\ntau_pre : ms (parameter, positive, constant)\ntau_post : ms (parameter, positive, constant)\ntau_eligibility : ms (parameter, positive, constant)\na_plus : 1 (parameter, constant)\na_minus : 1 (parameter, constant)",
  "events": {
    "on_pre": {
      "assign": [
        "weight = clamp(weight + a_plus*x_post, w_min, w_max)",
        "eligibility += x_post",
        "x_pre += 1"
      ]
    },
    "on_post": {
      "assign": [
        "weight = clamp(weight - a_minus*x_pre, w_min, w_max)",
        "eligibility -= x_pre",
        "x_post += 1"
      ]
    },
    "on_trigger": {
      "assign": ["weight = clamp(weight + reward*eligibility, w_min, w_max)", "eligibility = 0"]
    }
  },
  "integration": {"method": "forward_euler", "dt_unit": "ms"},
  "reference": {"abs_tolerance": 1e-6, "rel_tolerance": 1e-6, "fixture": "rstdp_reward_trace"}
}
```

## 11. 名称、作用域与外部量

变量按以下作用域解析：

1. state、parameter、subexpression。
2. `inputs` 声明的输入量。
3. 事件上下文内置量：`time_ms`、`dt_ms`、`pre_index`、`post_index`、`synapse_index`、`weight`、`reward`。
4. 显式注册的纯函数和数学常量。

不存在隐式 Python locals/globals 查找。未知标识符、同名变量、保留名冲突、循环子表达式依赖、跨作用域写入均为生成时错误。

保留名包括：`time_ms`、`dt_ms`、`not_refractory`、`weight`、`pre_index`、`post_index`、`synapse_index`、所有以 `_` 开头的名称，以及 C++/CUDA emitter 的保留符号。

## 12. 编译管线

```text
.nbmodel.json
    -> JSON envelope validation
    -> equation lexer/parser
    -> typed equation AST + dimension graph
    -> dependency, unit, event-order and backend validation
    -> Equation IR
    -> C++ Custom* derived class / CUDA static entry / Python reference
    -> differential baseline and wheel packaging
```

生成器输出到 CMake build directory。最终用户安装 neuronbridge wheel 时只得到已经编译好的 native extension 和生成的 Python schema，不需要 Python parser、Jinja、SymPy 或 C++ build tools。

## 13. 必须拒绝的格式

以下内容必须在生成阶段失败：

```text
dv/dt = np.sin(v) : mV                 # 任意 Python namespace
dv/dt = user_function(v) : mV          # 未注册函数
dv/dt = v + 1*nA : mV                  # 单位不匹配
dx/dt = -x/tau : 1 (event_driven)      # 若 x 依赖 continuous state，则不允许
weight = weight + 1                    # 方程块外对 rule state 直接写入
for i in range(N): ...                  # 循环或任意代码
```

## 14. 与 Brian2 的关系

借鉴项：

- `dx/dt = expression : unit` 的方程可读性。
- ODE、子表达式、参数的分层。
- threshold/reset/refractory 与突触事件的分离。
- flags、单位检查与 event-driven 方程的依赖限制。
- 从解析后的方程表示生成不同后端代码。

不借鉴项：

- 不隐式捕获 Python namespace。
- 不允许任意 Python、NumPy、C++ 或 CUDA 代码进入方程字符串。
- 不直接兼容或执行 Brian2 文件。
- 不在第一阶段实现 Brian2 全部 flags、随机方程、standalone device 或符号积分器。

Brian2 的方程语言和 event-driven 限制可作为用户理解本规范的数学参照，但 NeuronBridge 的接受语法和 backend 能力以本文件为准。[Brian2 equation reference](https://brian2.readthedocs.io/en/2.7.1/user/equations.html)

## 15. 实施状态

本规范是下一阶段实现 `Custom*` base、equation parser 和 code generator 的输入契约。当前项目中尚未创建上述通用基类、parser 或 `.nbmodel.json` 文件；现有 C++ 与 Python API 均不受本文件影响。

运行时类、生成产物、CMake target 和性能门禁的具体设计见 [自定义方程模型具体实现设计](custom_model_codegen_implementation_design.md)。
