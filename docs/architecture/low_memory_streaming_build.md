# NeuronBridge 双构建路径与低内存流式构建设计

## 当前落地状态（2026-09-21）

本文同时记录最终目标和分阶段实现。当前已经完成 C++ 原生解析，以及主网、dense/GPU、跨子网路由、OuterDynamic 和 InputConv 的直接流式构建：

- 保留 `Simulation(Network, config)` 快速构建；
- 新增 `Simulation(path, config, build_options=...)` 路径重载；
- 新增独立的 `neuronbridge.nbnet` 描述文件生成 helper；
- `.nbnet` 使用版本化小端格式、JSON 静态元数据、固定宽度二进制连接记录和 SHA-256；
- `NbnetDescriptionBuilder.append_connections()` 逐批写盘，不在 builder 中保存完整连接图；
- 支持 `from_network()`、`from_batches()`、`inspect()`、`validate()`、`is_valid()`；
- 文件加载不调用 `Network.to_native()` 或 Python `_native_description()`；
- C++ `NbnetReader` 负责 header、边界/溢出检查、SHA-256、metadata 和定长连接记录；
- pybind 路径构造重载把路径直接交给 C++，按 `memory_budget_mb` 分批读取并支持 mmap；
- 无 dense/OuterDynamic/InputConv 的主网使用两遍扫描：第一遍统计学习状态，第二遍直接填充最终 `Interconnections[]`；
- 原始连接索引在直接构建时保持为 `wordination` 索引，可继续使用权重读写 API；
- dense/GPU、跨子网路由、OuterDynamic 和 InputConv 均进入 C++ 直接流式构建路径；
- 已实现 fast/streaming 数值一致性、学习规则/原始索引、dense 跨子网、OuterDynamic/InputConv、损坏文件校验和独立进程峰值内存测试。

当前路径不再恢复完整 `std::list<ConnectionDescription>`。纯主网直接对 `NbnetReader` 做两遍扫描；混合网络首先以受预算限制的批次构建 dense 最终布局和边界规划，然后由过滤/重映射连接源直接构建主网。只有 main→dense、OuterDynamic 和 InputConv 所必需的合成接口连接会短暂保存在 staging 中。所有 `.nbnet` 构建均报告 `build_stats["runtime_build_path"] == "streaming_direct"`。

当前 Windows 基准（4096 神经元、16 MiB staging、mmap+checksum）：

| 网络 | 突触数 | fast 峰值工作集 | `streaming_direct` 峰值 | 降幅 | streaming 构建耗时 |
|---|---:|---:|---:|---:|---:|
| 主网 | 500,000 | 304.5 MB | 133.5 MB | 56.1% | 0.181 s |
| 主网 | 1,000,000 | 565.0 MB | 223.8 MB | 60.4% | 0.337 s |
| dense/GPU | 500,000 | 357.7 MB | 233.7 MB | 34.7% | 0.337 s |
| dense/GPU | 1,000,000 | 597.4 MB | 350.8 MB | 41.3% | 0.509 s |

## 1. 背景

NeuronBridge 当前通过 Python `Network` 对象描述神经元层、连接、学习规则、OuterDynamic 和 InputConv，再由 pybind11 转换为 C++ 描述结构，最终构造主网和 dense/GPU 子网。

这种方式适合小型和中型网络，接口直观且构建速度快，因此必须保留。但在超大规模网络中，构建期间可能同时存在以下数据：

1. Python `Connection` 对象及其中的 Python 数值对象；
2. `Connection.normalized()` 生成的列表副本；
3. pybind11 转换得到的 C++ `std::vector`；
4. `NativeNetworkDescription` 中的完整连接描述；
5. dense/main 分区与布局构建产生的临时数组；
6. 最终运行时神经元、突触、学习状态和 GPU 数据。

因此，构建峰值内存可能显著高于仿真稳定运行时的内存。仅增加“从 JSON 或 CSV 读取网络”的接口不能解决问题；如果读取后仍然把所有连接展开为当前描述对象，内存复制依然存在。

## 2. 设计目标

新增低内存流式构建路径，同时完整保留当前快速构建接口。

主要目标：

- 当前 `Network` API 保持默认行为，不破坏现有代码；
- 小型网络继续使用快速内存构建；
- 大型网络能够从版本化二进制文件分块读取；
- 避免构建期间同时保存多份完整连接数据；
- CPU、传统 GPU、dense/GPU、OuterDynamic 和 InputConv 功能一致；
- 两种构建路径生成等价的运行时网络；
- Windows 与 Linux 使用相同的文件格式和 Python API；
- 构建失败时不向用户暴露半初始化的 `Simulation`；
- 提供内存、吞吐量和构建阶段统计。

理想的低内存目标为：

```text
流式构建峰值内存 <= 最终运行时内存 + 用户配置的 staging 内存预算
```

首版建议默认 staging 内存预算为 128 MiB。

## 3. 非目标和物理限制

流式构建只能减少构建过程中的重复副本和临时数据，不能消除仿真运行必须持有的内容，包括：

- 神经元状态；
- 最终突触结构；
- 学习规则状态；
- 事件队列；
- 必需的 CPU/GPU 运行时镜像；
- DebugMonitor 明确要求保留的数据。

如果某一运行时后端必须在主机端永久保留完整镜像，该部分属于稳定运行内存，不计入可释放的构建 staging buffer。

## 4. 双构建路径

### 4.1 快速构建路径

保留当前 API，并继续作为默认选择：

```python
network = nb.Network()
network.add_layer(...)
network.connect(...)

sim = nb.Simulation(
    network,
    nb.SimulationConfig(steps=10_000, timestep=0.1),
)
```

适用场景：

- 小型和中型网络；
- 单元测试；
- 交互式实验；
- 需要频繁修改拓扑的网络；
- 相比内存峰值，更重视首次构建速度。

该路径由 `FastNetworkBuilder` 实现，可继续使用完整内存描述和现有优化。

### 4.2 低内存流式构建路径

`Simulation` 对外提供文件路径重载。它仍然是唯一的仿真类型，但文件重载必须进入独立的流式构建函数，不能复用当前完整描述构建过程：

```python
options = nb.StreamingBuildOptions(
    memory_budget_mb=128,
    mmap=True,
    verify_checksum=True,
)

sim = nb.Simulation(
    "large_network.nbnet",
    nb.SimulationConfig(
        steps=10_000,
        timestep=0.1,
        queues=8,
    ),
    build_options=options,
)
```

接口同时接受 `str` 和 `os.PathLike`。两者通过 `os.fspath()` 归一化后进入同一个原生流式构造函数：

```python
from pathlib import Path

sim = nb.Simulation(
    Path("large_network.nbnet"),
    config,
    build_options=options,
)
```

适用场景：

- 百万级及以上突触；
- 内存受限设备；
- 固定网络重复加载；
- 大型 dense/GPU 或混合网络；
- 更重视构建峰值内存，而不是首次生成网络文件的速度。

全部文件路径均不经过完整的 Python `Network`、`Network.to_native()`、`NativeNetworkDescription.connections` 或完整 `std::list<ConnectionDescription>`。dense/GPU 内部连接直接写入 dense runtime layout，main/dense 与 dense/dense 边界由流式分区器生成最终路由，OuterDynamic 和 InputConv 仅追加其运行时必需的合成接口。

Python 层只根据第一个参数是 `Network` 还是路径选择 pybind11 构造重载，不参与实际构建。C++ 层可以在同一个 `NativeSimulation` 上提供两个构造函数，但两个构造函数必须分别委托给：

```text
fast_build::CreateSimulation(...)
streaming_build::CreateSimulation(...)
```

两者不能相互调用，只能在构建完成后共享运行时控制代码。

### 4.3 不自动替换现有默认行为

`nb.Simulation(network, config)` 必须保持快速构建。不能因为网络达到某个隐藏阈值而在后台自动写文件或改变构建语义。

路径重载始终表示流式构建，`Network` 重载始终表示快速构建。首版不提供 `build_mode="auto"`，避免参数含义和内存行为不透明。

如果用户已经在 Python 内存中构造完整网络，再自动切换到流式构建通常已经无法避免 Python 对象带来的内存开销。

### 4.4 对外重载、对内隔离

推荐内部结构：

```text
Simulation 构造重载
├── Simulation(Network, Config)
│    └── FastSimulationBuilder::Build()
│         └── RuntimeBuildResult
│
└── Simulation(Path, Config, StreamingBuildOptions)
     └── StreamingSimulationBuilder::Build()
          └── RuntimeBuildResult

Simulation
└── 接管 RuntimeBuildResult 并提供统一运行接口
```

统一交接结构只包含构建完成后的运行时资源：

```cpp
struct RuntimeBuildResult {
    std::unique_ptr<Network> main_network;
    std::vector<std::unique_ptr<DenseSubnetworkModel>> dense_subnetworks;
    std::unique_ptr<RuntimeRoutingTable> routing;
    std::unique_ptr<RuntimeLearningState> learning_state;
    BuildStats stats;
};
```

隔离规则：

- 快速路径不得包含 `.nbnet` 解析、mmap、外部排序或 staging 内存预算；
- 流式路径不得恢复完整 `std::list<ConnectionDescription>`；
- 流式路径不得调用现有全量 `PrepareBlackBoxDenseBuild()`；
- 两条路径只能共享最终运行时结构、仿真推进、通信、DebugMonitor、权重和 reset；
- 关闭流式构建功能后，现有 `Simulation(Network, config)` 必须仍能独立编译和运行。

## 5. `.nbnet` 生成 helper 与增量写入接口

`.nbnet` 的生成、查看和验证必须独立于 `Simulation`。建议提供 `neuronbridge.nbnet` helper：

```python
import neuronbridge.nbnet as nbnet
```

当前公共接口：

```text
nbnet.from_network(...)
nbnet.from_batches(...)
nbnet.inspect(...)
nbnet.validate(...)
nbnet.is_valid(...)
nbnet.NbnetDescriptionBuilder
```

### 5.1 从现有 `Network` 生成

```python
network = nb.Network()
network.add_layer(...)
network.connect(...)

result = nb.nbnet.from_network(
    network,
    "network.nbnet",
)
```

该接口用于迁移现有项目、小型网络和 fast/streaming 等价性测试。由于完整 `Network` 已经存在于 Python 内存中，它不能解决网络描述本身的峰值内存问题。

### 5.2 从批次迭代器生成大型网络

真正低内存的 helper 接受惰性连接批次：

```python
result = nb.nbnet.from_batches(
    "large_network.nbnet",
    layers=layers,
    connection_batches=generate_connection_batches(),
    learning_rules=learning_rules,
    outer_dynamics=outer_dynamics,
    input_convs=input_convs,
    options=nb.NbnetWriteOptions(overwrite=False),
)
```

`connection_batches` 必须惰性消费。helper 每次只保留当前批次和受预算限制的排序缓冲，不能调用 `list(connection_batches)`。

### 5.3 描述文件增量 Builder

为避免用户先创建巨大的 Python 连接列表，新增增量 writer：

```python
with nb.NbnetDescriptionBuilder(
    "large_network.nbnet",
    options=nb.NbnetWriteOptions(overwrite=False),
) as builder:
    builder.add_layer(...)
    builder.add_learning_rule(...)
    builder.add_outer_dynamic(...)
    builder.add_input_conv(...)

    for batch in generate_connections(batch_size=100_000):
        builder.append_connections(
            source=batch.source,
            target=batch.target,
            synapse_type=batch.synapse_type,
            weight=batch.weight,
            max_weight=batch.max_weight,
            delay=batch.delay,
            synapse_rule=batch.synapse_rule,
            trigger_rule=batch.trigger_rule,
        )

    result = builder.finalize()
```

`from_network()` 和 `from_batches()` 都复用 `NbnetDescriptionBuilder`，不分别维护二进制编码逻辑。这个 helper 的职责是帮助用户生成静态 `.nbnet` 网络描述文件，不负责运行仿真，也不是 `Simulation` 的别名。

当前 `append_connections()` 支持有长度的 Python 序列、标量广播和 NumPy 一维数组式索引。以下能力放在后续批量编码优化中：

- NumPy 连续数组；
- Python buffer protocol；
- 迭代器产生的固定大小批次；
- 无逐元素 Python 循环的 buffer 解码；
- dtype/shape 的零拷贝校验。

如果后续要求输入连接按运行时顺序排列，builder 将使用受内存预算限制的外部归并排序，而不是在内存中一次性排序全部连接；当前 v1 文件保持用户写入顺序。

### 5.4 helper 返回值

helper 返回不可变结果对象，而不只是路径：

```python
@dataclass(frozen=True)
class NbnetBuildResult:
    path: Path
    file_bytes: int
    neuron_count: int
    connection_count: int
    checksum: str
```

更细的 section 计数、writer staging 峰值和写入耗时可在格式分区完成后追加，不能通过一次性扫描重新引入内存峰值。

### 5.5 inspect、validate 与命令行入口

```python
info = nb.nbnet.inspect("large_network.nbnet")
report = nb.nbnet.validate(
    "large_network.nbnet",
    verify_checksum=True,
)
```

同时提供：

```text
python -m neuronbridge.nbnet inspect large_network.nbnet
python -m neuronbridge.nbnet validate large_network.nbnet
```

后续可以增加 `from-csv` 和 `from-npz`。不建议使用需要一次性展开全部连接的大型 JSON 作为主要连接输入格式。

### 5.6 写入安全要求

- 默认不覆盖现有文件，除非明确设置 `overwrite=True`；
- 在目标目录写临时文件，完整校验后原子重命名；
- 失败时清理临时文件和外部排序 run；
- 检查磁盘空间、数组长度、dtype、NaN/Inf、索引范围和负 delay；
- 保留原始 connection index；
- Windows 和 Linux 写出相同字节序和格式；
- 返回实际 staging 峰值和耗时；
- Writer 关闭后禁止继续追加，`finalize()` 行为必须幂等或明确拒绝重复调用。

## 6. `.nbnet` 文件格式

### 6.1 基本要求

- 固定 magic；
- 明确格式版本；
- 固定小端序；
- 使用 `uint32_t`、`uint64_t`、`int32_t` 和 IEEE-754 `float32` 等固定宽度类型；
- 每个 section 包含 offset、长度、记录数量和 checksum；
- 对齐要求写入文件头；
- 解析前执行整数溢出和文件边界检查；
- 未识别的主版本必须拒绝加载；
- Windows 和 Linux 读取结果一致。

### 6.2 建议布局

```text
NbnetHeader
├── magic / major_version / minor_version
├── endian / alignment
├── neuron_count / connection_count
├── section_table_offset
└── global_checksum

Metadata sections
├── layers
├── learning rules
├── OuterDynamic descriptions
└── InputConv descriptions

Connection sections
├── main -> main
├── main -> dense
├── dense -> dense（同一子网内部）
├── dense -> main
├── dense -> dense（跨子网）
└── network -> OuterDynamic
```

按所有权提前分 section，可以避免加载时先创建完整连接集合，再由 `PrepareBlackBoxDenseBuild` 重新分类。

### 6.3 连接记录

概念上的连接记录如下：

```cpp
struct ConnectionRecordV1 {
    std::uint32_t source;
    std::uint32_t target;
    float weight;
    float max_weight;
    std::uint32_t delay;
    std::int32_t synapse_rule;
    std::int32_t trigger_rule;
    std::uint8_t type;
    std::uint8_t flags;
    std::uint16_t reserved;
    std::uint32_t original_index;
};
```

最终编码可以采用 AoS 或 SoA。对于 GPU/dense section，SoA 通常更适合直接上传；对于主网顺序扫描，固定记录 AoS 更简单。格式允许不同 section 声明自己的编码方式。

`original_index` 用于保持现有连接索引和权重访问语义。后续可将运行时 `wordination` 从指针数组优化为紧凑的 32/64 位索引映射。

### 6.4 排序约定

主网连接建议预排序为：

```text
target_queue -> source_neuron -> delay -> target_neuron -> original_index
```

dense 内部连接按照 dense runtime 所需的 source/delay slice 和 target 布局排序。

预排序文件可以让 loader 直接写入最终位置，避免构建期全量 `std::sort` 和额外排序副本。

## 7. C++ 流式构建流程

### 7.1 第一遍扫描：规划

第一遍只处理文件头、metadata、section 索引和必要的计数信息：

- 验证文件版本和 checksum；
- 验证神经元、学习规则、dense 子网和 OuterDynamic 索引范围；
- 统计每个连接 section 的数量；
- 统计每个队列、学习规则和 pending channel 的数量；
- 计算最终 CPU 与 GPU 分配大小；
- 使用 checked arithmetic 防止 `count * sizeof(T)` 溢出。

第一遍不创建完整连接描述。

### 7.2 一次性分配最终结构

根据第一遍结果：

- 创建神经元和模型实例；
- 分配最终主网突触数组；
- 分配学习规则状态；
- 分配 dense runtime layout；
- 分配 CSR offset 和必要索引；
- 分配不超过用户预算的 staging buffer。

### 7.3 第二遍扫描：填充

第二遍按 section 和批次读取：

- 主网连接直接构造到最终 `Interconnections[]`；
- dense 内部连接直接写入最终 dense layout；
- main/dense 边界连接直接写入接口路由结构；
- 学习规则计数和索引直接绑定；
- 当前批次处理完成后立即复用 staging buffer；
- 每个 dense 子网完成后可上传 GPU，并释放仅构建期使用的数据。

### 7.4 原子发布

构建期间使用私有 `NetworkBuildTransaction` 持有所有资源。只有完整校验和初始化成功后，才将生成的网络提交给 `Simulation`。

任何异常都必须：

- 销毁已分配的 CPU 资源；
- 释放已分配的 CUDA 资源；
- 关闭文件映射；
- 保留包含 section、记录索引和错误原因的诊断信息；
- 不留下可运行但不完整的仿真对象。

## 8. 现有内部结构需要配合优化

新增文件接口并不足以达到目标，还需要以下内部改造。

### 8.1 时间驱动神经元临时索引

当前代码为每个“模型 × 队列”分配 `neuronsNum` 大小的 CPU 和 GPU 临时索引数组。应改为：

1. 先统计每个模型/队列的实际神经元数量；
2. 按准确数量分配；
3. 第二次遍历填充。

该改动也可以降低快速构建路径的内存峰值。

### 8.2 输出连接结构

当前输出连接构建使用多层指针数组和临时计数数组。由于 `.nbnet` 已按 source/queue 排序，可以改为连续 CSR：

```text
output_offsets[neuron * queue + 1]
connection range in final Interconnections[]
```

这样可以减少大量小分配和临时指针数组。

### 8.3 学习连接结构

按 neuron/rule/kind 建立连续 offsets 和索引 arena，替代构建期多组四级指针临时结构。只为实际存在学习连接的组合分配空间。

### 8.4 dense 构建

新增 streaming dense builder，直接消费已分区的 `.nbnet` section。不能先恢复为完整 `ConnectionDescription`，再调用现有全量 `PrepareBlackBoxDenseBuild`。

现有快速路径仍可继续使用 `PreparedSimulationBuild`。

## 9. 统一运行时语义

`Simulation` 对外提供重载构造，对内由两个互相独立的 Builder 生成相同的 `RuntimeBuildResult`：

```text
Network + Connection objects      .nbnet file
             |                         |
             v                         v
 fast_build::CreateSimulation  streaming_build::CreateSimulation
             |                         |
             +------------+------------+
                          v
                  RuntimeBuildResult
                          |
                          v
                      Simulation
```

`Simulation` 只接管已经构建完成的运行时资源，不在公共初始化函数中通过 `if (streaming)` 混合两套算法。CMake 应将两条路径拆为独立 target：

```text
NeuronBridge::FastBuilder -------> NeuronBridge::Runtime
NeuronBridge::StreamingBuilder --> NeuronBridge::Runtime
```

`StreamingBuilder` 不依赖 `FastBuilder`。建议提供 `NR_ENABLE_STREAMING_BUILD`，用于验证关闭流式功能后快速路径仍能独立构建。

构建完成后，下列功能不能感知网络来自哪条路径：

- 普通运行与 realtime 运行；
- Heap 与 TimingWheel；
- 单线程与 OpenMP；
- CPU、传统 GPU 和 dense/GPU；
- 同步与异步通信；
- DebugMonitor；
- 权重读取、保存与恢复；
- `reset_state`；
- OuterDynamic 和 InputConv。

## 10. 构建统计接口

建议提供：

```python
stats = sim.build_stats
```

包含：

```python
{
    "mode": "streaming",
    "file_bytes": 0,
    "connection_count": 0,
    "peak_rss_bytes": 0,
    "steady_state_rss_bytes": 0,
    "staging_peak_bytes": 0,
    "bytes_read": 0,
    "mmap_page_faults": 0,
    "planning_seconds": 0.0,
    "allocation_seconds": 0.0,
    "fill_seconds": 0.0,
    "gpu_upload_seconds": 0.0,
    "total_seconds": 0.0,
}
```

Windows 使用进程 Working Set/Private Bytes，Linux 使用 RSS/PSS 采样。Python `tracemalloc` 不能替代原生内存统计。

## 11. 等价性测试

同一网络分别通过 fast 和 streaming 构建，并比较：

- 神经元、连接和学习规则数量；
- 原始连接索引映射；
- 初始权重和最大权重；
- CPU/GPU/dense 分区；
- 每步 spike；
- 神经元 state；
- 学习后的权重；
- DebugMonitor state、spike 和 weight 输出；
- Heap 与 TimingWheel；
- 单线程与多线程；
- 同步与异步通信绑定；
- OuterDynamic 与 InputConv 输出；
- reset 前后行为。

验收标准：

- 确定性 CPU 模型逐步完全一致；
- 浮点、GPU 和 dense 模型满足现有数值容差；
- 两种路径保存的权重能够互相加载；
- 大规模测试中 streaming 峰值满足设定内存预算；
- 损坏、截断、版本不支持和 checksum 错误的文件必须安全拒绝。

## 12. 分阶段实施

### 阶段 A：文件接口与 Python 侧低内存路径（已完成）

- 增加 `.nbnet` helper、版本、checksum、metadata 和固定记录；
- 增加 `Simulation(path, ...)` 重载；
- 文件读取与 `Network.to_native()` 隔离；
- 按预算分批读取并增加构建统计；
- 证明 Python 描述构建峰值和进程 Peak Working Set 下降；
- 保持当前原生运行时结构。

该阶段降低了 Python 描述和转换副本，但还不是真正的原生直接流式运行时构建。

### 阶段 B：批量 buffer 编码与文件分区

- 为连接增加 NumPy/buffer protocol 批量入口；
- 避免逐元素 Python 打包和解包；
- 增加构建内存统计；
- 按 main/dense/GPU/OuterDynamic 所有权写入独立 section；
- 增加外部排序和原始连接索引；
- 完成 Windows/Linux 字节级交叉读取测试。

### 阶段 C：主网直接构建

- 两遍扫描；
- 最终数组一次分配；
- CSR 输出结构；
- 学习规则流式绑定；
- 与 fast 路径进行逐步等价性测试。

### 阶段 D：dense/GPU 与混合网络

- 按所有权划分连接 section；
- dense 子网逐个构建与上传；
- 支持 main/dense、dense/main 和跨 dense 路由；
- 支持 GPU 权重、reset 和 DebugMonitor。

### 阶段 E：OuterDynamic、InputConv 和发布稳定化

- 支持所有外部动力学和卷积输入描述；
- Windows/Linux 交叉加载；
- 文件格式兼容测试和模糊测试；
- 百万级及更大网络内存基准；
- 完善文档、迁移工具和错误提示。

## 13. 最终建议 API

```python
# 小型网络：默认快速构建
sim = nb.Simulation(network, config)

# 大型网络：先通过独立 helper 生成文件
result = nb.nbnet.from_batches(
    "network.nbnet",
    layers=layers,
    connection_batches=generate_connection_batches(),
    options=nb.NbnetWriteOptions(overwrite=False),
)

# 同一个 Simulation 类型，通过路径重载进入独立流式构建函数
sim = nb.Simulation(
    "network.nbnet",
    config,
    build_options=nb.StreamingBuildOptions(
        memory_budget_mb=128,
        mmap=True,
        verify_checksum=True,
    ),
)
```

设计原则是：`Simulation` 对外重载，构建函数对内隔离，运行时统一；`.nbnet` 生成 helper 独立于仿真生命周期。这样既保留现有 API 的便利性，也能使大型网络使用真正受内存预算约束且可单独维护的构建路径。
