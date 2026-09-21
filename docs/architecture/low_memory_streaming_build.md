# NeuronBridge 双构建路径与低内存流式构建设计

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

新增文件入口：

```python
options = nb.LowMemoryBuildOptions(
    memory_budget_mb=128,
    mmap=True,
    verify_checksum=True,
    release_build_buffers=True,
)

sim = nb.Simulation.from_network_file(
    "large_network.nbnet",
    nb.SimulationConfig(
        steps=10_000,
        timestep=0.1,
        queues=8,
    ),
    build_options=options,
)
```

适用场景：

- 百万级及以上突触；
- 内存受限设备；
- 固定网络重复加载；
- 大型 dense/GPU 或混合网络；
- 更重视构建峰值内存，而不是首次生成网络文件的速度。

该路径由 `StreamingNetworkBuilder` 实现，不经过完整的 Python `Network` 和 `NativeNetworkDescription.connections`。

### 4.3 不自动替换现有默认行为

`nb.Simulation(network, config)` 必须保持快速构建。不能因为网络达到某个隐藏阈值而在后台自动写文件或改变构建语义。

如后续提供 `build_mode="auto"`，只能作为显式选择：

```python
sim = nb.Simulation.from_network_file(
    path,
    config,
    build_mode="auto",
)
```

如果用户已经在 Python 内存中构造完整网络，再自动切换到流式构建通常已经无法避免 Python 对象带来的内存开销。

## 5. 增量文件写入接口

为避免用户先创建巨大的 Python 连接列表，新增增量 writer：

```python
with nb.NetworkFileWriter(
    "large_network.nbnet",
    memory_budget_mb=128,
) as writer:
    writer.add_layer(...)
    writer.add_learning_rule(...)
    writer.add_outer_dynamic(...)
    writer.add_input_conv(...)

    for batch in generate_connections(batch_size=100_000):
        writer.append_connections(
            source=batch.source,
            target=batch.target,
            synapse_type=batch.synapse_type,
            weight=batch.weight,
            max_weight=batch.max_weight,
            delay=batch.delay,
            synapse_rule=batch.synapse_rule,
            trigger_rule=batch.trigger_rule,
        )

    writer.finalize()
```

`append_connections()` 应支持：

- NumPy 连续数组；
- Python buffer protocol；
- 迭代器产生的固定大小批次；
- 明确的整数和浮点宽度；
- 批次级校验，不创建逐连接 Python 对象。

如果输入连接未按运行时顺序排列，writer 使用受 `memory_budget_mb` 限制的外部归并排序，而不是在内存中一次性排序全部连接。

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

两种构建入口最终生成相同的运行时抽象：

```text
Network + Connection objects      .nbnet file
             |                         |
             v                         v
     FastNetworkBuilder       StreamingNetworkBuilder
             |                         |
             +------------+------------+
                          v
                    RuntimeNetwork
                          |
                          v
                      Simulation
```

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

### 阶段 A：批量 buffer API

- 为连接增加 NumPy/buffer protocol 批量入口；
- 避免逐元素 Python 对象转换；
- 增加构建内存统计；
- 不改变现有运行时结构。

该阶段可以较快降低 Python 侧开销，但还不是真正的流式运行时构建。

### 阶段 B：`.nbnet` writer 和 reader

- 实现格式、版本、checksum 和 metadata；
- 实现分块 writer；
- 实现 mmap/分块 reader；
- 支持普通主网和无学习连接。

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

# 大型网络：显式低内存构建
sim = nb.Simulation.from_network_file(
    "network.nbnet",
    config,
    build_options=nb.LowMemoryBuildOptions(
        memory_budget_mb=128,
        mmap=True,
        verify_checksum=True,
    ),
)
```

设计原则是：保留快速路径的便利性和速度，通过独立文件入口为大型网络提供真正受内存预算约束的构建路径，二者最终共享完全一致的仿真运行时语义。
