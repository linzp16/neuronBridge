# InputCurrent 复杂网络访问冲突调查报告

## 结论

问题不是 `add_external_currents()` 数据长度或 wheel DLL 加载问题，而是无效连接定义没有被拒绝：

```python
Connection(source=0, target=2, synapse_type=0)
```

当 source layer 是 `InputCurrentNeuronModel` 时，连接必须使用：

```python
Connection(source=0, target=2, synapse_type=3)
```

`0` 表示普通 excitatory connection，`3` 才表示 external current connection。

## C++ 调用链

1. `ArrayInputCurrentDriver::LoadInputCurrent()` 根据 neuron id 创建 `InputCurrent` 事件。
2. `InputCurrent::ProcessEvent()` 遍历 source neuron 的输出连接并创建 `PropogatedCurrent`。
3. `PropogatedCurrent::ProcessEvent()` 调用目标模型的 `ProcessCurrent()`。
4. `TimeDrivenLIF_Exponential_Decay::ProcessCurrent()` 调用：

   ```cpp
   CurrentSynapeModel->SetInputCurrentPerSynapse(
       Target->index_in_NeuronModel,
       inter->subindex_type,
       current);
   ```

5. `CurrentSynapse::SetInputCurrentPerSynapse()` 直接访问 `currents_per_connection[neuron_index][synapse_index]`，没有空指针或范围检查。

## 触发原因

在 `Network::CreateConnections()` 中，每条连接先调用目标模型的 `CheckType()`。

对于 `TimeDrivenLIF_Exponential_Decay`：

- `type == 3`：设置 `I_EXT`，分配并递增 current-synapse slot；
- `type == 0`：只设置普通 excitatory 状态，不分配 current-synapse slot。

但是 `InputCurrent::ProcessEvent()` 不检查连接类型，会把 current event 沿所有输出连接传播。于是错误的 `type=0` 连接仍然进入目标模型的 `ProcessCurrent()`，而目标模型的 current-synapse slot 未初始化，最终造成 `0xC0000005` 访问冲突。

关键代码位置：

- `src/native/legacy/source_file_realtime_v1_async/Event/src/Current/InputCurrent.cpp:13`
- `src/native/legacy/source_file_realtime_v1_async/Event/src/Current/PropogatedCurrent.cpp:15`
- `src/native/legacy/source_file_realtime_v1_async/NeuralModel/src/TimeDriven/TimeDrivenLIF_Exponential_Decay.cpp:121`
- `src/native/legacy/source_file_realtime_v1_async/NeuralModel/src/TimeDriven/TimeDrivenLIF_Exponential_Decay.cpp:137`
- `src/native/legacy/source_file_realtime_v1_async/Current/src/CurrentSynapes.cpp:48`
- `src/native/legacy/source_file_realtime_v1_async/Network/src/Network.cu:313`
- `python/src/neuronbridge/model.py:208`

## 复现和验证

错误定义：

- `D:\testfile\repro\reproduce_input_current_connection_crash.py`
- 结果：初始化成功，进入 `run()` 时退出码 `-1073741819`。

正确类型：

- `D:\testfile\repro\reproduce_input_current_connection_type3.py`
- 结果：正常完成，退出码 `0`。

## 修复建议

建议分三层修复：

1. Python API：当 source layer 是 `InputCurrentNeuronModel` 时，自动要求或默认生成 `synapse_type=3`，并对显式错误类型抛出 `ValueError`。
2. Native 构建：在 `Network::CreateConnections()` 中拒绝 `InputCurrentNeuronModel` 输出连接的非 `type=3` 定义，返回明确错误，不让非法状态进入运行阶段。
3. 防御性编程：在 `CurrentSynapse::SetInputCurrentPerSynapse()` 增加空指针和索引边界检查，避免配置错误升级为进程崩溃。

## 影响范围

该问题影响直接从 `InputCurrentNeuronModel` 建立普通 `Connection`，再通过 `add_external_currents()` 驱动的网络。使用 `synapse_type=3` 的合法 current connection 不受此复现影响。
