# legacy GPU LIF 外部电流问题定位

## 结论

`TimeDrivenLIF_Exponential_Decay_GPU` 和 `TimeDrivenLIF_Voltage_jump_GPU` 的外部电流并非没有进入 GPU kernel。实际问题是两个 GPU 接口的 `UpdateState()` 没有执行 GPU 状态到 CPU 状态向量的 D2H 回传，因此 Python `neuron_state()` 读取到的是旧的初始化值。

## 复现证据

低电流曲线测试中，GPU LIF 的 `neuron_state()` 保持不变，看起来像没有响应电流。但将电流提高到 `1000` 后：

```text
TimeDrivenLIF_Exponential_Decay_GPU: output spikes = 4
TimeDrivenLIF_Voltage_jump_GPU:      output spikes = 4
```

这证明 GPU kernel 已经完成积分并产生了放电；同时 `neuron_state()` 仍显示 `last_update=0` 和初始膜电位，说明问题是状态可见性/回传，而不是电流事件完全丢失。

## 源码原因

两个问题模型的 `UpdateState()` 只执行：

1. `AuxStateCPU -> AuxStateGPU` 输入拷贝；
2. GPU kernel 更新；
3. `InternalSpikeGPU -> InternalSpikeCPU` 拷贝；
4. 清理输入缓冲区。

但在同步后缺少以下回传：

```cpp
cudaMemcpyAsync(State_GPU->Vector_of_StateVariable,
                State_GPU->Vector_of_StateVariableGPU, ...,
                cudaMemcpyDeviceToHost, computeStream);
cudaMemcpyAsync(State_GPU->LastUpdate,
                State_GPU->LastUpdateGPU, ...,
                cudaMemcpyDeviceToHost, computeStream);
cudaMemcpyAsync(State_GPU->LastSpike,
                State_GPU->LastSpikingGPU, ...,
                cudaMemcpyDeviceToHost, computeStream);
```

对比之下，`TimeDrivenLIF_Exponential_double_GPU_Interface::UpdateState()` 已在 `StateVector->IsMonitored` 分支中完成上述三类 D2H 回传，因此其 CPU 可见曲线正常。

## 相关源码

- `NeuralModel/src/TimeDrivenGPU/TimeDrivenLIF_Exponential_Decay_GPU_Interface.cu:70-84`
- `NeuralModel/src/TimeDrivenGPU/TimeDrivenLIF_Voltage_jump_GPU_Interface.cu:70-84`
- 对照实现：`NeuralModel/src/TimeDrivenGPU/TimeDrivenLIF_Exponential_double_GPU_Interface.cu:188-208`
- 外部电流写入：两个问题模型的 `ProcessCurrent()`，约在 `:60-64`
- GPU kernel 读取电流：对应 `.cuh` 中 `UpdateState()` 和 `CaculateDifferentialEquation()`

## 另一个独立差异

Voltage Jump GPU 的初始化使用 `V_rest`：

```cpp
float new_init[] = {this->V_rest + this->init[0], this->init[1]};
```

而 CPU Voltage Jump 使用 `V_reset` 初始化。因此 CPU/GPU 初始膜电位也不一致，需要单独统一。

## 修复建议

在两个问题模型的 `UpdateState()` 中，和 Double/Izhikevich GPU 实现一样，在 kernel 同步前加入 `Vector_of_StateVariable`、`LastUpdate`、`LastSpike` 的 D2H 拷贝，并确保这些拷贝在同一个 compute stream 上完成。然后重新执行三后端曲线和高电流 spike 回归测试。
