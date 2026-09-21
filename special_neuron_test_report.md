# 特殊神经元测试报告

日期：2026-09-18

测试 wheel：`neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`

## 测试覆盖

- `TriggerRelayNeuronModel`：legacy CPU、legacy GPU
- `EDLUTLikeLIF`：legacy GPU 专用实现
- `CustomLifConductanceV1`：legacy CPU、legacy GPU、Dense GPU
- 模型支持矩阵：InputSpike、InputCurrent、TriggerRelay、EDLUTLikeLIF、CustomLifConductanceV1

## 结果

| 模型 | 后端 | 结果 | 观察 |
|---|---|---|---|
| TriggerRelayNeuronModel | legacy CPU | PASS | 4 个外部 spike 均被 relay 输出 |
| TriggerRelayNeuronModel | legacy GPU | PASS | 4 个外部 spike 均被 relay 输出 |
| EDLUTLikeLIF | legacy GPU | PASS | 80 步产生 8 个 spike，状态正常推进 |
| CustomLifConductanceV1 | legacy CPU | PASS | 状态推进到 `last_update=60`，膜电位约 `-58.0921` |
| CustomLifConductanceV1 | legacy GPU | PASS | 电压与 CPU 一致，膜电位约 `-58.0921` |
| CustomLifConductanceV1 | Dense GPU | PASS | snapshot 正常，膜电位约 `-58.0921` |

## 参数问题记录

第一次 CustomLifConductance 测试传入 `t_ref=0.0`，C++ 模型要求整数，触发 `boost::bad_any_cast`。改为整数 `t_ref=50` 后测试通过。这是测试参数类型错误，不是 wheel 实现缺陷。

完整结果：[special_neuron_results.json](D:\testfile\special_neuron_tests\special_neuron_results.json)

测试脚本：[special_neuron_tests.py](D:\testfile\repro\special_neuron_tests.py)
