# 公开示例验收测试报告

测试目录：`D:\testfile\10_examples`

## 结果

第一组 9 个不依赖外部数据的公开示例全部通过：

- `phase2_network_description.py`
- `my_first_app.py`
- `phase3_run_simulation.py`
- `dense_subnetwork_smoke.py`
- `dense_subnetwork_export.py`
- `dense_mixed.py`
- `dense_mixed_current.py`
- `dense_run_no_debug.py`
- `inputconv_poisson_dense.py`

扩展示例全部通过：

- `pattern_motion.py grating 0 --input-source queue`
- `pattern_motion.py plaid 1 --input-source queue`
- `zmqcommunication_client.py --server`
- 小规模 `cerebellum_framework.py`

## 参数校验记录

首次执行 plaid 示例时传入方向参数 `120`，程序按设计返回：

`selected direction block must be in [0, 7]`

该失败是测试参数超出合法范围，属于正确的输入校验行为。改用合法 block `1` 后通过。

## 结果文件

- `example_results.json`
- `extended_example_results.json`
- 各示例对应的 `.log` 文件

## 结论

公开用户入口、Dense、InputConv、ZeroMQ、pattern motion 和小规模 cerebellum 示例均已在 wheel 环境下通过。未发现 wheel 功能性问题。
