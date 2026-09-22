# 流式网络与训练权重迁移测试报告

## 结论

测试通过。Windows wheel 生成的 `.nbnet` 网络和训练后权重快照能够：

1. 复制到独立目录后，由新的 Windows Python 进程重新读取；
2. 由 Ubuntu 20.04 Linux wheel 直接读取；
3. 保持连接权重、继续训练结果和输出 spike 完全一致。

## 测试流程

1. 使用 `NbnetDescriptionBuilder` 增量生成网络，不创建完整 Python `Network`；
2. 网络包含 3 个神经元、2 条连接和 `AdditiveKernalChange` 学习规则；
3. 通过 `Simulation(.nbnet, ..., StreamingBuildOptions(...))` 进入 C++ 直接流式路径；
4. 注入训练 spike，使连接 0 的权重由 `8.0` 变化为 `8.401209831237793`；
5. 使用 `save_weights()` 保存训练后权重；
6. 将 `.nbnet` 和权重文件复制到独立目录并校验 SHA-256；
7. 启动新的 Windows Python 进程，只读取迁移后的两个文件；
8. 使用 Ubuntu 20.04 Linux wheel 读取同一组 Windows 生成的文件；
9. 在源端、Windows 迁移端和 Linux 迁移端注入相同测试 spike，比较权重与输出。

## 结果

| 检查项 | 源端 | Windows 迁移端 | Ubuntu 20.04 迁移端 |
|---|---:|---:|---:|
| 加载训练权重 | `8.401209831237793` | `8.401209831237793` | `8.401209831237793` |
| 测试输入后权重 | `8.802419662475586` | `8.802419662475586` | `8.802419662475586` |
| 输出 spike 数量 | 8 | 8 | 8 |
| 输出 spike 时刻 | `2,5,8,11,14,17,20,23` | 相同 | 相同 |
| 构建模式 | `streaming_file_cpp` | `streaming_file_cpp` | `streaming_file_cpp` |
| 运行时路径 | `streaming_direct` | `streaming_direct` | `streaming_direct` |

迁移后文件校验值：

- `.nbnet` SHA-256：`eaafe27a04dbc8d52f0622acdcf6593f2360be88a73a3995bbd83016ec37e392`
- 权重快照 SHA-256：`8126c556fa021375d54d8296e7c1e31c4989ad1fbf020dcc20d19923a5bf2a2a`

## 验证范围说明

本测试验证的是静态网络和训练权重迁移。膜电位、事件队列、仿真时间等动态运行状态不包含在权重快照中，因此迁移后从新的动态状态开始运行，这是当前接口的预期行为。

## 产物

- `source/trained_network.nbnet`：原始流式网络；
- `source/trained_weights.txt`：训练后权重；
- `migrated_destination/trained_network.nbnet`：迁移后的网络；
- `migrated_destination/trained_weights.txt`：迁移后的权重；
- `migrated_destination/consumer_result.json`：Windows 新进程结果；
- `migrated_destination/consumer_linux_result.json`：Ubuntu 20.04 结果；
- `migration_report.json`：完整 Windows 测试记录。
