# 完整小脑模型流式构建验证报告

日期：2026-09-21  
运行平台：Windows 64-bit，Python 3.12.4  
Wheel：`artifacts/wheel_streaming_full_direct/neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl`

## 结论

此前因 Python 全量连接列表内存占用过高而无法成功构建的完整 2-DOF 小脑模型，已经通过 `.nbnet` 直接流式接口成功生成、构造、初始化并运行 1 个仿真步。C++ 返回的运行路径为 `streaming_direct`，mmap 与 SHA-256 校验均已生效，未出现内存非法访问。

## 模型规模

配置与 `D:/code_optimized/network_release/projects/cerebellum_framework/cerebellum_framework_arm_repro.cpp` 的默认纯主网配置一致：

| 项目 | 数值 |
|---|---:|
| `nn` | 15 |
| `nv` | 10 |
| 每个关节 GC 数量 | 22,500 |
| PC / CF / DCN（每组） | 200 / 200 / 200 |
| 总神经元 | 47,400 |
| GC→PC 学习突触 | 18,000,000 |
| GC→DCN 固定突触 | 18,000,000 |
| PC→DCN | 800 |
| CF→PC trigger | 800 |
| 主网总突触 | 36,001,600 |
| OuterDynamic 输出连接 | 800 |

模型包含 `CerebullarLearningRule`、四组 GC→PC 可塑性投射、CF trigger 输入、DCN 输出和 `StrictMatlabPlanarArm2DOFOuterDynamic`。

## 构建方式

- Python writer 每批生成 100,000 条连接，未创建完整 `Network.connections`；
- `.nbnet` 文件大小：1,152,341,737 bytes；
- C++ staging 预算：64 MiB；
- C++ 使用 mmap、checksum 校验和两遍主网扫描；
- 第一遍统计连接及学习规则状态，第二遍直接写最终 `Interconnections[]`。

## 实测结果

| 阶段 | 结果 |
|---|---:|
| `.nbnet` 生成耗时 | 131.142 s |
| 生成阶段峰值工作集 | 47,730,688 bytes（45.5 MiB） |
| checksum/reader 阶段 | 4.092 s |
| C++ Simulation 构造总耗时 | 18.091 s |
| 构造峰值工作集 | 7,356,493,824 bytes（6.85 GiB） |
| 构造完成工作集 | 5,628,370,944 bytes（5.24 GiB） |
| `Simulation.init()` | PASS |
| 1 步 smoke run | PASS，0.046 s |
| `runtime_build_path` | `streaming_direct` |
| checksum | verified |

构建峰值与构建后稳定内存的差约 1.61 GiB，峰值中已不包含数千万条 Python `int`/`float` 和完整 `ConnectionDescription` 副本。构建后的约 5.24 GiB 主要是 3600 万条突触对应的最终运行时结构、传播索引和 Cerebellar 学习状态。

## 产物

- 网络文件：`artifacts/full_cerebellum_streaming/full_cerebellum.nbnet`
- 机器可读报告：`artifacts/full_cerebellum_streaming/build_report.json`
- 可重复执行脚本：`tests/streaming_build/build_full_cerebellum_streaming.py`
- `.nbnet` 文件 SHA-256：`537060EA01EDD0B13857E4E420B10F30AA1FEA3CCBFBE0106A4F8B075EF30C0F`
- `.nbnet` 内容区校验值：`a802a3ab58615bcef5d00fb8787b8b0092a756325c233b713c44351cbafb14a6`

脚本默认就是完整规模。再次验证已有文件时可使用 `--skip-generate`，从而直接测试 C++ 构造路径。
