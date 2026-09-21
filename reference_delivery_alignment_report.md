# 与 delivery_cpu_only_final_bundle 的配置对齐报告

## 结论

成功 bundle 的交付配置默认运行模式是推理模式，但这不表示该项目没有经过训练：

```cpp
inline constexpr bool TRAINING_ENABLED = false;
```

其最终交付程序会加载训练后权重文件；当前任务不读取该权重，而是使用其训练配置从随机初始化重新训练。

## 主要配置差异

| 配置项 | 成功 bundle | 原当前流程 | 已对齐方式 |
|---|---:|---:|---|
| 输入 phase 数 | 100 | 64 | Python reference 模式使用 100 |
| 每个 phase 步数 | 3000 | 10000 | Python reference 模式使用 3000 |
| 每个环神经元数 | 32 | 32 | 一致 |
| 环数量 | 2 | 2 | 一致 |
| timestep | 0.1 ms | 0.1 ms | 一致 |
| 通信间隔 | 1000 | 最近为 200 | reference 模式恢复为 1000 |
| 输入脉冲 | phase 起点+10，每 100 步 | 相同机制 | 一致 |
| 输入权重 | seed 20260425，抖动 ±0.35 | 相同机制 | 一致 |
| 环内抑制 | 1.2 到 5.0，按环距离二次变化 | 相同机制 | 一致 |
| 目标序列 | CoppeliaSim IK 实际采样 | 解析 IK 重新计算 | 已接入成功 bundle 的 100-phase 序列 |
| 训练状态 | 交付运行默认关闭；历史上经过训练 | 默认训练/随机初始化 | 当前按训练配置重新训练 |

## 已加入的 Python 对齐配置

[reference_delivery_config.py](D:/neuronBridge/reference_delivery_config.py)

启动快速训练时使用：

```powershell
python fast_train_attractor.py --reference-delivery --epochs 10
```

该模式使用 100 个 phase、每 phase 3000 步、32 个神经元/环、成功 bundle 的 100 个目标索引序列和角度范围。

## 对齐冒烟测试

结果目录：

[fast_training_reference_delivery_smoke](D:/testfile/fast_training_reference_delivery_smoke)

验证结果：

- phase 数：100
- 每 phase 步数：3000
- 通信间隔：1000
- ring size：32
- phase 控制次数：100
- 通信批次：300
- wheel 通信闭环：PASS

## 说明

成功 bundle 的预训练权重不作为本次对齐输入。本次结果只验证从随机初始化出发，在相同 phase、目标序列、通信和网络配置下能否重新获得训练趋势。
