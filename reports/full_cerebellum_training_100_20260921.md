# 完整小脑模型 100 轮训练报告

日期：2026-09-21  
网络：47,400 个神经元，36,001,600 条主网突触  
构建路径：`.nbnet` / `streaming_direct`  
控制对象：内置 `StrictMatlabPlanarArm2DOFOuterDynamic`

## 结论

完整小脑模型已经完成 100 轮连续训练。网络只进行一次流式构建；轮间调用 `reset(preserve_weights=True)`，重置事件队列、神经动态状态和机械臂状态，同时保留 GC→PC 学习权重。每轮包含 48 个控制窗口，每窗口 200 个仿真步，共执行 960,000 个训练步和 4,800 个闭环控制窗口。

训练通过完整性检查。第 1 轮平均末端误差为 0.145190 m，第 100 轮为 0.010639 m，下降 92.67%；最佳结果出现在第 98 轮，平均误差为 0.005766 m。100 轮线性趋势为 -0.0003269 m/epoch。

## 关键结果

| Epoch | 平均误差 (m) | RMSE (m) |
|---:|---:|---:|
| 1 | 0.145190 | 0.162017 |
| 2 | 0.077307 | 0.093740 |
| 5 | 0.074446 | 0.104819 |
| 10 | 0.013581 | 0.016304 |
| 20 | 0.010116 | 0.012363 |
| 50 | 0.010791 | 0.012041 |
| 75 | 0.010899 | 0.013985 |
| 98（最佳） | 0.005766 | 0.006566 |
| 100 | 0.010639 | 0.012724 |

- 前 10 轮平均误差：0.057421 m；
- 后 10 轮平均误差：0.009515 m；
- 第 100 轮/第 1 轮误差比：0.07327；
- 初始 256 点 GC→PC 权重采样均值：1.841504；
- 最终 256 点 GC→PC 权重采样均值：1.839422；
- 最终权重文件：249,916,997 bytes；
- 权重 SHA-256：`1336E7D089CC14B9507870AF4A363CC4087B37FB337A96791BCC27B84C199B6D`。

逐轮误差存在波动，但长期趋势明确下降。第 1 轮轨迹明显偏离目标；第 98 轮和第 100 轮轨迹已基本贴合目标曲线。第 100 轮并非最佳轮，因此后续用于部署时可考虑在训练期间按验证误差保存最佳权重，而不只保存最后一轮权重。

## 性能

| 阶段 | 耗时/内存 |
|---|---:|
| C++ 流式构造 | 18.507 s |
| 100 轮 epoch 循环 | 15.265 s |
| 训练及最终保存流程 | 26.993 s |
| 最终权重保存 | 11.499 s |
| 峰值工作集 | 7,353,659,392 bytes（约 6.85 GiB） |

## 产物

- 完整 JSON 报告：`artifacts/full_cerebellum_training_100/training_report.json`
- 4,800 个控制窗口轨迹：`artifacts/full_cerebellum_training_100/trajectory.jsonl`
- 最终权重：`artifacts/full_cerebellum_training_100/weights_final.bin`
- 训练汇总图：`artifacts/full_cerebellum_training_100/training_summary.png`
- SVG 汇总图：`artifacts/full_cerebellum_training_100/training_summary.svg`
- 可重复训练脚本：`tests/streaming_build/train_full_cerebellum_streaming.py`
- 绘图脚本：`tests/streaming_build/plot_full_cerebellum_training.py`
