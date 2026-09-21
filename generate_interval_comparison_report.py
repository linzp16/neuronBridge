from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    output = Path(sys.argv[1])
    paths = [Path(value) for value in sys.argv[2:]]
    rows = []
    for path in paths:
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        interval = path.name.rsplit("_", 1)[-1]
        timing = result["timing_seconds"]
        rows.append((interval, result, timing))
    rows.sort(key=lambda item: int(item[0]))
    lines = [
        "# CoppeliaSim 控制频率对比测试",
        "",
        "基线为此前 `communication_interval=1000` 的完整闭环测试：总耗时约 84.37 s，640 个通信批次，64 次奖励。",
        "",
        "| 控制间隔（步） | 总耗时（s） | 控制段（s） | 通信批次 | 反馈批次 | 奖励 | 惩罚 | 轨迹采样 | 结果 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for interval, result, timing in rows:
        lines.append(
            f"| {interval} | {timing['total_run_wall_seconds']:.2f} | "
            f"{timing['coppeliasim_control_seconds']:.2f} | "
            f"{result['received_output_batches']} | {result['feedback_batches']} | "
            f"{result['rewards']} | {result['punishments']} | "
            f"{result['trajectory_samples']} | {'PASS' if result['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "- `2000` 步：总耗时约 43.03 s，320 个通信批次，64 个反馈全部收到；52 次 reward、12 次 punishment，在速度和轨迹控制质量之间取得了比 5000 步更好的平衡。",
        "- `5000` 步：总耗时下降到约 17.79 s，约为基线的 21.1%，但 64 个 phase 中只有 31 个得到 reward，33 个为 punishment，说明控制更新过稀导致跟踪精度明显下降。",
        "- `10000` 步：总耗时下降到约 9.35 s，但只收到 63 个反馈批次，最后一个 phase 的反馈没有在仿真结束前被请求触发，因此结果为 FAIL。",
        "- 降低控制频率确实有效降低 Remote API 耗时，但当前每个 phase 需要足够的关节更新次数才能保持轨迹误差在 reward 阈值内。",
        "- 当前不建议直接采用 5000 或 10000 步作为最终训练配置；更合适的优化方向是合并 Remote API 调用、去掉固定 sleep，并保持较高的控制更新频率。",
    ]
    path = output / "control_interval_comparison.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
