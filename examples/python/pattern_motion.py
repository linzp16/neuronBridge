"""Migrated pattern-motion dense/InputConv example."""

from __future__ import annotations

import argparse
from pathlib import Path

import neuronbridge as nb

from _pattern_motion_builder import (
    DIRS,
    PatternMotionSpec,
    build_pattern_motion_network,
    count_layer_firings,
    make_input_conv_frames,
    winner_dir,
    write_direction_montage_bmp,
    write_stimulus_file,
    write_vector_tsv,
)
from _example_paths import artifact_dir


def _remap_dense_local_ids(snapshot: dict, local_ids: list[int]) -> list[int]:
    mapping = snapshot.get("local_to_original_neuron_ids") or []
    remapped: list[int] = []
    for local_id in local_ids:
        if 0 <= local_id < len(mapping) and int(mapping[local_id]) > 0:
            remapped.append(int(mapping[local_id]) - 1)
        else:
            remapped.append(local_id)
    return remapped


def _max_by_original_range(values: list[float], mapping: list[int], base: int, count: int) -> float:
    best = -1.0e30
    for local_id, value in enumerate(values):
        original_id = int(mapping[local_id]) - 1 if 0 <= local_id < len(mapping) and int(mapping[local_id]) > 0 else local_id
        if base <= original_id < base + count:
            best = max(best, float(value))
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the neuronbridge pattern-motion example.")
    parser.add_argument("mode", nargs="?", choices=["grating", "plaid"], default="grating")
    parser.add_argument("direction", nargs="?", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=artifact_dir("pattern_motion"))
    parser.add_argument("--full", action="store_true", help="Use the C++ example dimensions and step count.")
    parser.add_argument("--steps", type=int, default=None, help="Override the number of simulation steps.")
    parser.add_argument("--input-source", choices=["file", "queue", "zmq-async"], default="file")
    parser.add_argument("--frame-topic", default="inputconv_pattern_motion")
    return parser.parse_args()


def run_pattern_motion(
    mode: str,
    selected_block: int,
    output_dir: Path,
    *,
    full: bool = False,
    steps: int | None = None,
    input_source: str = "file",
    frame_topic: str = "inputconv_pattern_motion",
) -> dict[str, int | float | str]:
    if selected_block < 0 or selected_block >= DIRS:
        raise ValueError(f"selected direction block must be in [0, {DIRS - 1}]")
    if input_source not in {"file", "queue", "zmq-async"}:
        raise ValueError("input_source must be 'file', 'queue', or 'zmq-async'")
    spec = PatternMotionSpec.full() if full else PatternMotionSpec()
    simulation_steps = spec.simulation_steps if steps is None else int(steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    stimulus_path = write_stimulus_file(output_dir / "selected_stimulus.dat", spec, mode=mode, selected_block=selected_block)
    dynamic_input = input_source != "file"

    network = build_pattern_motion_network(spec, stimulus_path, dynamic_input=dynamic_input, stimulus_mode=mode)
    sim = nb.Simulation(network, nb.SimulationConfig(steps=simulation_steps, timestep=1.0))
    sim.init()
    sim.set_dense_subnetwork_full_firing_export_enabled(spec.dense_name)
    published_frames = 0
    published_messages = 0
    source_received_frames = 0
    source_consumed_frames = 0
    source_queued_frames = 0

    if dynamic_input:
        source_name = "pattern_motion_frames"
        frames = make_input_conv_frames(spec, mode=mode, selected_block=selected_block)
        if input_source == "queue":
            sim.add_input_conv_frames(source_name, frames)
            published_frames = len(frames)
            published_messages = len(frames)
        else:
            with nb.InputConvFramePublisher(topic=frame_topic) as publisher:
                sim.add_zmq_async_input_conv_frame_source(
                    source_name,
                    subscribe_address="127.0.0.1",
                    subscribe_port=publisher.port,
                    topic=publisher.topic,
                    max_buffered_frames_per_camera=max(32, len(frames) * 16),
                )
                sim.bind_input_conv_frame_source(0, source_name)
                for frame in frames:
                    publisher.publish_frame_until_received(
                        frame,
                        sim,
                        source_name,
                        repeat=1,
                        interval_s=0.0,
                        timeout_s=2.0,
                        poll_interval_s=0.01,
                        max_attempts=80,
                    )
                source_status = sim.input_conv_frame_source_status(source_name)
                source_received_frames = int(source_status["received_frames"])
                source_queued_frames = int(source_status["queued_frames"])
                published_frames = publisher.published_frames
                published_messages = publisher.published_messages
        if input_source != "zmq-async":
            sim.bind_input_conv_frame_source(0, source_name)
            source_status = sim.input_conv_frame_source_status(source_name)
            source_received_frames = int(source_status["received_frames"])
            source_queued_frames = int(source_status["queued_frames"])

    v1_counts = [0] * spec.v1_count
    cds_counts = [0] * spec.cds_count
    pds_counts = [0] * spec.pds_count
    pds_fs_counts = [0] * spec.pds_fs_count
    lip_counts = [0] * spec.lip_count
    v1_rate_sums = [0.0] * spec.v1_count
    v1_current_update_sums = [0.0] * spec.v1_count
    v1_update_sample_count = 0
    pds_max_membrane = -1.0e30
    pds_fs_max_membrane = -1.0e30
    pds_max_gexc = 0.0
    pds_fs_max_gexc = 0.0
    pds_max_ginh = 0.0
    pds_fs_max_ginh = 0.0

    for time_step in range(simulation_steps):
        rates = sim.input_conv_output(0)
        if rates:
            for index, value in enumerate(rates):
                v1_rate_sums[index] += value
            if time_step % spec.v1_update_interval_steps == 0:
                for index, value in enumerate(rates):
                    v1_current_update_sums[index] += value
                v1_update_sample_count += 1
        sim.run(1)
        snapshot = sim.dense_subnetwork_snapshot(spec.dense_name)
        firing_ids = _remap_dense_local_ids(snapshot, [int(value) for value in snapshot["full_firing_ids"]])
        count_layer_firings(firing_ids, v1_counts, spec.v1_base, spec.v1_count, spec.width * spec.height)
        count_layer_firings(
            firing_ids,
            cds_counts,
            spec.cds_base,
            spec.cds_count,
            spec.width * spec.height,
            spatial_major_interleaved=True,
        )
        count_layer_firings(firing_ids, pds_counts, spec.pds_base, spec.pds_count, spec.pds_pool_size)
        count_layer_firings(firing_ids, pds_fs_counts, spec.pds_fs_base, spec.pds_fs_count, spec.pds_pool_size)
        count_layer_firings(firing_ids, lip_counts, spec.lip_base, spec.lip_count, spec.lip_pool_size)

        membrane = snapshot["membrane_v"]
        gexc = snapshot["gexc"]
        ginh = snapshot["ginh"]
        mapping = snapshot.get("local_to_original_neuron_ids") or []
        pds_max_membrane = max(pds_max_membrane, _max_by_original_range(membrane, mapping, spec.pds_base, spec.pds_count))
        pds_fs_max_membrane = max(
            pds_fs_max_membrane,
            _max_by_original_range(membrane, mapping, spec.pds_fs_base, spec.pds_fs_count),
        )
        pds_max_gexc = max(pds_max_gexc, _max_by_original_range(gexc, mapping, spec.pds_base, spec.pds_count))
        pds_fs_max_gexc = max(pds_fs_max_gexc, _max_by_original_range(gexc, mapping, spec.pds_fs_base, spec.pds_fs_count))
        pds_max_ginh = max(pds_max_ginh, _max_by_original_range(ginh, mapping, spec.pds_base, spec.pds_count))
        pds_fs_max_ginh = max(pds_fs_max_ginh, _max_by_original_range(ginh, mapping, spec.pds_fs_base, spec.pds_fs_count))

    if dynamic_input:
        source_status = sim.input_conv_frame_source_status("pattern_motion_frames")
        source_received_frames = int(source_status["received_frames"])
        source_consumed_frames = int(source_status["consumed_frames"])
        source_queued_frames = int(source_status["queued_frames"])

    v1_current_update_mean = [0.0] * spec.v1_count
    if v1_update_sample_count > 0:
        scale = 1.0 / float(v1_update_sample_count)
        v1_current_update_mean = [value * scale for value in v1_current_update_sums]

    write_vector_tsv(output_dir / "v1_spike_counts.tsv", v1_counts, spec.width * spec.height)
    write_vector_tsv(output_dir / "v1_rate_sums.tsv", v1_rate_sums, spec.width * spec.height)
    write_vector_tsv(output_dir / "v1_current_update_mean.tsv", v1_current_update_mean, spec.width * spec.height)
    write_direction_montage_bmp(output_dir / "inputconv_rate_heatmap.bmp", v1_rate_sums, spec.width, spec.height, spec.width * spec.height)
    write_vector_tsv(output_dir / "cds_counts.tsv", cds_counts, spec.width * spec.height)
    write_vector_tsv(output_dir / "pds_counts.tsv", pds_counts, spec.pds_pool_size)
    write_vector_tsv(output_dir / "pds_fs_counts.tsv", pds_fs_counts, spec.pds_pool_size)
    write_vector_tsv(output_dir / "lip_counts.tsv", lip_counts, spec.lip_pool_size)

    summary = {
        "stimulus_mode": mode,
        "input_source": input_source,
        "frame_topic": frame_topic if input_source == "zmq-async" else "",
        "published_frames": published_frames,
        "published_messages": published_messages,
        "source_received_frames": source_received_frames,
        "source_consumed_frames": source_consumed_frames,
        "source_queued_frames": source_queued_frames,
        "generated_stimulus": str(stimulus_path),
        "selected_block": selected_block,
        "selected_direction_deg": selected_block * 45,
        "frames": spec.frames_per_direction,
        "simulation_steps": simulation_steps,
        "v1_update_sample_count": v1_update_sample_count,
        "dense_subnetwork_count": sim.dense_subnetwork_count,
        "v1_spike_winner": winner_dir(v1_counts, spec.width * spec.height),
        "v1_rate_winner": winner_dir(v1_rate_sums, spec.width * spec.height),
        "v1_current_update_mean_winner": winner_dir(v1_current_update_mean, spec.width * spec.height),
        "cds_winner": winner_dir(cds_counts, spec.width * spec.height),
        "pds_winner": winner_dir(pds_counts, spec.pds_pool_size),
        "pds_fs_winner": winner_dir(pds_fs_counts, spec.pds_pool_size),
        "lip_winner": winner_dir(lip_counts, spec.lip_pool_size),
        "pds_max_membrane": pds_max_membrane,
        "pds_fs_max_membrane": pds_fs_max_membrane,
        "pds_max_gexc": pds_max_gexc,
        "pds_fs_max_gexc": pds_fs_max_gexc,
        "pds_max_ginh": pds_max_ginh,
        "pds_fs_max_ginh": pds_fs_max_ginh,
    }
    with (output_dir / "summary.txt").open("w", encoding="utf-8", newline="\n") as output:
        for key, value in summary.items():
            output.write(f"{key}={value}\n")
    try:
        import matplotlib.pyplot as plt

        sim.plot_input_conv_rate_heatmap(0, width=spec.width, height=spec.height)
        plt.tight_layout()
        plt.savefig(output_dir / "inputconv_rate_heatmap.png", dpi=160)
        plt.close()
    except ImportError:
        pass
    return summary


def main() -> None:
    args = parse_args()
    summary = run_pattern_motion(
        args.mode,
        args.direction,
        args.output_dir,
        full=args.full,
        steps=args.steps,
        input_source=args.input_source,
        frame_topic=args.frame_topic,
    )
    print(f"Saved pattern-motion outputs to: {args.output_dir}")
    for key in (
        "stimulus_mode",
        "input_source",
        "frame_topic",
        "published_frames",
        "published_messages",
        "source_received_frames",
        "source_consumed_frames",
        "source_queued_frames",
        "selected_block",
        "selected_direction_deg",
        "simulation_steps",
        "v1_update_sample_count",
        "dense_subnetwork_count",
        "v1_spike_winner",
        "v1_rate_winner",
        "v1_current_update_mean_winner",
        "cds_winner",
        "pds_winner",
        "pds_fs_winner",
        "lip_winner",
    ):
        print(f"{key}={summary[key]}")


if __name__ == "__main__":
    main()
