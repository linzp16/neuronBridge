"""Release-wheel acceptance for learning rules, OuterDynamic, and plotting APIs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import traceback

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import neuronbridge as nb


RULES = (
    "STDP",
    "R_STDP",
    "AdditiveKernalChange",
    "CerebullarLearningRule",
    "CustomPairStdpV1",
    "CustomRStdpV1",
    "CustomRStdpPersistentV1",
)
TRIGGER_RULES = {
    "R_STDP",
    "AdditiveKernalChange",
    "CerebullarLearningRule",
    "CustomRStdpV1",
    "CustomRStdpPersistentV1",
}


def explicit_lif(count: int, *, output: bool = False, dense_name: str | None = None) -> nb.NeuronLayer:
    parameters = {
        "V_rest": nb.float32(-60.0),
        "V_reset": nb.float32(-60.0),
        "V_th": nb.float32(-58.0),
        "tau": nb.float32(20.0),
        "R": nb.float32(1.0),
        "t_ref": nb.int32(1),
        "gexc_tau": nb.float32(5.0),
        "ginh_tau": nb.float32(10.0),
        "Eexc": nb.float32(0.0),
        "Einh": nb.float32(-80.0),
    }
    if dense_name:
        parameters["dense_subnetwork_name"] = dense_name
    return nb.NeuronLayer(
        "TimeDrivenLIF_Exponential_double", count, output=output, monitored=True, parameters=parameters
    )


def rule_parameters(name: str) -> dict:
    if name == "AdditiveKernalChange":
        return {
            "a1pre": nb.float32(0.2),
            "a2prepre": nb.float32(-0.05),
            "LTP_tau": nb.float32(16.8),
        }
    if name == "CerebullarLearningRule":
        return {
            "a1pre": nb.float32(0.001),
            "a2prepre": nb.float32(-0.08),
            "initpos": nb.float32(0.0),
            "maxpos": nb.float32(20.0),
            "kernel_step_size": nb.float32(1.0),
            "min_weight": nb.float32(0.1),
            "random_seed": nb.int32(17),
        }
    return {}


def build_rule_network(name: str) -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer.input_spike(2))
    network.add_layer(explicit_lif(1, output=True))
    network.add_learning_rule(nb.LearningRule(name, parameters=rule_parameters(name)))
    network.connect(
        nb.Connection(source=0, target=2, weight=8.0, max_weight=20.0, delay=1, synapse_rule=0)
    )
    if name in TRIGGER_RULES:
        network.connect(
            nb.Connection(source=1, target=2, weight=0.0, max_weight=0.0, delay=1, trigger_rule=0)
        )
    return network


def run_rule(name: str, mode: str, output_dir: Path) -> dict:
    network = build_rule_network(name)
    config = nb.SimulationConfig(steps=40, timestep=1.0, event_queue="timing_wheel", timing_wheel_size=64)
    if mode == "fast":
        sim = nb.Simulation(network, config)
    else:
        path = output_dir / f"rule_{name}_{mode}.nbnet"
        nb.nbnet.from_network(network, path, options=nb.NbnetWriteOptions(overwrite=True))
        sim = nb.Simulation(path, config, build_options=nb.StreamingBuildOptions(memory_budget_mb=8))
    sim.init()
    before = sim.get_connection_weight(0)
    sim.add_external_spikes(
        [1, 4, 8, 11, 15, 18, 22, 25, 29, 32],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ).run()
    after = sim.get_connection_weight(0)
    return {
        "rule": name,
        "mode": mode,
        "before": before,
        "after": after,
        "finite": math.isfinite(after),
        "output_spikes": len(sim.output_spikes()),
        "build_mode": sim.build_stats["mode"],
    }


def build_outer(model: str) -> nb.Network:
    network = nb.Network()
    network.add_layer(nb.NeuronLayer("InputSpikeNeuronModel", 4, output=True))
    if model == "counter":
        outer = nb.OuterDynamic.spike_counter(name="counter", slot_count=2, type_count=2)
    elif model == "planar":
        outer = nb.OuterDynamic.planar_arm_2dof(name="arm", update_timestep=5, communication_interval=5)
    else:
        outer = nb.OuterDynamic.strict_matlab_planar_arm_2dof(
            name="arm", update_timestep=5, communication_interval=5
        )
    network.add_outer_dynamic(outer)
    network.connect_outer_dynamic(
        nb.OuterDynamicConnection(
            source=[0, 1, 2, 3],
            target_outer_dynamic=0,
            target_joint=[0, 0, 1, 1],
            synapse_type=[0, 1, 0, 1],
            weight=[0.1, 0.1, 0.05, 0.05],
            delay=1,
        )
    )
    return network


def run_outer(model: str, mode: str, output_dir: Path) -> dict:
    network = build_outer(model)
    config = nb.SimulationConfig(steps=30, timestep=1.0, event_queue="heap")
    if mode == "fast":
        sim = nb.Simulation(network, config)
    else:
        path = output_dir / f"outer_{model}_{mode}.nbnet"
        nb.nbnet.from_network(network, path, options=nb.NbnetWriteOptions(overwrite=True))
        sim = nb.Simulation(path, config, build_options=nb.StreamingBuildOptions(memory_budget_mb=8))
    sim.init().add_external_spikes([1, 2, 6, 7, 11, 12], [0, 2, 1, 3, 0, 2]).run()
    if model == "counter":
        state = sim.outer_dynamic_spike_counter_snapshot("counter")
        observed = sum(state["spike_counts"])
    else:
        state = sim.outer_dynamic_state()
        observed = len(state["q"])
    return {
        "model": model,
        "mode": mode,
        "observed": observed,
        "state": state,
        "build_mode": sim.build_stats["mode"],
    }


def write_plot_tables(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "spikes.csv").write_text(
        "time_step,global_neuron_id,local_neuron_id\n0,0,0\n2,1,1\n4,0,0\n", encoding="utf-8"
    )
    (path / "neuron_state.csv").write_text(
        "time_step,global_neuron_id,field_name,value\n0,0,voltage,-60\n1,0,voltage,-55\n2,0,voltage,-58\n",
        encoding="utf-8",
    )
    (path / "weights.csv").write_text(
        "time_step,component_kind,component_index,component_name,synapse_id,value\n"
        "0,MainNetwork,0,main_network,0,1.0\n0,MainNetwork,0,main_network,1,2.0\n"
        "1,MainNetwork,0,main_network,0,1.5\n1,MainNetwork,0,main_network,1,2.5\n",
        encoding="utf-8",
    )
    rows = ["time_step,component_index,component_name,field_name,index,value"]
    for step in range(4):
        for field, values in {
            "q": (0.1 * step, 0.2 * step),
            "q_des": (0.1 * step + 0.01, 0.2 * step - 0.01),
            "qv": (0.03 * step, -0.02 * step),
            "tau_total": (0.5 * step, -0.25 * step),
        }.items():
            for index, value in enumerate(values):
                rows.append(f"{step},0,arm,{field},{index},{value}")
    (path / "outer_dynamic_state.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def save_axis(axis, path: Path) -> None:
    figure = axis[0].figure if isinstance(axis, list) else axis.figure
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_plots(output_dir: Path) -> dict:
    monitor_dir = output_dir / "plot_tables"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    write_plot_tables(monitor_dir)
    result = nb.open_debug_monitor(monitor_dir)
    plotters = {
        "spike_raster": lambda: result.plot_spike_raster(),
        "state_trace": lambda: result.plot_state_trace(0),
        "weight_trace": lambda: result.plot_weight_trace("main_network", [0, 1]),
        "weight_summary": lambda: result.plot_weight_summary("main_network"),
        "weight_distribution": lambda: result.plot_weight_distribution("main_network", bins=4),
        "outer_trace": lambda: result.plot_outer_dynamic_trace(outer_dynamic="arm"),
        "outer_tracking_error": lambda: result.plot_outer_dynamic_tracking_error(outer_dynamic="arm"),
        "outer_phase_plane": lambda: result.plot_outer_dynamic_phase_plane(outer_dynamic="arm"),
        "outer_torque": lambda: result.plot_outer_dynamic_torque(outer_dynamic="arm"),
    }
    files = {}
    for name, plotter in plotters.items():
        path = plot_dir / f"{name}.png"
        save_axis(plotter(), path)
        files[name] = {"path": str(path.resolve()), "bytes": path.stat().st_size}

    network = nb.Network()
    network.add_layer(explicit_lif(4))
    network.add_input_conv(
        nb.InputConv.v1_grating(
            width=2,
            height=2,
            channels=1,
            output_source_indices=[0, 1, 2, 3],
            output_target_neuron_ids=[0, 1, 2, 3],
        )
    )
    sim = nb.Simulation(network, nb.SimulationConfig(steps=3, timestep=1.0)).init().run()
    heatmap_path = plot_dir / "inputconv_rate_heatmap.png"
    save_axis(sim.plot_input_conv_rate_heatmap(0, width=2, height=2), heatmap_path)
    files["inputconv_rate_heatmap"] = {
        "path": str(heatmap_path.resolve()), "bytes": heatmap_path.stat().st_size
    }
    return files


def helper_metadata_check() -> dict:
    layer = nb.NeuronLayer.lif_double(1, output=True, monitored=True)
    return {
        "expected_output": True,
        "actual_output": layer.output,
        "expected_monitored": True,
        "actual_monitored": layer.monitored,
        "parameters": sorted(layer.parameters),
        "pass": layer.output and layer.monitored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"cases": [], "helper_metadata": helper_metadata_check()}

    def record(category: str, name: str, function) -> None:
        try:
            detail = function()
            passed = bool(detail.get("finite", True)) and int(detail.get("observed", 1)) > 0
            report["cases"].append({"category": category, "name": name, "status": "PASS" if passed else "FAIL", "detail": detail})
        except Exception as exc:
            report["cases"].append({
                "category": category, "name": name, "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            })

    for rule in RULES:
        for mode in ("fast", "streaming"):
            record("learning", f"{rule}:{mode}", lambda rule=rule, mode=mode: run_rule(rule, mode, args.output_dir))
    for model in ("counter", "planar", "strict"):
        for mode in ("fast", "streaming"):
            record("outer_dynamic", f"{model}:{mode}", lambda model=model, mode=mode: run_outer(model, mode, args.output_dir))
    record("plotting", "all_public_plot_methods", lambda: {"files": run_plots(args.output_dir)})
    report["pass_without_known_helper_bug"] = all(item["status"] == "PASS" for item in report["cases"])
    report["pass"] = report["pass_without_known_helper_bug"] and report["helper_metadata"]["pass"]
    report_path = args.output_dir / "learning_outer_plot_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
