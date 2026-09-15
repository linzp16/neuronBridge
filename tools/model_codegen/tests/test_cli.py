"""Regression tests for the maintainer-only model code generation scaffold."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

from tools.model_codegen import cli
from tools.model_codegen.equation_ast import (
    EquationSyntaxError,
    emit_cpp,
    identifiers_in,
    parse_equations,
    parse_expression,
)
from tools.model_codegen.model_ir import compile_time_driven_neuron
from tools.model_codegen.legacy_cpu_emitter import emit_descriptor_header
from tools.model_codegen.neuron_emitter import render, render_dense
from tools.model_codegen.legacy_gpu_neuron_emitter import render_legacy_gpu
from tools.model_codegen.model_ir import ModelIrError


def _spec(implementation_name: str = "CustomLifConductanceV1") -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "custom_time_driven_neuron",
        "variant_id": 10001,
        "dense_factory_model_id": 10001,
        "implementation_name": implementation_name,
        "canonical_name": implementation_name,
        "extends": {"base_class": "CustomTimeDrivenNeuronModel"},
        "api": {"visibility": "experimental"},
        "backends": {"legacy_cpu": True, "dense_gpu": False},
        "parameters": {
            "v_threshold": {"api_name": "threshold", "default": 1.0},
            "v_reset": {"api_name": "reset", "default": 0.0},
        },
        "equations": "dv/dt = -v : volt",
        "inputs": [],
        "events": {"spike": {"threshold": "v >= v_threshold", "reset": {"v": "v_reset"}}},
        "integration": {"method": "forward_euler", "step_policy": "legacy_stride_squared"},
        "runtime": {
            "voltage_state": "v",
            "state_update": {"v": {"method": "forward_euler", "initial": "0"}},
        },
    }


class ModelCodegenCliTest(unittest.TestCase):
    def test_generator_and_cmake_do_not_name_concrete_generated_models(self) -> None:
        root = Path(__file__).resolve().parents[3]
        texts = [
            (root / "tools/model_codegen/neuron_emitter.py").read_text(encoding="utf-8"),
            (root / "tools/model_codegen/learning_rule_emitter.py").read_text(encoding="utf-8"),
            (root / "cmake/NeuronBridgeModelCodegen.cmake").read_text(encoding="utf-8"),
        ]
        for concrete_name in (
            "CustomLifConductanceV1",
            "CustomRStdpV1",
            "CustomRStdpPersistentV1",
            "CustomPairStdpV1",
        ):
            self.assertTrue(all(concrete_name not in text for text in texts), concrete_name)
        self.assertFalse((root / "tools/model_codegen/lif_runtime.py").exists())
        self.assertFalse((root / "tools/model_codegen/rstdp_runtime.py").exists())

    def test_neuron_emitter_has_no_model_identity_whitelist(self) -> None:
        path = Path(__file__).resolve().parents[3] / "models/specs/custom_lif_conductance_v1.nbmodel.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["implementation_name"] = "GeneratedConductanceProbe"
        spec["canonical_name"] = "GeneratedConductanceProbe"
        spec["variant_id"] = 10991
        spec["dense_factory_model_id"] = 10991
        header, source, _ = render(spec)
        dense, cuda = render_dense(spec)
        self.assertIn("class GeneratedConductanceProbe final", header)
        self.assertIn("GeneratedConductanceProbe::SetParameters", source)
        self.assertIn("DenseGeneratedConductanceProbe final", dense)
        self.assertIn("UpdateDenseGeneratedConductanceProbeDeviceEntry", cuda)

    def test_dense_roles_allow_renamed_neuron_symbols(self) -> None:
        path = Path(__file__).resolve().parents[3] / "models/specs/custom_lif_conductance_v1.nbmodel.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        replacements = {
            "v_rest": "rest_level",
            "tau_m": "membrane_scale",
            "v_threshold": "firing_level",
            "r": "input_gain",
            "v_reset": "reset_level",
            "tau_exc": "positive_decay",
            "e_exc": "positive_reversal",
            "tau_inh": "negative_decay",
            "e_inh": "negative_reversal",
            "t_ref": "quiet_steps",
            "g_exc": "positive_drive",
            "g_inh": "negative_drive",
            "i_input": "external_drive",
            "v": "membrane",
        }
        def rename(source: str) -> str:
            for old in sorted(replacements, key=len, reverse=True):
                source = source.replace(f"d{old}/dt", f"d{replacements[old]}/dt")
                source = re.sub(rf"\b{re.escape(old)}\b", replacements[old], source)
            return source

        spec["equations"] = rename(spec["equations"])
        spec["events"]["spike"]["threshold"] = rename(
            spec["events"]["spike"]["threshold"]
        )
        spec["events"]["spike"]["reset"] = {
            replacements.get(target, target): rename(value)
            for target, value in spec["events"]["spike"]["reset"].items()
        }
        spec["parameters"] = {
            replacements.get(name, name): declaration
            for name, declaration in spec["parameters"].items()
        }
        spec["inputs"] = [
            {**item, "name": replacements.get(item["name"], item["name"])}
            for item in spec["inputs"]
        ]
        runtime = spec["runtime"]
        runtime["voltage_state"] = replacements[runtime["voltage_state"]]
        runtime["refractory_parameter"] = replacements[runtime["refractory_parameter"]]
        runtime["state_update"] = {
            replacements.get(name, name): {
                key: replacements.get(value, value) if key == "initial" else value
                for key, value in declaration.items()
            }
            for name, declaration in runtime["state_update"].items()
        }
        runtime["dense"]["roles"] = {
            role: replacements.get(name, name)
            for role, name in runtime["dense"]["roles"].items()
        }
        header, source, _ = render(spec)
        dense, cuda = render_dense(spec)
        self.assertIn("float membrane_scale_", header)
        self.assertIn("positive_drive_decay_", source)
        self.assertIn('PendingChannel::ExcitatoryConductance, "positive_drive"', dense)
        self.assertIn('PendingChannel::Current, "external_drive"', dense)
        self.assertIn("r[index] * input_current", cuda)
        self.assertNotIn("input_gain", cuda)

    def test_cpu_emitter_supports_multiple_euler_states(self) -> None:
        spec = _spec("GeneratedTwoStateProbe")
        spec["parameters"] = {
            "threshold": {"api_name": "threshold", "default": 1.0},
            "reset": {"api_name": "reset", "default": 0.0},
            "coupling": {"api_name": "coupling", "default": 0.5},
        }
        spec["equations"] = (
            "dx/dt = y - x : 1\n"
            "dy/dt = -y + coupling * x : 1"
        )
        spec["events"] = {
            "spike": {"threshold": "x >= threshold", "reset": {"x": "reset", "y": "0"}}
        }
        spec["runtime"] = {
            "voltage_state": "x",
            "state_update": {
                "x": {"method": "forward_euler", "initial": "0"},
                "y": {"method": "forward_euler", "initial": "0"},
            },
        }
        header, source, descriptor = render(spec)
        self.assertIn("N_DifferentialStates = 2", header)
        self.assertIn("d[0]", header)
        self.assertIn("d[1]", header)
        self.assertIn("std::array<float, 2>", header)
        self.assertIn("s[0] = reset_", header)
        self.assertIn("GeneratedTwoStateProbe::SetParameters", source)
        self.assertIn("    2,\n    2,", descriptor)

    def test_multiple_neuron_specs_share_the_generic_generator(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        source_spec = json.loads(
            (project_root / "models/specs/custom_lif_conductance_v1.nbmodel.json").read_text(
                encoding="utf-8"
            )
        )
        second_spec = json.loads(json.dumps(source_spec))
        second_spec["implementation_name"] = "GeneratedConductanceProbe"
        second_spec["canonical_name"] = "GeneratedConductanceProbe"
        second_spec["variant_id"] = 10992
        second_spec["dense_factory_model_id"] = 10992

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_root = root / "specs"
            output_root = root / "generated"
            spec_root.mkdir()
            (spec_root / "first.nbmodel.json").write_text(
                json.dumps(source_spec), encoding="utf-8"
            )
            (spec_root / "second.nbmodel.json").write_text(
                json.dumps(second_spec), encoding="utf-8"
            )

            self.assertEqual(
                cli.main(
                    [
                        "--spec-root",
                        str(spec_root),
                        "--output-root",
                        str(output_root),
                        "--manifest",
                        str(output_root / "manifest.json"),
                    ]
                ),
                0,
            )

            aggregate = (output_root / "src/GeneratedNeuronModels.cpp").read_text(
                encoding="utf-8"
            )
            registry = (
                output_root / "include/neuronbridge_codegen/CustomLegacyNeuronRegistry.inc"
            ).read_text(encoding="utf-8")
            dense_models = (
                output_root / "include/neuronbridge_codegen/CustomGeneratedDenseNeuronModels.h"
            ).read_text(encoding="utf-8")
            for name in ("CustomLifConductanceV1", "GeneratedConductanceProbe"):
                self.assertIn(f'#include "{name}.cpp"', aggregate)
                self.assertIn(name, registry)
                self.assertIn(name, dense_models)

    def test_real_runtime_generation_is_deterministic_and_not_empty(self) -> None:
        spec_root = Path(__file__).resolve().parents[3] / "models/specs"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["--spec-root", str(spec_root), "--output-root", str(root),
                    "--manifest", str(root / "manifest.json")]
            self.assertEqual(cli.main(args), 0)
            first = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(cli.main(args), 0)
            self.assertEqual(first, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})
            self.assertIn(b"ForwardEulerMethod<CustomLifConductanceV1>", first[Path("src/CustomLifConductanceV1.cpp")])
            self.assertIn(b"NPGR_CUSTOM_LEGACY_NEURON", first[Path("include/neuronbridge_codegen/CustomLegacyNeuronRegistry.inc")])

    def test_comparison_operator_is_lowered_from_ast(self) -> None:
        path = Path(__file__).resolve().parents[3] / "models/specs/custom_lif_conductance_v1.nbmodel.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["events"]["spike"]["threshold"] = "v >= v_threshold"
        header, _, _ = render(spec)
        self.assertIn("s[0] >= v_threshold_", header)

    def test_dense_lif_is_generated_from_voltage_ast(self) -> None:
        path = Path(__file__).resolve().parents[3] / "models/specs/custom_lif_conductance_v1.nbmodel.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        host, cuda = render_dense(spec)
        self.assertIn("DenseCustomLifConductanceV1 final", host)
        self.assertIn("kDenseCustomLifConductanceV1TauM", host)
        self.assertIn("UpdateDenseCustomLifConductanceV1DeviceEntry", cuda)
        self.assertIn("r[index] * input_current", cuda)
        self.assertIn("if (next_v > v_threshold[index])", cuda)
        self.assertNotIn("UpdateLifExponentialDoubleDeviceEntry", cuda)
        spec["events"]["spike"] = {
            "threshold": "v >= v_threshold",
            "reset": {"v": "v_reset + 1"},
        }
        _, cuda = render_dense(spec)
        self.assertIn("if (next_v >= v_threshold[index])", cuda)
        self.assertIn("next_v = (v_reset[index] + 1.0f);", cuda)
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["equations"] = spec["equations"].replace("-g_exc / tau_exc", "-g_exc * tau_exc")
        with self.assertRaises(ModelIrError):
            render(spec)

    def test_legacy_gpu_lif_is_generated_and_registered(self) -> None:
        path = Path(__file__).resolve().parents[3] / "models/specs/custom_lif_conductance_v1.nbmodel.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        header, cuda = render_legacy_gpu(spec)
        self.assertIn("class CustomLifConductanceV1_GPU final", header)
        self.assertIn("__global__ void UpdateCustomLifConductanceV1_GPU", cuda)
        self.assertIn("expf(-dt / tau_exc_)", cuda)
        self.assertIn("state[0 * neuron_count + index] > v_threshold_", cuda)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(cli.main([
                "--spec-root", str(path.parent), "--output-root", str(output),
                "--manifest", str(output / "manifest.json")]), 0)
            registry = (output / "include/neuronbridge_codegen/CustomLegacyGpuNeuronRegistry.inc").read_text(encoding="utf-8")
            aggregate = (output / "src/GeneratedLegacyGpuNeuronModels.cu").read_text(encoding="utf-8")
            self.assertIn("CustomLifConductanceV1_GPU", registry)
            self.assertIn('#include "CustomLifConductanceV1_GPU.cu"', aggregate)

    def test_dense_neuron_defaults_and_aliases_come_from_spec(self) -> None:
        path = Path(__file__).resolve().parents[3] / "models/specs/custom_lif_conductance_v1.nbmodel.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        spec["parameters"]["tau_m"]["default"] = 27.5
        spec["parameters"]["tau_m"]["api_name"] = "membrane_tau"
        spec["parameters"]["tau_m"]["aliases"] = ["tau", "tau_m_ms"]
        dense, _ = render_dense(spec)
        self.assertIn('"tau_m_ms", DenseFieldStorage::Float32, DenseFieldRole::Parameter, 27.5f', dense)
        self.assertIn('{"membrane_tau", "tau", "tau_m_ms"}, 27.5f', dense)

    def test_neuron_ir_owns_state_slots_and_input_routes(self) -> None:
        spec = _spec()
        spec["equations"] = (
            "dv/dt = -v / tau_m : mV\n"
            "dg_exc/dt = -g_exc / tau_exc : uS\n"
            "dg_inh/dt = -g_inh / tau_inh : uS\n"
            "di_input/dt = 0 : nA"
        )
        spec["inputs"] = [
            {"name": "g_exc", "kind": "excitatory_conductance"},
            {"name": "g_inh", "kind": "inhibitory_conductance"},
            {"name": "i_input", "kind": "current"},
        ]
        spec["events"] = {"spike": {"threshold": "v >= v_threshold", "reset": {"v": "v_reset"}}}
        spec["parameters"].update({
            "tau_m": {"api_name": "tau_m", "default": 20.0, "constraints": {"exclusive_min": 0.0}},
            "tau_exc": {"api_name": "tau_exc", "default": 5.0, "constraints": {"exclusive_min": 0.0}},
            "tau_inh": {"api_name": "tau_inh", "default": 10.0, "constraints": {"exclusive_min": 0.0}},
        })
        spec["runtime"] = {
            "voltage_state": "v",
            "state_update": {
                "v": {"method": "forward_euler", "initial": "0"},
                "g_exc": {"method": "exact_exponential", "initial": "0"},
                "g_inh": {"method": "exact_exponential", "initial": "0"},
                "i_input": {"method": "hold", "initial": "0"},
            },
        }

        model = compile_time_driven_neuron(spec)

        self.assertEqual(model.state_names, ("v", "g_exc", "g_inh", "i_input"))
        self.assertEqual([binding.connection_type for binding in model.input_bindings], [0, 1, 3])
        self.assertEqual(model.voltage_slot, 0)
        self.assertEqual(model.differential_state_count, 1)
        emitted = emit_descriptor_header(model)
        self.assertIn("kCustomLifConductanceV1Descriptor", emitted)
        self.assertIn("CustomInputDelivery::AddToCurrentAccumulator", emitted)

    def test_equation_ast_parses_and_emits_arithmetic(self) -> None:
        equations = parse_equations("dv/dt = (v_rest - v + i_input) / tau_m : mV")
        self.assertEqual(len(equations), 1)
        self.assertEqual(
            emit_cpp(
                equations[0].expression,
                {"v_rest": "v_rest_", "v": "state[kV]", "i_input": "state[kI]", "tau_m": "tau_m_"},
            ),
            "(((v_rest_ - state[kV]) + state[kI]) / tau_m_)",
        )

    def test_equation_ast_emits_registered_math_functions_for_cpu_and_cuda(self) -> None:
        cases = {
            "exp(x)": ("std::exp(x_)", "expf(x_)"),
            "expm1(x)": ("std::expm1(x_)", "expm1f(x_)"),
            "log(x)": ("std::log(x_)", "logf(x_)"),
            "log1p(x)": ("std::log1p(x_)", "log1pf(x_)"),
            "sqrt(x)": ("std::sqrt(x_)", "sqrtf(x_)"),
            "abs(x)": ("std::fabs(x_)", "fabsf(x_)"),
            "min(x, y)": ("std::fmin(x_, y_)", "fminf(x_, y_)"),
            "max(x, y)": ("std::fmax(x_, y_)", "fmaxf(x_, y_)"),
            "pow(x, y)": ("std::pow(x_, y_)", "powf(x_, y_)"),
        }
        for source, (cpu, cuda) in cases.items():
            with self.subTest(source=source):
                expression = parse_expression(source)
                symbols = {"x": "x_", "y": "y_"}
                self.assertEqual(emit_cpp(expression, symbols), cpu)
                self.assertEqual(emit_cpp(expression, symbols, "cuda"), cuda)

    def test_equation_ast_tracks_nested_function_identifiers_and_exponents(self) -> None:
        expression = parse_expression("max(exp(x), sqrt(abs(y))) + 1e-3")
        self.assertEqual(identifiers_in(expression), frozenset({"x", "y"}))
        emitted = emit_cpp(expression, {"x": "x_", "y": "y_"}, "cuda")
        self.assertIn("fmaxf(expf(x_), sqrtf(fabsf(y_)))", emitted)
        self.assertIn("1e-3f", emitted)

    def test_equation_ast_rejects_unknown_functions_and_wrong_arity(self) -> None:
        for source in ("sin(x)", "exp()", "exp(x, y)", "min(x)", "pow(x, y, 2)"):
            with self.subTest(source=source), self.assertRaises(EquationSyntaxError):
                parse_expression(source)

    def test_empty_spec_root_emits_deterministic_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_root = root / "specs"
            output_root = root / "generated"
            manifest = output_root / "manifest.json"

            result = cli.main(
                [
                    "--spec-root",
                    str(spec_root),
                    "--output-root",
                    str(output_root),
                    "--manifest",
                    str(manifest),
                ]
            )

            self.assertEqual(result, 0)
            self.assertTrue(
                (output_root / "src" / "GeneratedNeuronCatalogEntries.cpp").is_file()
            )
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["spec_count"], 0)

    def test_check_validates_spec_without_emitting_runtime_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_root = root / "specs"
            spec_root.mkdir()
            (spec_root / "custom_lif.nbmodel.json").write_text(
                json.dumps(_spec()), encoding="utf-8"
            )
            output_root = root / "generated"

            result = cli.main(
                [
                    "--spec-root",
                    str(spec_root),
                    "--output-root",
                    str(output_root),
                    "--manifest",
                    str(output_root / "manifest.json"),
                    "--check",
                ]
            )

            self.assertEqual(result, 0)
            self.assertFalse(output_root.exists())

    def test_layout_emission_writes_only_generated_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_root = root / "specs"
            spec_root.mkdir()
            spec = _spec()
            spec["events"] = {"spike": {"threshold": "v >= v_threshold", "reset": {"v": "v_reset"}}}
            (spec_root / "custom_lif.nbmodel.json").write_text(json.dumps(spec), encoding="utf-8")
            output_root = root / "generated"

            result = cli.main([
                "--spec-root", str(spec_root), "--output-root", str(output_root),
                "--manifest", str(output_root / "manifest.json"), "--emit-layout-only",
            ])

            self.assertEqual(result, 0)
            self.assertTrue((output_root / "include" / "neuronbridge_codegen" / "CustomLifConductanceV1Descriptor.h").is_file())
            self.assertFalse((output_root / "src" / "GeneratedNeuronCatalogEntries.cpp").exists())

    def test_builtin_name_conflict_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_root = root / "specs"
            spec_root.mkdir()
            (spec_root / "conflict.nbmodel.json").write_text(
                json.dumps(_spec("STDP")), encoding="utf-8"
            )

            result = cli.main(
                [
                    "--spec-root",
                    str(spec_root),
                    "--output-root",
                    str(root / "generated"),
                    "--manifest",
                    str(root / "generated" / "manifest.json"),
                    "--check",
                ]
            )

            self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
