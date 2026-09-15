"""CPU learning-rule generation protocol regressions."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.model_codegen import cli
from tools.model_codegen.equation_ast import EquationSyntaxError
from tools.model_codegen.model_ir import ModelIrError
from tools.model_codegen.learning_rule_emitter import render, render_cuda


def spec():
    path = Path(__file__).resolve().parents[3] / "models/specs/custom_rstdp_v1.nbmodel.json"
    return json.loads(path.read_text(encoding="utf-8"))


def pair_spec():
    path = Path(__file__).resolve().parents[3] / "models/specs/custom_pair_stdp_v1.nbmodel.json"
    return json.loads(path.read_text(encoding="utf-8"))


class RStdpTest(unittest.TestCase):
    def test_learning_emitter_has_no_model_identity_whitelist(self):
        item = pair_spec()
        item["implementation_name"] = "GeneratedPairRuleProbe"
        item["canonical_name"] = "GeneratedPairRuleProbe"
        item["variant_id"] = 20991
        item["dense_factory_model_id"] = 91
        header, source = render(item)
        cuda = render_cuda(item)
        self.assertIn("class GeneratedPairRuleProbe final", header)
        self.assertIn("GeneratedPairRuleProbe::ApplyPreSynaticSpike", source)
        self.assertIn("ApplyDenseGeneratedPairRuleProbePreDeviceEntry", cuda)

    def test_dense_roles_allow_renamed_states_and_parameters(self):
        item = pair_spec()
        replacements = {
            "pre_trace": "arrival_memory",
            "post_trace": "firing_memory",
            "max_ltp": "potentiation_scale",
            "max_ltd": "depression_scale",
            "ltp_tau": "arrival_tau",
            "ltd_tau": "firing_tau",
        }
        for old, new in replacements.items():
            item["equations"] = item["equations"].replace(old, new)
            item["events"] = {
                event: [statement.replace(old, new) for statement in statements]
                for event, statements in item["events"].items()
            }
        item["parameters"] = {
            replacements.get(name, name): declaration
            for name, declaration in item["parameters"].items()
        }
        item["runtime"]["dense"]["roles"] = {
            role: replacements.get(name, name)
            for role, name in item["runtime"]["dense"]["roles"].items()
        }
        header, source = render(item)
        cuda = render_cuda(item)
        self.assertIn("arrival_memory", source)
        self.assertIn("potentiation_scale_", header)
        self.assertIn("pre_trace += 1.0f", cuda)

    def test_pair_stdp_has_distinct_state_and_event_contract(self):
        item = pair_spec()
        header, source = render(item)
        cuda = render_cuda(item)
        self.assertIn("CustomLearningRuleModel(true, false)", source)
        self.assertIn("SynapseState(count, 2)", source)
        self.assertIn("LearningRuleIndex_withPost", source)
        self.assertNotIn("eligibility", source)
        self.assertNotIn("on_trigger", source)
        self.assertIn("ApplyDenseCustomPairStdpV1PreDeviceEntry", cuda)
        self.assertIn("weight += ((-max_ltd) * post_trace);", cuda)
        self.assertNotIn("kDenseRStdpEligibility", cuda)

    def test_pair_stdp_rejects_trigger_and_rstdp_state(self):
        item = pair_spec(); item["events"]["on_trigger"] = ["weight += 1"]
        with self.assertRaises(ModelIrError):
            render(item)
        item = pair_spec(); item["equations"] += "\ndeligibility/dt = 0 : 1"
        with self.assertRaises(ModelIrError):
            render(item)
    def test_equation_events_are_emitted_not_ignored(self):
        item = spec()
        header, source = render(item)
        self.assertIn("public CustomLearningRuleModel", header)
        self.assertIn("public SynapseState", source)
        self.assertNotIn("STDP_State", source)
        item["events"]["on_pre"][1] = "eligibility += -max_ltd * post_trace * 2"
        self.assertNotEqual(source, render(item)[1])
        self.assertIn("* 2.0f", render(item)[1])

    def test_cuda_events_are_lowered_from_ast(self):
        item = spec()
        cuda = render_cuda(item)
        self.assertIn("ApplyDenseCustomRStdpV1PreDeviceEntry", cuda)
        self.assertIn("eligibility += ((-max_ltd) * post_trace);", cuda)
        self.assertIn("weight += (trigger_factor * eligibility);", cuda)
        self.assertNotIn("ApplyRStdpPreDeviceEntry", cuda)
        item["events"]["on_post"][1] = "eligibility += pre_trace * max_ltp * 2"
        self.assertIn("* 2.0f", render_cuda(item))

    def test_reject_invalid_contracts(self):
        changes = [
            ("extends", {"base_class": "R_STDP"}),
            ("api", {"visibility": "public"}),
            ("integration", {"method": "euler"}),
            ("equations", spec()["equations"].replace("deligibility/dt = 0", "deligibility/dt = -eligibility")),
            ("equations", spec()["equations"] + "\nunvalidated : 1"),
            ("trigger", {}),
            ("events", {"on_pre": []}),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                item = spec(); item[key] = value
                with self.assertRaises((EquationSyntaxError, ModelIrError)): render(item)

    def test_reject_event_injection_unknown_symbol_and_wrong_order(self):
        for statement in ["eligibility += sin(post_trace)", "eligibility += unknown",
                          "eligibility += 1; system(0)", "eligibility += trigger_factor"]:
            item = spec(); item["events"]["on_pre"][1] = statement
            with self.subTest(statement=statement), self.assertRaises((EquationSyntaxError, ModelIrError)):
                render(item)
        item = spec(); item["events"]["on_pre"].reverse()
        self.assertIn("s[2] +=", render(item)[1])

    def test_math_functions_lower_in_cpu_and_cuda_events(self):
        item = spec()
        item["events"]["on_pre"][1] = (
            "eligibility += -max_ltd * expm1(log1p(abs(post_trace)))"
        )
        _, cpu = render(item)
        cuda = render_cuda(item)
        self.assertIn("std::expm1(std::log1p(std::fabs(s[1])))", cpu)
        self.assertIn("expm1f(log1pf(fabsf(post_trace)))", cuda)

    def test_parameter_validation_and_defaults(self):
        for value in [0, -1, True, float("nan"), float("inf"), "16.8", 1e100]:
            item = spec(); item["parameters"]["ltp_tau"]["default"] = value
            with self.subTest(value=value), self.assertRaises(ModelIrError): render(item)
        item = spec(); item["parameters"]["max_ltp"]["default"] = 0.25
        self.assertIn("max_ltp_ = 0.25f", render(item)[0])

    def test_dense_host_normalizes_public_keys_and_spec_defaults(self):
        item = pair_spec()
        item["parameters"]["max_ltp"] = {
            "api_name": "potentiation_scale",
            "aliases": ["Max_LTP"],
            "default": 0.25,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); specs = root / "specs"; specs.mkdir()
            (specs / "rule.nbmodel.json").write_text(json.dumps(item), encoding="utf-8")
            output = root / "generated"
            self.assertEqual(cli.main(["--spec-root", str(specs), "--output-root", str(output),
                                      "--manifest", str(output / "manifest.json")]), 0)
            host = (output / "include/neuronbridge_codegen/DenseCustomPairStdpV1.h").read_text()
            self.assertIn('resolved.RuleParameter["Max_LTP"] = '
                          'resolved.RuleParameter["potentiation_scale"]', host)
            self.assertIn('resolved.RuleParameter["Max_LTP"] = 0.25f', host)

    def test_rule_only_build_and_empty_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); specs = root / "specs"; specs.mkdir()
            path = specs / "rule.nbmodel.json"; path.write_text(json.dumps(spec()), encoding="utf-8")
            output = root / "generated"
            args = ["--spec-root", str(specs), "--output-root", str(output), "--manifest", str(output / "manifest.json")]
            self.assertEqual(cli.main(args), 0)
            self.assertIn("CustomRStdpV1", (output / "src/GeneratedLearningRuleCatalogEntries.cpp").read_text())
            path.unlink()
            self.assertEqual(cli.main(args), 0)
            self.assertNotIn("CustomRStdpV1", (output / "src/GeneratedLearningRuleCatalogEntries.cpp").read_text())
            self.assertFalse((output / "include/neuronbridge_codegen/CustomRStdpV1.h").exists())

    def test_multiple_rules_emit_isolated_cpu_and_dense_registries(self):
        persistent_path = Path(__file__).resolve().parents[3] / "models/specs/custom_rstdp_persistent_v1.nbmodel.json"
        persistent = json.loads(persistent_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); specs = root / "specs"; specs.mkdir()
            (specs / "first.nbmodel.json").write_text(json.dumps(spec()), encoding="utf-8")
            (specs / "second.nbmodel.json").write_text(json.dumps(persistent), encoding="utf-8")
            output = root / "generated"
            self.assertEqual(cli.main(["--spec-root", str(specs), "--output-root", str(output),
                                      "--manifest", str(output / "manifest.json")]), 0)
            registry = (output / "include/neuronbridge_codegen/CustomLegacyLearningRuleRegistry.inc").read_text()
            dense = (output / "include/neuronbridge_codegen/CustomDenseLearningModelList.inc").read_text()
            self.assertEqual(registry.count("NPGR_CUSTOM_LEGACY_RULE"), 2)
            self.assertEqual(dense.count("NPGR_DENSE_LEARNING_MODEL"), 2)
            self.assertIn("CustomRStdpPersistentV1", registry)
            self.assertIn("kDenseCustomRStdpPersistentV1ModelId, 5", dense)
            self.assertIn("kDenseCustomRStdpV1ModelId, 4", dense)
            self.assertNotIn("ApplyRStdpPreDeviceEntry", dense)
            device = (output / "include/neuronbridge_codegen/CustomGeneratedDenseLearningDeviceUpdates.cuh").read_text()
            self.assertIn("DenseCustomRStdpV1DeviceUpdate.cuh", device)
            self.assertIn("DenseCustomRStdpPersistentV1DeviceUpdate.cuh", device)
            persistent_host = (output / "include/neuronbridge_codegen/DenseCustomRStdpPersistentV1.h").read_text()
            self.assertIn('resolved.RuleParameter["ClearEligibilityAfterTrigger"] = false', persistent_host)
            self.assertEqual(json.loads((output / "manifest.json").read_text())["spec_count"], 2)

    def test_check_rejects_rule_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); specs = root / "specs"; specs.mkdir()
            item = spec(); item["events"]["on_pre"][1] = "eligibility += not_a_parameter"
            (specs / "bad.nbmodel.json").write_text(json.dumps(item), encoding="utf-8")
            output = root / "generated"
            self.assertEqual(cli.main(["--spec-root", str(specs), "--output-root", str(output),
                                      "--manifest", str(output / "manifest.json"), "--check"]), 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
