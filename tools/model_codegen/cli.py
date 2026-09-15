"""Deterministic maintainer-only scaffold for custom equation model generation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from . import GENERATOR_VERSION
from .equation_ast import EquationSyntaxError, parse_equations
from .model_ir import ModelIrError, compile_time_driven_neuron
from .legacy_cpu_emitter import write_descriptor_header
from .neuron_emitter import emit_runtime
from .learning_rule_emitter import emit_runtime as emit_rule_runtime, render as validate_rule


SUPPORTED_KINDS = {
    "custom_time_driven_neuron",
    "custom_learning_rule",
}

BUILTIN_NAMES = {
    "TimeDrivenLIF_Exponential_double",
    "TimeDrivenLIF_Exponential_Decay",
    "TimeDrivenIzhikevic_Exponential_Decay",
    "TimeDrivenLIF_Voltage_jump",
    "TimeDrivenLIF_Exponential_triple",
    "PoissonRate",
    "STDP",
    "R_STDP",
    "AdditiveKernalChange",
    "CerebullarLearningRule",
}


class SpecError(ValueError):
    """A spec violates a generator-owned invariant."""


def _read_specs(spec_root: Path) -> list[dict[str, Any]]:
    if not spec_root.exists():
        return []

    specs: list[dict[str, Any]] = []
    for path in sorted(spec_root.rglob("*.nbmodel.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SpecError(f"{path}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise SpecError(f"{path}: top-level value must be an object")
        value["_source_path"] = path.as_posix()
        specs.append(value)
    return specs


def _require_string(spec: dict[str, Any], name: str) -> str:
    value = spec.get(name)
    if not isinstance(value, str) or not value:
        raise SpecError(f"{spec['_source_path']}: {name} must be a non-empty string")
    return value


def _validate_specs(specs: list[dict[str, Any]]) -> None:
    variant_ids: set[int] = set()
    implementation_names: set[str] = set()
    canonical_names: set[str] = set(BUILTIN_NAMES)
    dense_neuron_ids: set[int] = set()
    dense_rule_ids: set[int] = set()

    for spec in specs:
        if spec.get("schema_version") != 1:
            raise SpecError(f"{spec['_source_path']}: schema_version must be 1")
        kind = _require_string(spec, "kind")
        if kind not in SUPPORTED_KINDS:
            raise SpecError(f"{spec['_source_path']}: unsupported kind {kind!r}")

        variant_id = spec.get("variant_id")
        if not isinstance(variant_id, int) or isinstance(variant_id, bool):
            raise SpecError(f"{spec['_source_path']}: variant_id must be an integer")
        if variant_id in variant_ids:
            raise SpecError(f"{spec['_source_path']}: duplicate variant_id {variant_id}")
        variant_ids.add(variant_id)

        implementation_name = _require_string(spec, "implementation_name")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", implementation_name) is None:
            raise SpecError(f"{spec['_source_path']}: implementation_name must be a C++ identifier")
        if implementation_name in implementation_names or implementation_name in BUILTIN_NAMES:
            raise SpecError(
                f"{spec['_source_path']}: implementation_name conflicts with an existing model")
        implementation_names.add(implementation_name)

        canonical_name = spec.get("canonical_name", implementation_name)
        if not isinstance(canonical_name, str) or not canonical_name:
            raise SpecError(f"{spec['_source_path']}: canonical_name must be a non-empty string")
        if canonical_name in canonical_names:
            raise SpecError(f"{spec['_source_path']}: canonical_name conflicts with an existing model")
        canonical_names.add(canonical_name)

        extends = spec.get("extends")
        if not isinstance(extends, dict) or not isinstance(extends.get("base_class"), str):
            raise SpecError(f"{spec['_source_path']}: extends.base_class is required")
        equations = spec.get("equations")
        if not isinstance(equations, str) or not equations.strip():
            raise SpecError(f"{spec['_source_path']}: equations must be a non-empty string")
        try:
            parse_equations(equations)
        except EquationSyntaxError as error:
            raise SpecError(f"{spec['_source_path']}: invalid equations: {error}") from error
        if kind == "custom_time_driven_neuron":
            dense_id = spec.get("dense_factory_model_id")
            if not isinstance(dense_id, int) or isinstance(dense_id, bool) or dense_id < 0:
                raise SpecError(f"{spec['_source_path']}: dense_factory_model_id must be non-negative")
            if dense_id in dense_neuron_ids:
                raise SpecError(f"{spec['_source_path']}: duplicate dense neuron model id {dense_id}")
            dense_neuron_ids.add(dense_id)
            try:
                compile_time_driven_neuron(spec)
            except (EquationSyntaxError, ModelIrError) as error:
                raise SpecError(f"{spec['_source_path']}: cannot lower neuron model: {error}") from error
        elif kind == "custom_learning_rule":
            dense_id = spec.get("dense_factory_model_id")
            if not isinstance(dense_id, int) or isinstance(dense_id, bool) or dense_id < 0:
                raise SpecError(f"{spec['_source_path']}: dense_factory_model_id must be non-negative")
            if dense_id in dense_rule_ids:
                raise SpecError(f"{spec['_source_path']}: duplicate dense learning model id {dense_id}")
            dense_rule_ids.add(dense_id)
            validate_rule(spec)


def _manifest(specs: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for spec in specs:
        normalized = {key: value for key, value in spec.items() if key != "_source_path"}
        encoded = json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        entries.append(
            {
                "source": spec["_source_path"],
                "variant_id": spec["variant_id"],
                "implementation_name": spec["implementation_name"],
                "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            }
        )
    return {
        "generator_version": GENERATOR_VERSION,
        "spec_count": len(entries),
        "specs": entries,
    }


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8", newline="\n")


def _clear_managed_outputs(output_root: Path) -> None:
    """Remove only directories exclusively owned by this generator invocation."""
    for path in (output_root / "include" / "neuronbridge_codegen", output_root / "src"):
        if path.exists():
            shutil.rmtree(path)


def _emit_empty_scaffold(output_root: Path, manifest: dict[str, Any]) -> None:
    include_root = output_root / "include" / "neuronbridge_codegen"
    source_root = output_root / "src"
    _write(include_root / "CustomGeneratedModels.h", "#pragma once\n")
    _write(include_root / "CustomGeneratedLegacyGpuModels.cuh", "#pragma once\n")
    _write(include_root / "CustomGeneratedDenseNeuronModels.h", "#pragma once\n")
    _write(include_root / "CustomGeneratedDenseNeuronDeviceUpdates.cuh", "#pragma once\n")
    _write(include_root / "CustomDenseNeuronModelList.inc", "// No generated dense neuron models.\n")
    _write(source_root / "GeneratedNeuronModels.cpp", "// No generated neuron models.\n")
    _write(source_root / "GeneratedLegacyGpuNeuronModels.cu", "// No generated legacy GPU neuron models.\n")
    _write(source_root / "GeneratedLearningRuleModels.cpp", "// No generated learning rule models.\n")
    _write(
        include_root / "CustomLegacyNeuronRegistry.inc",
        "// Generated by tools.model_codegen. No custom legacy neuron entries yet.\n",
    )
    _write(
        include_root / "CustomLegacyGpuNeuronRegistry.inc",
        "// Generated by tools.model_codegen. No custom legacy GPU neuron entries yet.\n",
    )
    _write(
        include_root / "CustomNeuronCatalog.inc",
        "// Generated by tools.model_codegen. No custom neuron catalog entries yet.\n",
    )
    _write(
        source_root / "GeneratedNeuronCatalogEntries.cpp",
        "#include \"neuron_model/GeneratedNeuronCatalog.h\"\n\n"
        "namespace npgr {\n\n"
        "void RegisterGeneratedNeuronCatalogEntries(\n"
        "    std::vector<NeuronModelCatalogEntry>* entries) {\n"
        "    (void)entries;\n"
        "}\n\n"
        "}  // namespace npgr\n",
    )
    _write(
        output_root / "manifest.json",
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def _emit_layout_headers(output_root: Path, specs: list[dict[str, Any]]) -> None:
    for spec in specs:
        if spec["kind"] == "custom_time_driven_neuron":
            write_descriptor_header(output_root / "include", compile_time_driven_neuron(spec))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--emit-layout-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        specs = _read_specs(args.spec_root)
        _validate_specs(specs)
        manifest = _manifest(specs)
        if args.check:
            print(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
            return 0
        if args.emit_layout_only:
            _emit_layout_headers(args.output_root, specs)
            _write(args.manifest, json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
            return 0
        _clear_managed_outputs(args.output_root)
        unsupported = [s for s in specs if s["kind"] not in {"custom_time_driven_neuron", "custom_learning_rule"}]
        if unsupported:
            raise SpecError("no concrete emitter for requested backend")
        neurons = [s for s in specs if s["kind"] == "custom_time_driven_neuron"]
        rules = [s for s in specs if s["kind"] == "custom_learning_rule"]
        if neurons:
            emit_runtime(args.output_root, neurons)
            _write(args.output_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        else:
            _emit_empty_scaffold(args.output_root, manifest)
        emit_rule_runtime(args.output_root, rules)
        if args.manifest != args.output_root / "manifest.json":
            _write(args.manifest, json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
        return 0
    except (SpecError, ModelIrError, EquationSyntaxError) as error:
        print(f"model_codegen: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
