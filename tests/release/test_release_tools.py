from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def load_wheel_inspector():
    path = ROOT / "scripts" / "inspect_neuronbridge_wheel.py"
    spec = importlib.util.spec_from_file_location("wheel_inspector", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_versions_are_consistent():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_release_metadata.py"), "--root", str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert set(report["versions"].values()) == {"0.1.0a0"}
    assert report["project_license_present"] is False
    assert report["public_release_ready"] is False


def test_wheel_inspector_accepts_required_runtime_layout(tmp_path: Path):
    wheel = tmp_path / "neuronbridge-0.1.0a0-cp312-cp312-win_amd64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("neuronbridge/_core.cp312-win_amd64.pyd", b"pyd")
        archive.writestr("neuronbridge/cudart64_12.dll", b"cuda")
        archive.writestr("neuronbridge/pinocchio_default.dll", b"pin")
        archive.writestr("neuronbridge/libzmq-mt-4_3_5.dll", b"zmq")
    report = load_wheel_inspector().inspect_wheel(wheel)
    assert report["result"] == "PASS"
    assert report["native_dll_count"] == 3


def test_wheel_inspector_rejects_build_products(tmp_path: Path):
    wheel = tmp_path / "neuronbridge-invalid.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("neuronbridge/Release/debug.pdb", b"debug")
    report = load_wheel_inspector().inspect_wheel(wheel)
    assert report["result"] == "FAIL"
    assert report["forbidden_entries"]
    assert report["missing_requirements"]


def test_ci_separates_hosted_checks_from_cuda_release_runner():
    hosted = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    cuda = (ROOT / ".github" / "workflows" / "windows-cuda-release.yml").read_text(encoding="utf-8")
    assert "runs-on: windows-2022" in hosted
    assert "NR_ENABLE_CUDA=ON" not in hosted
    assert "runs-on: [self-hosted, Windows, X64, neuronbridge-cuda]" in cuda
    assert "validate_neuronbridge_wheel.ps1" in cuda


def test_portable_bundle_owns_python_and_offline_dependencies():
    script = (ROOT / "scripts" / "build_neuronbridge_offline_bundle.ps1").read_text(encoding="utf-8")
    assert 'Join-Path $runtimeDir "python.exe"' in script
    assert "--no-index" in script
    assert "--ignore-installed" in script
    assert "SHA256SUMS.txt" in script


def test_python_extension_bundles_msvc_and_openmp_runtimes():
    cmake = (ROOT / "python" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "CMAKE_INSTALL_OPENMP_LIBRARIES TRUE" in cmake
    assert "include(InstallRequiredSystemLibraries)" in cmake
    assert "${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}" in cmake
    assert 'install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION neuronbridge)' in cmake
