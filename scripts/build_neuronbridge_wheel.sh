#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd -- "${script_dir}/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "build_neuronbridge_wheel.sh requires Linux" >&2
  exit 2
fi

python_exe="${PYTHON_EXE:-python3}"
dist_dir="${NR_DIST_DIR:-${root}/artifacts/wheel-linux}"
build_dir="${NR_BUILD_DIR:-${root}/build/wheel-linux}"
raw_dir="${dist_dir}/raw"
dependency_mode="${NR_DEPENDENCY_MODE:-BUNDLED}"
dependency_key="${NR_DEPENDENCY_PLATFORM_KEY:-linux-x86_64-gcc-cuda12}"
dependency_root="${NR_DEPENDENCY_ROOT:-${root}/dependencies/vendor/linux-x86_64-gcc-cuda12/r1}"
cuda_architectures="${NR_CUDA_ARCHITECTURES:-52;61;75;80;86;89}"
wheel_platform="${NR_WHEEL_PLATFORM:-manylinux_2_28_x86_64}"

mkdir -p "${dist_dir}" "${raw_dir}" "${build_dir}"
find "${raw_dir}" -maxdepth 1 -type f -name 'neuronbridge-*.whl' -delete
find "${dist_dir}" -maxdepth 1 -type f -name 'neuronbridge-*.whl' -delete

"${python_exe}" -c \
  'import build, pybind11, scikit_build_core, wheel, auditwheel'

if [[ "${dependency_mode^^}" == "BUNDLED" && ! -d "${dependency_root}" ]]; then
  echo "Linux dependency bundle is missing: ${dependency_root}" >&2
  echo "Set NR_DEPENDENCY_ROOT or use NR_DEPENDENCY_MODE=SYSTEM." >&2
  exit 2
fi

cmake_args=(
  "-DNR_ENABLE_PYTHON=ON"
  "-DNR_ENABLE_CUDA=ON"
  "-DNR_ENABLE_MODEL_CODEGEN=ON"
  "-DNR_BUILD_TESTS=OFF"
  "-DNR_BUILD_EXAMPLES=OFF"
  "-DNR_DEPENDENCY_MODE=${dependency_mode}"
  "-DNR_DEPENDENCY_PLATFORM_KEY=${dependency_key}"
  "-DNR_CUDA_RUNTIME_LINKAGE=STATIC"
  "-DNR_CUDA_ARCHITECTURES=${cuda_architectures}"
  "-DPython3_EXECUTABLE=${python_exe}"
)
if [[ -n "${dependency_root}" ]]; then
  cmake_args+=("-DNR_DEPENDENCY_ROOT=${dependency_root}")
fi

printf -v joined_cmake_args ' %q' "${cmake_args[@]}"
export CMAKE_ARGS="${joined_cmake_args# } ${CMAKE_ARGS:-}"

"${python_exe}" -m build "${root}/python" \
  --wheel \
  --no-isolation \
  --outdir "${raw_dir}" \
  "--config-setting=cmake.build-type=Release" \
  "--config-setting=build-dir=${build_dir}"

mapfile -t raw_wheels < <(find "${raw_dir}" -maxdepth 1 -type f \
  -name 'neuronbridge-*.whl' -print)
if [[ "${#raw_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one raw wheel, found ${#raw_wheels[@]}" >&2
  exit 2
fi

"${python_exe}" -m auditwheel show "${raw_wheels[0]}"
"${python_exe}" -m auditwheel repair "${raw_wheels[0]}" \
  --plat "${wheel_platform}" \
  --exclude libcuda.so.1 \
  --wheel-dir "${dist_dir}"

mapfile -t repaired_wheels < <(find "${dist_dir}" -maxdepth 1 -type f \
  -name 'neuronbridge-*.whl' -print)
if [[ "${#repaired_wheels[@]}" -ne 1 ]]; then
  echo "Expected exactly one repaired wheel, found ${#repaired_wheels[@]}" >&2
  exit 2
fi

"${python_exe}" "${script_dir}/inspect_neuronbridge_wheel.py" \
  "${repaired_wheels[0]}" --output "${dist_dir}/wheel-contents.json"
"${python_exe}" "${script_dir}/check_linux_wheel_runtime.py" \
  "${repaired_wheels[0]}" --output "${dist_dir}/runtime-dependencies.json"

echo "NeuronBridge Linux wheel ready: ${repaired_wheels[0]}"
