[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$DistDir = "",
    [string]$BuildDir = "",
    [ValidateSet("Release")][string]$Config = "Release",
    [switch]$BuildIsolation,
    [switch]$SkipRepair
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-NeuronBridgeRoot
$PythonExe = Resolve-Python312Path -PythonExe $PythonExe
if ([string]::IsNullOrWhiteSpace($DistDir)) {
    $DistDir = Join-Path $root "artifacts\wheel"
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $root "build\wheel"
}
$DistDir = Assert-PathWithinRoot -Path $DistDir -Root $root
$BuildDir = Assert-PathWithinRoot -Path $BuildDir -Root $root
$rawDir = Assert-PathWithinRoot -Path (Join-Path $DistDir "raw") -Root $root
New-Item -ItemType Directory -Force -Path $DistDir, $rawDir | Out-Null
Get-ChildItem -LiteralPath $rawDir -Filter "neuronbridge-*.whl" -File -ErrorAction SilentlyContinue |
    Remove-Item -Force
Get-ChildItem -LiteralPath $DistDir -Filter "neuronbridge-*.whl" -File -ErrorAction SilentlyContinue |
    Remove-Item -Force

& $PythonExe -c "import build, pybind11, scikit_build_core, wheel"
if ($LASTEXITCODE -ne 0) {
    throw "Wheel build tools are missing. Install build, pybind11, scikit-build-core, and wheel into $PythonExe."
}
if (-not $SkipRepair) {
    & $PythonExe -c "import delvewheel"
    if ($LASTEXITCODE -ne 0) {
        throw "delvewheel is required for Windows wheel repair. Install it into $PythonExe."
    }
}

$pythonProject = Join-Path $root "python"
$pythonCmake = $PythonExe.Replace('\', '/')
$cmakeArgs = @(
    "-DNR_ENABLE_PYTHON=ON",
    "-DNR_ENABLE_CUDA=ON",
    "-DNR_ENABLE_MODEL_CODEGEN=ON",
    "-DNR_BUNDLE_MSVC_RUNTIME=ON",
    "-DNR_BUILD_TESTS=OFF",
    "-DNR_BUILD_EXAMPLES=OFF",
    "-DNR_DEPENDENCY_MODE=BUNDLED",
    "-DPython3_EXECUTABLE=$pythonCmake"
)
$buildArguments = @(
    "-m", "build", $pythonProject,
    "--wheel",
    "--verbose",
    "--outdir", $rawDir,
    "--config-setting=cmake.build-type=$Config",
    "--config-setting=build-dir=$BuildDir"
)
if (-not $BuildIsolation) {
    $buildArguments += "--no-isolation"
}
$quotedBuild = $buildArguments | ForEach-Object { '"' + $_.Replace('"', '""') + '"' }
$commandLine = 'set "CMAKE_ARGS=' + ($cmakeArgs -join ' ') + '" && "' +
    $PythonExe + '" ' + ($quotedBuild -join ' ')
Invoke-CleanVsDevCommand -CommandLine $commandLine

$rawWheel = Get-LatestNeuronBridgeWheel -Directory $rawDir
if ($SkipRepair) {
    Copy-Item -LiteralPath $rawWheel.FullName -Destination $DistDir -Force
} else {
    $manifest = Read-DependencyManifest
    $bundle = Get-DependencyBundle -Manifest $manifest
    $dependencyRoot = Join-Path $root ("dependencies\vendor\" + $bundle.install_subdir.Replace('/', '\'))
    $searchPaths = @(
        (Join-Path $dependencyRoot "pinocchio-cpp\Library\bin"),
        (Join-Path $dependencyRoot "zeromq\bin")
    )
    if (-not [string]::IsNullOrWhiteSpace($env:CUDA_PATH)) {
        $searchPaths += (Join-Path $env:CUDA_PATH "bin")
    }
    $addPath = ($searchPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Container }) -join ';'
    & $PythonExe -m delvewheel repair $rawWheel.FullName `
        --add-path $addPath --ignore-existing --analyze-existing --wheel-dir $DistDir
    if ($LASTEXITCODE -ne 0) {
        throw "delvewheel failed to repair $($rawWheel.FullName)"
    }
}

$wheel = Get-LatestNeuronBridgeWheel -Directory $DistDir
$privateRuntimeScript = Join-Path $PSScriptRoot "private_runtime_wheel.py"
& $PythonExe $privateRuntimeScript $wheel.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Failed to isolate bundled MSVC/OpenMP runtime DLL names in $($wheel.FullName)"
}
$report = Join-Path $DistDir "wheel-contents.json"
& $PythonExe (Join-Path $PSScriptRoot "inspect_neuronbridge_wheel.py") `
    $wheel.FullName --output $report
if ($LASTEXITCODE -ne 0) {
    throw "Wheel content inspection failed: $($wheel.FullName)"
}
Write-Host "NeuronBridge wheel ready: $($wheel.FullName)" -ForegroundColor Green
Write-Host "Wheel inspection report: $report"
