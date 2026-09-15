[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$WheelPath = "",
    [string]$ValidationDir = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-NeuronBridgeRoot
$PythonExe = Resolve-Python312Path -PythonExe $PythonExe
if ([string]::IsNullOrWhiteSpace($WheelPath)) {
    $WheelPath = (Get-LatestNeuronBridgeWheel -Directory (Join-Path $root "artifacts\wheel")).FullName
}
if ([string]::IsNullOrWhiteSpace($ValidationDir)) {
    $ValidationDir = Join-Path $root "build\wheel-validation"
}
$ValidationDir = Assert-PathWithinRoot -Path $ValidationDir -Root $root
$venvDir = Join-Path $ValidationDir "venv"
$isolationDir = Join-Path $ValidationDir "isolated-workdir"
if (Test-Path -LiteralPath $ValidationDir) {
    Remove-Item -LiteralPath $ValidationDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $isolationDir | Out-Null

& $PythonExe -m venv $venvDir
if ($LASTEXITCODE -ne 0) { throw "Failed to create validation venv: $venvDir" }
$venvPython = Join-Path $venvDir "Scripts\python.exe"
& $venvPython -m pip install --disable-pip-version-check --no-input --no-deps $WheelPath
if ($LASTEXITCODE -ne 0) { throw "Failed to install wheel into isolated venv." }

$smokeSource = Join-Path $PSScriptRoot "validate_neuronbridge_wheel_cuda_runtime.py"
$smokeCopy = Join-Path $isolationDir "validate_installed_wheel.py"
Copy-Item -LiteralPath $smokeSource -Destination $smokeCopy -Force
$report = Join-Path $ValidationDir "cuda-runtime-report.json"
$oldPythonPath = $env:PYTHONPATH
$oldCmakePrefixPath = $env:CMAKE_PREFIX_PATH
try {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:CMAKE_PREFIX_PATH -ErrorAction SilentlyContinue
    Push-Location $isolationDir
    & $venvPython $smokeCopy --output $report `
        --forbid-path (Join-Path $root "python\src") `
        --forbid-path (Join-Path $root "build\wheel\python")
    if ($LASTEXITCODE -ne 0) { throw "Installed-wheel CUDA runtime smoke test failed." }
    Pop-Location
} finally {
    if ((Get-Location).Path -eq $isolationDir) { Pop-Location }
    if ($null -ne $oldPythonPath) { $env:PYTHONPATH = $oldPythonPath }
    if ($null -ne $oldCmakePrefixPath) { $env:CMAKE_PREFIX_PATH = $oldCmakePrefixPath }
}
Write-Host "Installed-wheel validation passed: $WheelPath" -ForegroundColor Green
Write-Host "Runtime report: $report"
