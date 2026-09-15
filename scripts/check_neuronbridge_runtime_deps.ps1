[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$WheelPath = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-NeuronBridgeRoot
$PythonExe = Resolve-Python312Path -PythonExe $PythonExe
if ([string]::IsNullOrWhiteSpace($WheelPath)) {
    $WheelPath = (Get-LatestNeuronBridgeWheel -Directory (Join-Path $root "artifacts\wheel")).FullName
}
if (-not (Test-Path -LiteralPath $WheelPath -PathType Leaf)) {
    throw "Wheel not found: $WheelPath"
}
& $PythonExe (Join-Path $PSScriptRoot "inspect_neuronbridge_wheel.py") $WheelPath
if ($LASTEXITCODE -ne 0) {
    throw "Static wheel inspection failed."
}
& $PythonExe -m delvewheel show $WheelPath
if ($LASTEXITCODE -ne 0) {
    throw "delvewheel dependency inspection failed."
}
