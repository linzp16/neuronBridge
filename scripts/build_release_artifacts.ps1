[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$DependencyWheelhouse = "",
    [string]$PythonRuntimeRoot = "",
    [switch]$SkipOfflineBundle,
    [switch]$PublicRelease
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-NeuronBridgeRoot
$PythonExe = Resolve-Python312Path -PythonExe $PythonExe
$metadataArgs = @((Join-Path $PSScriptRoot "check_release_metadata.py"), "--root", $root)
if ($PublicRelease) { $metadataArgs += "--require-license" }
& $PythonExe @metadataArgs
if ($LASTEXITCODE -ne 0) { throw "Release metadata gate failed." }
& (Join-Path $PSScriptRoot "verify_dependencies.ps1") -Mode All
& (Join-Path $PSScriptRoot "build_neuronbridge_wheel.ps1") -PythonExe $PythonExe
& (Join-Path $PSScriptRoot "check_neuronbridge_runtime_deps.ps1") -PythonExe $PythonExe
& (Join-Path $PSScriptRoot "validate_neuronbridge_wheel.ps1") -PythonExe $PythonExe
& (Join-Path $PSScriptRoot "build_source_package.ps1")
if (-not $SkipOfflineBundle) {
    $offlineArgs = @{
        PythonExe = $PythonExe
        DependencyWheelhouse = $DependencyWheelhouse
    }
    if (-not [string]::IsNullOrWhiteSpace($PythonRuntimeRoot)) {
        $offlineArgs.PythonRuntimeRoot = $PythonRuntimeRoot
    }
    & (Join-Path $PSScriptRoot "build_neuronbridge_offline_bundle.ps1") @offlineArgs
}
& (Join-Path $PSScriptRoot "write_release_checksums.ps1")
Write-Host "NeuronBridge release artifacts passed Stage 7 validation." -ForegroundColor Green
