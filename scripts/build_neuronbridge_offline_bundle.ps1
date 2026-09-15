[CmdletBinding(DefaultParameterSetName = "RuntimeRoot")]
param(
    [string]$PythonExe = "",
    [string]$WheelPath = "",
    [string]$BundleDir = "",
    [string]$DependencyWheelhouse = "",
    [Parameter(ParameterSetName = "RuntimeRoot")][string]$PythonRuntimeRoot = "",
    [Parameter(ParameterSetName = "RuntimeArchive")][string]$PythonRuntimeArchive = "",
    [switch]$DownloadDependencies,
    [string]$IndexUrl = "https://pypi.org/simple",
    [bool]$CreateArchive = $true
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-NeuronBridgeRoot
$PythonExe = Resolve-Python312Path -PythonExe $PythonExe
$version = Get-NeuronBridgeVersion
if ([string]::IsNullOrWhiteSpace($WheelPath)) {
    $WheelPath = (Get-LatestNeuronBridgeWheel -Directory (Join-Path $root "artifacts\wheel")).FullName
}
if ([string]::IsNullOrWhiteSpace($BundleDir)) {
    $BundleDir = Join-Path $root "artifacts\offline\neuronbridge-$version-windows-x64-py312"
}
$BundleDir = Assert-PathWithinRoot -Path $BundleDir -Root $root
if ([string]::IsNullOrWhiteSpace($PythonRuntimeRoot) -and
    [string]::IsNullOrWhiteSpace($PythonRuntimeArchive)) {
    $PythonRuntimeRoot = Split-Path -Parent $PythonExe
}
if (-not (Test-Path -LiteralPath $WheelPath -PathType Leaf)) {
    throw "Wheel not found: $WheelPath"
}
if (Test-Path -LiteralPath $BundleDir) {
    Remove-Item -LiteralPath $BundleDir -Recurse -Force
}
$runtimeDir = Join-Path $BundleDir "runtime"
$wheelhouseDir = Join-Path $BundleDir "wheelhouse"
$examplesDir = Join-Path $BundleDir "examples"
$docsDir = Join-Path $BundleDir "docs"
New-Item -ItemType Directory -Force -Path $runtimeDir, $wheelhouseDir, $examplesDir, $docsDir | Out-Null

if (-not [string]::IsNullOrWhiteSpace($PythonRuntimeArchive)) {
    if (-not (Test-Path -LiteralPath $PythonRuntimeArchive -PathType Leaf)) {
        throw "Python runtime archive not found: $PythonRuntimeArchive"
    }
    Expand-Archive -LiteralPath $PythonRuntimeArchive -DestinationPath $runtimeDir -Force
} else {
    if (-not (Test-Path -LiteralPath $PythonRuntimeRoot -PathType Container)) {
        throw "Python runtime root not found: $PythonRuntimeRoot"
    }
    $excludedDirectories = @(
        (Join-Path $PythonRuntimeRoot "conda-meta"),
        (Join-Path $PythonRuntimeRoot "include"),
        (Join-Path $PythonRuntimeRoot "libs"),
        (Join-Path $PythonRuntimeRoot "Scripts"),
        (Join-Path $PythonRuntimeRoot "share"),
        (Join-Path $PythonRuntimeRoot "Tools"),
        (Join-Path $PythonRuntimeRoot "Lib\site-packages"),
        (Join-Path $PythonRuntimeRoot "Library\include"),
        (Join-Path $PythonRuntimeRoot "Library\lib"),
        (Join-Path $PythonRuntimeRoot "Library\share")
    )
    $arguments = @(
        $PythonRuntimeRoot, $runtimeDir, "/E", "/COPY:DAT", "/DCOPY:DAT",
        "/R:2", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP",
        "/XF", "*.pdb", "*.lib", "*.exp", "*.pyc", "/XD"
    ) + $excludedDirectories
    & robocopy.exe @arguments | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Failed to copy the Python 3.12 runtime." }
}

$runtimePython = Join-Path $runtimeDir "python.exe"
if (-not (Test-Path -LiteralPath $runtimePython -PathType Leaf)) {
    throw "The portable runtime does not contain python.exe."
}
$pthFile = Get-ChildItem -LiteralPath $runtimeDir -Filter "python312._pth" -File -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -ne $pthFile) {
    $pth = Get-Content -LiteralPath $pthFile.FullName
    $pth = @($pth | Where-Object { $_ -ne "#import site" }) + @("Lib/site-packages", "import site")
    Set-Content -LiteralPath $pthFile.FullName -Value ($pth | Select-Object -Unique) -Encoding ascii
}

Copy-Item -LiteralPath $WheelPath -Destination $wheelhouseDir -Force
if (-not [string]::IsNullOrWhiteSpace($DependencyWheelhouse)) {
    if (-not (Test-Path -LiteralPath $DependencyWheelhouse -PathType Container)) {
        throw "Dependency wheelhouse not found: $DependencyWheelhouse"
    }
    Get-ChildItem -LiteralPath $DependencyWheelhouse -Filter "*.whl" -File |
        Where-Object { $_.Name -notlike "neuronbridge-*" } |
        Copy-Item -Destination $wheelhouseDir -Force
}
if ($DownloadDependencies) {
    & $PythonExe -m pip download --disable-pip-version-check --only-binary=:all: `
        --dest $wheelhouseDir --index-url $IndexUrl `
        "numpy>=2.0" "matplotlib>=3.7" "pandas>=2.0" "pyzmq>=27"
    if ($LASTEXITCODE -ne 0) { throw "Failed to download offline dependency wheels." }
}

$sitePackages = Join-Path $runtimeDir "Lib\site-packages"
New-Item -ItemType Directory -Force -Path $sitePackages | Out-Null
& $PythonExe -m pip install --disable-pip-version-check --no-input --no-index --ignore-installed `
    --find-links $wheelhouseDir --target $sitePackages "$WheelPath[communication]"
if ($LASTEXITCODE -ne 0) {
    throw "Offline dependency resolution failed. Supply -DependencyWheelhouse or -DownloadDependencies."
}
Copy-Item -Path (Join-Path $root "examples\python\*.py") -Destination $examplesDir -Force
Copy-Item -LiteralPath (Join-Path $root "examples\README.md") -Destination $docsDir -Force
Copy-Item -LiteralPath (Join-Path $root "dependencies\THIRD_PARTY_NOTICES.md") -Destination $docsDir -Force

$launcher = @'
$ErrorActionPreference = "Stop"
$bundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $bundleRoot "runtime\python.exe") @args
exit $LASTEXITCODE
'@
Set-Content -LiteralPath (Join-Path $BundleDir "neuronbridge-python.ps1") -Value $launcher -Encoding utf8
$smoke = @'
$ErrorActionPreference = "Stop"
$bundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $bundleRoot "runtime\python.exe"
Push-Location $bundleRoot
try {
    & $python -c "import matplotlib, neuronbridge as nb, numpy, pandas, zmq; info=nb.backend_info(); print(info); assert info['native_extension_loaded'] and info['cuda_enabled']"
    if ($LASTEXITCODE -ne 0) { throw "Portable NeuronBridge import failed." }
    & $python (Join-Path $bundleRoot "examples\dense_run_no_debug.py") --help
    if ($LASTEXITCODE -ne 0) { throw "Portable example smoke test failed." }
} finally {
    Pop-Location
}
'@
Set-Content -LiteralPath (Join-Path $BundleDir "run_smoke_test.ps1") -Value $smoke -Encoding utf8
$readme = @"
# NeuronBridge portable package $version

This package includes a Python 3.12 runtime, NeuronBridge, visualization and
communication dependencies, examples, and native runtime DLLs. Users do not
need to install Python, MSVC, CMake, CUDA Toolkit, pybind11, or ZeroMQ.
An NVIDIA driver is still required when CUDA simulation is used.

Run `neuronbridge-python.ps1` as the Python entry point and
`run_smoke_test.ps1` to validate the package after extraction.
"@
Set-Content -LiteralPath (Join-Path $BundleDir "README.md") -Value $readme -Encoding utf8

$hashLines = Get-ChildItem -LiteralPath $BundleDir -Recurse -File |
    Where-Object { $_.Name -ne "SHA256SUMS.txt" } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = [System.IO.Path]::GetRelativePath($BundleDir, $_.FullName).Replace('\', '/')
        "$(Get-Sha256 -Path $_.FullName)  $relative"
    }
Set-Content -LiteralPath (Join-Path $BundleDir "SHA256SUMS.txt") -Value $hashLines -Encoding ascii

& $runtimePython -c "import matplotlib, neuronbridge as nb, numpy, pandas, zmq; info=nb.backend_info(); print(info); assert info['native_extension_loaded'] and info['cuda_enabled']"
if ($LASTEXITCODE -ne 0) { throw "Portable runtime import validation failed." }
if ($CreateArchive) {
    $archivePath = "$BundleDir.zip"
    if (Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath -Force }
    & tar.exe -a -c -f $archivePath -C (Split-Path -Parent $BundleDir) (Split-Path -Leaf $BundleDir)
    if ($LASTEXITCODE -ne 0) { throw "Failed to create offline bundle archive." }
    Write-Host "Offline archive: $archivePath"
}
Write-Host "Portable offline package ready: $BundleDir" -ForegroundColor Green
