[CmdletBinding()]
param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$failures = 0

function Test-Tool {
    param([string]$Name, [string]$CommandName)

    $path = Resolve-CommandPath -Name $CommandName
    $passed = -not [string]::IsNullOrWhiteSpace($path)
    Write-CheckResult -Name $Name -Passed $passed -Detail $(
        if ($passed) { $path } else { "$CommandName was not found" }
    )
    if (-not $passed) {
        $script:failures += 1
    }
}

Write-Host "NeuronBridge Windows development environment check"
Write-Host "Repository: $(Get-NeuronBridgeRoot)"

Test-Tool -Name "Git" -CommandName "git.exe"
Test-Tool -Name "Git LFS" -CommandName "git-lfs.exe"
Test-Tool -Name "CMake" -CommandName "cmake.exe"
Test-Tool -Name "CUDA compiler" -CommandName "nvcc.exe"

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vsInstall = ""
if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
    $vsInstall = (& $vswhere -latest -products * -version "[17.0,18.0)" `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1)
}
$hasVs2022 = -not [string]::IsNullOrWhiteSpace($vsInstall)
Write-CheckResult -Name "Visual Studio 2022 C++" -Passed $hasVs2022 -Detail $(
    if ($hasVs2022) { $vsInstall } else { "VS2022 with the x64 C++ tools was not found" }
)
if (-not $hasVs2022) {
    $failures += 1
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = $env:NR_PYTHON_EXECUTABLE
}
$pythonDetail = ""
$hasPython312 = $false
if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
    if (Test-Path -LiteralPath $PythonExe -PathType Leaf) {
        $pythonDetail = (& $PythonExe -c "import sys; print(sys.executable); print(f'{sys.version_info.major}.{sys.version_info.minor}')") -join " | "
        $hasPython312 = $LASTEXITCODE -eq 0 -and $pythonDetail.EndsWith("3.12")
    } else {
        $pythonDetail = "Configured Python executable does not exist: $PythonExe"
    }
} elseif ($null -ne (Get-Command "py.exe" -ErrorAction SilentlyContinue)) {
    $pythonOutput = (& py.exe -3.12 -c "import sys; print(sys.executable); print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null)
    if ($LASTEXITCODE -eq 0) {
        $pythonDetail = $pythonOutput -join " | "
        $hasPython312 = $pythonDetail.EndsWith("3.12")
    } else {
        $pythonDetail = "The Python launcher has no registered Python 3.12 runtime"
    }
} else {
    $pythonDetail = "Set -PythonExe or NR_PYTHON_EXECUTABLE to a Python 3.12 executable"
}
Write-CheckResult -Name "Python 3.12" -Passed $hasPython312 -Detail $pythonDetail
if (-not $hasPython312) {
    $failures += 1
}

$manifest = Join-Path (Get-NeuronBridgeRoot) "dependencies\manifest.json"
$hasManifest = Test-Path -LiteralPath $manifest -PathType Leaf
Write-CheckResult -Name "Dependency manifest" -Passed $hasManifest -Detail $manifest
if (-not $hasManifest) {
    $failures += 1
}

if ($failures -gt 0) {
    throw "Environment check failed with $failures issue(s)."
}

Write-Host "Environment check completed." -ForegroundColor Green
