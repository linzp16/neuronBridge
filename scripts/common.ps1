$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-NeuronBridgeRoot {
    $scriptsDir = Split-Path -Parent $PSScriptRoot
    return [System.IO.Path]::GetFullPath($scriptsDir)
}

function Resolve-CommandPath {
    param([Parameter(Mandatory = $true)][string]$Name)

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        return $null
    }
    return $command.Source
}

function Write-CheckResult {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][bool]$Passed,
        [Parameter(Mandatory = $true)][string]$Detail
    )

    $label = if ($Passed) { "PASS" } else { "FAIL" }
    $color = if ($Passed) { "Green" } else { "Red" }
    Write-Host ("[{0}] {1}: {2}" -f $label, $Name, $Detail) -ForegroundColor $color
}

function Assert-PathWithinRoot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Root
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $fullRoot = [System.IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
    if (-not $fullPath.StartsWith($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Path is outside the permitted root: $fullPath"
    }
    return $fullPath
}

function Get-Sha256 {
    param([Parameter(Mandatory = $true)][string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Read-DependencyManifest {
    $manifestPath = Join-Path (Get-NeuronBridgeRoot) "dependencies\manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Dependency manifest not found: $manifestPath"
    }
    return Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
}

function Get-DependencyBundle {
    param(
        [Parameter(Mandatory = $true)]$Manifest,
        [string]$PlatformKey = "windows-x64-msvc"
    )

    $bundle = @($Manifest.bundles) |
        Where-Object { $_.platform_key -eq $PlatformKey } |
        Sort-Object revision -Descending |
        Select-Object -First 1
    if ($null -eq $bundle) {
        throw "No dependency bundle is registered for platform: $PlatformKey"
    }
    return $bundle
}

function Invoke-DirectoryCopy {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string[]]$ExcludedFiles = @()
    )

    if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
        throw "Dependency source directory not found: $Source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $arguments = @($Source, $Destination, "/E", "/COPY:DAT", "/DCOPY:DAT",
        "/R:2", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
    if ($ExcludedFiles.Count -gt 0) {
        $arguments += "/XF"
        $arguments += $ExcludedFiles
    }
    & robocopy.exe @arguments | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed with exit code ${LASTEXITCODE}: $Source -> $Destination"
    }
}

function Resolve-Python312Path {
    param([string]$PythonExe = "")

    $root = Get-NeuronBridgeRoot
    $candidates = @()
    if (-not [string]::IsNullOrWhiteSpace($PythonExe)) {
        $candidates += $PythonExe
    }
    if (-not [string]::IsNullOrWhiteSpace($env:NEURONBRIDGE_PYTHON)) {
        $candidates += $env:NEURONBRIDGE_PYTHON
    }
    $candidates += @(
        (Join-Path $root ".venv\Scripts\python.exe"),
        "python"
    )

    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate -ErrorAction SilentlyContinue
        $path = if ($null -ne $command) { $command.Source } else { $candidate }
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            continue
        }
        $version = & $path -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $version -eq "3.12") {
            return [System.IO.Path]::GetFullPath($path)
        }
    }
    throw "Python 3.12 was not found. Pass -PythonExe or set NEURONBRIDGE_PYTHON."
}

function Resolve-VsDevCmdPath {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
        $installation = & $vswhere -latest -version '[17.0,18.0)' -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($installation)) {
            $candidate = Join-Path $installation "Common7\Tools\VsDevCmd.bat"
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                return $candidate
            }
        }
    }
    throw "Visual Studio 2022 C++ tools were not found through vswhere.exe."
}

function Invoke-CleanVsDevCommand {
    param(
        [Parameter(Mandatory = $true)][string]$CommandLine,
        [string]$VsDevCmdPath = ""
    )

    if ([string]::IsNullOrWhiteSpace($VsDevCmdPath)) {
        $VsDevCmdPath = Resolve-VsDevCmdPath
    }
    $developerCommand = 'call "' + $VsDevCmdPath +
        '" -arch=x64 -host_arch=x64 && ' + $CommandLine
    $pathValue = $env:PATH
    if ([string]::IsNullOrWhiteSpace($pathValue)) {
        $pathValue = [Environment]::GetEnvironmentVariable("Path", "Process")
    }
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = Join-Path $env:SystemRoot "System32\cmd.exe"
    $startInfo.Arguments = "/d /s /c `"$developerCommand`""
    $startInfo.WorkingDirectory = (Get-Location).Path
    $startInfo.UseShellExecute = $false
    $startInfo.EnvironmentVariables.Clear()
    foreach ($entry in [Environment]::GetEnvironmentVariables("Process").GetEnumerator()) {
        $key = [string]$entry.Key
        if ([string]::Equals($key, "PATH", [System.StringComparison]::OrdinalIgnoreCase)) {
            continue
        }
        if (-not $startInfo.EnvironmentVariables.ContainsKey($key)) {
            $startInfo.EnvironmentVariables.Add($key, [string]$entry.Value)
        }
    }
    $startInfo.EnvironmentVariables.Add("Path", $pathValue)
    $process = [System.Diagnostics.Process]::Start($startInfo)
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        throw "Visual Studio developer command failed: $CommandLine"
    }
}

function Get-NeuronBridgeVersion {
    $root = Get-NeuronBridgeRoot
    $version = (Get-Content -LiteralPath (Join-Path $root "VERSION") -Raw).Trim()
    if ($version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(?:a[0-9]+|b[0-9]+|rc[0-9]+)?$') {
        throw "VERSION is not a supported PEP 440 release version: $version"
    }
    return $version
}

function Get-LatestNeuronBridgeWheel {
    param([Parameter(Mandatory = $true)][string]$Directory)

    $wheel = Get-ChildItem -LiteralPath $Directory -Filter "neuronbridge-*.whl" `
        -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $wheel) {
        throw "No neuronbridge wheel was found under: $Directory"
    }
    return $wheel
}
