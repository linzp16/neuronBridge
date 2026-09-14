[CmdletBinding()]
param(
    [ValidateSet("Archive", "Installed", "All")][string]$Mode = "All",
    [string]$PlatformKey = "windows-x64-msvc"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-NeuronBridgeRoot
$manifest = Read-DependencyManifest
$bundle = Get-DependencyBundle -Manifest $manifest -PlatformKey $PlatformKey
$failures = 0

function Register-Result {
    param([string]$Name, [bool]$Passed, [string]$Detail)
    Write-CheckResult -Name $Name -Passed $Passed -Detail $Detail
    if (-not $Passed) {
        $script:failures += 1
    }
}

if ($Mode -in @("Archive", "All")) {
    $archivePath = Join-Path (Join-Path $root "dependencies") ($bundle.archive.Replace('/', '\'))
    $archiveExists = Test-Path -LiteralPath $archivePath -PathType Leaf
    Register-Result -Name "Dependency archive" -Passed $archiveExists -Detail $archivePath
    if ($archiveExists) {
        $archiveItem = Get-Item -LiteralPath $archivePath
        Register-Result -Name "Archive size" `
            -Passed ($archiveItem.Length -eq [int64]$bundle.archive_size_bytes) `
            -Detail "$($archiveItem.Length) bytes"
        $hash = Get-Sha256 -Path $archivePath
        Register-Result -Name "Archive SHA-256" `
            -Passed ($hash -eq $bundle.archive_sha256) -Detail $hash
        $listing = @(& tar.exe -tf $archivePath)
        Register-Result -Name "Archive readability" `
            -Passed ($LASTEXITCODE -eq 0 -and $listing.Count -gt 0) `
            -Detail "$($listing.Count) archive entries"
    }
}

if ($Mode -in @("Installed", "All")) {
    $installRoot = Join-Path (Join-Path $root "dependencies\vendor") `
        ($bundle.install_subdir.Replace('/', '\'))
    $installExists = Test-Path -LiteralPath $installRoot -PathType Container
    Register-Result -Name "Installed dependency root" -Passed $installExists -Detail $installRoot
    if ($installExists) {
        foreach ($required in @($bundle.required_files)) {
            $path = Join-Path $installRoot ($required.path.Replace('/', '\'))
            $exists = Test-Path -LiteralPath $path -PathType Leaf
            $size = if ($exists) { (Get-Item -LiteralPath $path).Length } else { 0 }
            Register-Result -Name $required.path `
                -Passed ($exists -and $size -ge [int64]$required.minimum_size) `
                -Detail "$size bytes"
        }
        $infoPath = Join-Path $installRoot "bundle-info.json"
        Register-Result -Name "Bundle metadata" `
            -Passed (Test-Path -LiteralPath $infoPath -PathType Leaf) `
            -Detail $infoPath

        $cmakeRoots = @(
            (Join-Path $installRoot "pinocchio-cpp\Library\lib\cmake"),
            (Join-Path $installRoot "pinocchio-cpp\Library\share")
        )
        $machinePathMatches = @(
            foreach ($cmakeRoot in $cmakeRoots) {
                if (Test-Path -LiteralPath $cmakeRoot -PathType Container) {
                    Get-ChildItem -LiteralPath $cmakeRoot -Recurse -File -Filter "*.cmake" |
                        Select-String -Pattern '(D:/anaconda|D:/bld|C:/bld)' -SimpleMatch:$false
                }
            }
        )
        Register-Result -Name "Relocatable CMake metadata" `
            -Passed ($machinePathMatches.Count -eq 0) `
            -Detail "$($machinePathMatches.Count) forbidden build-machine path reference(s)"
    }
}

if ($failures -gt 0) {
    throw "Dependency verification failed with $failures issue(s)."
}
Write-Host "Dependency verification completed." -ForegroundColor Green
