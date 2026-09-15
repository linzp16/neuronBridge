[CmdletBinding()]
param(
    [string]$OutputDir = "",
    [switch]$IncludeDependencyBundle,
    [switch]$AllowDirty
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-NeuronBridgeRoot
$version = Get-NeuronBridgeVersion
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $root "artifacts\source"
}
$OutputDir = Assert-PathWithinRoot -Path $OutputDir -Root $root
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
Push-Location $root
try {
    $status = @(git status --porcelain=v1 --untracked-files=no)
    if (-not $AllowDirty -and $status.Count -gt 0) {
        throw "Source package requires a clean tracked worktree."
    }
    $suffix = if ($IncludeDependencyBundle) { "-with-windows-deps" } else { "" }
    $packageName = "neuronbridge-source-$version$suffix"
    $stagingDir = Assert-PathWithinRoot -Path (Join-Path $OutputDir $packageName) -Root $OutputDir
    $archivePath = Assert-PathWithinRoot -Path (Join-Path $OutputDir "$packageName.zip") -Root $OutputDir
    if (Test-Path -LiteralPath $stagingDir) { Remove-Item -LiteralPath $stagingDir -Recurse -Force }
    if (Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath -Force }
    New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null

    $files = @(git -c core.quotePath=false ls-files) | Where-Object {
        $IncludeDependencyBundle -or -not $_.StartsWith("dependencies/bundles/")
    }
    foreach ($relative in $files) {
        $source = Join-Path $root $relative
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
        if ($IncludeDependencyBundle -and $relative.StartsWith("dependencies/bundles/")) {
            if ((Get-Item -LiteralPath $source).Length -lt 1024) {
                throw "Dependency bundle is an LFS pointer. Run git lfs pull before packaging."
            }
        }
        $destination = Join-Path $stagingDir $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
    $manifestFiles = @(
        Get-ChildItem -LiteralPath $stagingDir -Recurse -File | ForEach-Object {
            [ordered]@{
                path = [System.IO.Path]::GetRelativePath($stagingDir, $_.FullName).Replace('\', '/')
                size_bytes = $_.Length
                sha256 = Get-Sha256 -Path $_.FullName
            }
        }
    ) | Sort-Object path
    $manifest = [ordered]@{
        schema_version = 1
        package = $packageName
        version = $version
        source_commit = (git rev-parse HEAD)
        includes_windows_dependencies = [bool]$IncludeDependencyBundle
        project_license_present = (Test-Path -LiteralPath (Join-Path $root "LICENSE"))
        files = $manifestFiles
    }
    $manifest | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath (Join-Path $stagingDir "SOURCE_PACKAGE_MANIFEST.json") -Encoding utf8
    & tar.exe -a -c -f $archivePath -C $OutputDir $packageName
    if ($LASTEXITCODE -ne 0) { throw "Failed to create source package: $archivePath" }
    Write-Host "Source package ready: $archivePath" -ForegroundColor Green
    Write-Host "SHA-256: $(Get-Sha256 -Path $archivePath)"
} finally {
    Pop-Location
}
