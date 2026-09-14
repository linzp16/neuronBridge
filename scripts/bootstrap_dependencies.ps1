[CmdletBinding()]
param(
    [string]$PlatformKey = "windows-x64-msvc",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-NeuronBridgeRoot
$dependenciesRoot = Join-Path $root "dependencies"
$vendorRoot = Join-Path $dependenciesRoot "vendor"
$manifest = Read-DependencyManifest
$bundle = Get-DependencyBundle -Manifest $manifest -PlatformKey $PlatformKey
$archivePath = Join-Path $dependenciesRoot ($bundle.archive.Replace('/', '\'))

if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Dependency archive not found. Run git lfs pull first: $archivePath"
}
if ((Get-Item -LiteralPath $archivePath).Length -lt 1024) {
    $firstLine = Get-Content -LiteralPath $archivePath -TotalCount 1 -ErrorAction SilentlyContinue
    if ($firstLine -like "version https://git-lfs.github.com/spec/*") {
        throw "Dependency archive is only a Git LFS pointer. Run git lfs pull."
    }
}

& (Join-Path $PSScriptRoot "verify_dependencies.ps1") `
    -Mode Archive -PlatformKey $PlatformKey

$installRoot = Assert-PathWithinRoot `
    -Path (Join-Path $vendorRoot ($bundle.install_subdir.Replace('/', '\'))) `
    -Root $vendorRoot
$markerPath = Join-Path $installRoot ".installed.json"
if ((Test-Path -LiteralPath $markerPath -PathType Leaf) -and -not $Force) {
    $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    if ($marker.archive_sha256 -eq $bundle.archive_sha256) {
        Write-Host "Dependency bundle is already installed: $installRoot" -ForegroundColor Green
        & (Join-Path $PSScriptRoot "verify_dependencies.ps1") `
            -Mode Installed -PlatformKey $PlatformKey
        exit 0
    }
}
if ((Test-Path -LiteralPath $installRoot) -and -not $Force) {
    throw "A different or incomplete dependency installation exists. Use -Force after review: $installRoot"
}

New-Item -ItemType Directory -Force -Path $vendorRoot | Out-Null
$temporaryRoot = Assert-PathWithinRoot `
    -Path (Join-Path $vendorRoot (".tmp-" + $bundle.bundle_id + "-" + [guid]::NewGuid().ToString("N"))) `
    -Root $vendorRoot
New-Item -ItemType Directory -Force -Path $temporaryRoot | Out-Null

try {
    Write-Host "Extracting dependency bundle..."
    & tar.exe -xf $archivePath -C $temporaryRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract dependency archive."
    }
    $extractedRoot = Join-Path $temporaryRoot $bundle.extract_root
    if (-not (Test-Path -LiteralPath $extractedRoot -PathType Container)) {
        throw "Archive does not contain the declared root: $($bundle.extract_root)"
    }
    foreach ($required in @($bundle.required_files)) {
        $path = Join-Path $extractedRoot ($required.path.Replace('/', '\'))
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Extracted bundle is missing required file: $($required.path)"
        }
        if ((Get-Item -LiteralPath $path).Length -lt [int64]$required.minimum_size) {
            throw "Extracted dependency file is too small: $($required.path)"
        }
    }
    if (Test-Path -LiteralPath $installRoot) {
        Remove-Item -LiteralPath $installRoot -Recurse -Force
    }
    $installParent = Split-Path -Parent $installRoot
    New-Item -ItemType Directory -Force -Path $installParent | Out-Null
    Move-Item -LiteralPath $extractedRoot -Destination $installRoot
    $marker = [ordered]@{
        bundle_id = $bundle.bundle_id
        archive_sha256 = $bundle.archive_sha256
        installed_utc = (Get-Date).ToUniversalTime().ToString("o")
    }
    $marker | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $markerPath -Encoding utf8
} finally {
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}

& (Join-Path $PSScriptRoot "verify_dependencies.ps1") `
    -Mode Installed -PlatformKey $PlatformKey
Write-Host "Bundled dependencies installed under $installRoot" -ForegroundColor Green
