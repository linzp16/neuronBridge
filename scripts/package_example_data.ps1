[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$DataRoot,
    [string]$OutputDir = "",
    [switch]$AllowPartial
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-NeuronBridgeRoot
$version = Get-NeuronBridgeVersion
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) {
    throw "Example data root not found: $DataRoot"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $root "artifacts\example-data"
}
$OutputDir = Assert-PathWithinRoot -Path $OutputDir -Root $root
$packageName = "neuronbridge-example-data-$version"
$stagingDir = Assert-PathWithinRoot -Path (Join-Path $OutputDir $packageName) -Root $OutputDir
$archivePath = Assert-PathWithinRoot -Path (Join-Path $OutputDir "$packageName.zip") -Root $OutputDir
if (Test-Path -LiteralPath $stagingDir) { Remove-Item -LiteralPath $stagingDir -Recurse -Force }
if (Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath -Force }
New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null

$catalog = Get-Content -LiteralPath (Join-Path $root "examples\data\manifest.json") -Raw | ConvertFrom-Json
$included = @()
$missing = @()
foreach ($dataset in @($catalog.datasets)) {
    $inputDir = Join-Path $DataRoot (Join-Path $dataset.name "input_data")
    if (-not (Test-Path -LiteralPath $inputDir -PathType Container)) {
        $missing += $dataset.name
        continue
    }
    $destination = Join-Path $stagingDir (Join-Path $dataset.name "input_data")
    Invoke-DirectoryCopy -Source $inputDir -Destination $destination -ExcludedFiles @("*.pyc")
    $included += $dataset.name
}
if (-not $AllowPartial -and $missing.Count -gt 0) {
    Remove-Item -LiteralPath $stagingDir -Recurse -Force
    throw "Missing dataset directories: $($missing -join ', ')"
}
if ($included.Count -eq 0) {
    Remove-Item -LiteralPath $stagingDir -Recurse -Force
    throw "No dataset was found under $DataRoot"
}
$files = @(
    Get-ChildItem -LiteralPath $stagingDir -Recurse -File | ForEach-Object {
        [ordered]@{
            path = [System.IO.Path]::GetRelativePath($stagingDir, $_.FullName).Replace('\', '/')
            size_bytes = $_.Length
            sha256 = Get-Sha256 -Path $_.FullName
        }
    }
) | Sort-Object path
$bundleManifest = [ordered]@{
    schema_version = 1
    package = $packageName
    layout = "<archive-root>/<dataset>/input_data"
    included_datasets = $included
    omitted_datasets = $missing
    files = $files
}
$bundleManifest | ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $stagingDir "DATA_BUNDLE_MANIFEST.json") -Encoding utf8
& tar.exe -a -c -f $archivePath -C $OutputDir $packageName
if ($LASTEXITCODE -ne 0) { throw "Failed to create example data archive." }
Write-Host "Example data archive ready: $archivePath" -ForegroundColor Green
Write-Host "Included datasets: $($included -join ', ')"
Write-Host "SHA-256: $(Get-Sha256 -Path $archivePath)"
