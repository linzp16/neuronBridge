[CmdletBinding()]
param([string]$ArtifactRoot = "")

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$root = Get-NeuronBridgeRoot
if ([string]::IsNullOrWhiteSpace($ArtifactRoot)) {
    $ArtifactRoot = Join-Path $root "artifacts"
}
$ArtifactRoot = Assert-PathWithinRoot -Path $ArtifactRoot -Root $root
if (-not (Test-Path -LiteralPath $ArtifactRoot -PathType Container)) {
    throw "Artifact root not found: $ArtifactRoot"
}
$output = Join-Path $ArtifactRoot "SHA256SUMS.txt"
$artifactDirectories = @("wheel", "source", "offline", "example-data")
$releaseFiles = foreach ($directoryName in $artifactDirectories) {
    $directory = Join-Path $ArtifactRoot $directoryName
    if (Test-Path -LiteralPath $directory -PathType Container) {
        Get-ChildItem -LiteralPath $directory -File |
            Where-Object { $_.Extension -in @(".whl", ".zip") }
    }
}
$lines = $releaseFiles |
    Sort-Object FullName |
    ForEach-Object {
        $relative = [System.IO.Path]::GetRelativePath($ArtifactRoot, $_.FullName).Replace('\', '/')
        "$(Get-Sha256 -Path $_.FullName)  $relative"
    }
if ($lines.Count -eq 0) { throw "No wheel or ZIP artifacts found under $ArtifactRoot" }
Set-Content -LiteralPath $output -Value $lines -Encoding ascii
Write-Host "Release checksums: $output" -ForegroundColor Green
