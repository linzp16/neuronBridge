[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SourceVendorRoot,
    [int]$Revision = 1,
    [string]$CondaMetaDir = "",
    [switch]$IncludeDebugSymbols
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

if ($Revision -lt 1) {
    throw "Revision must be a positive integer."
}

$root = Get-NeuronBridgeRoot
$platformKey = "windows-x64-msvc"
$bundleId = "neuronbridge-deps-$platformKey-r$Revision"
$sourceRoot = [System.IO.Path]::GetFullPath($SourceVendorRoot)
$pinSource = Join-Path $sourceRoot "pinocchio-cpp"
$zmqSource = Join-Path $sourceRoot "zeromq"

foreach ($requiredSource in @($pinSource, $zmqSource)) {
    if (-not (Test-Path -LiteralPath $requiredSource -PathType Container)) {
        throw "Required dependency source is missing: $requiredSource"
    }
}

$artifactsRoot = Join-Path $root "artifacts\dependency-package"
$stagingRoot = Assert-PathWithinRoot -Path (Join-Path $artifactsRoot $bundleId) -Root $artifactsRoot
$payloadRoot = Join-Path $stagingRoot $bundleId
$bundleDirectory = Join-Path $root "dependencies\bundles\$platformKey"
$archivePath = Join-Path $bundleDirectory "$bundleId.zip"

if (Test-Path -LiteralPath $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
New-Item -ItemType Directory -Force -Path $payloadRoot | Out-Null
New-Item -ItemType Directory -Force -Path $bundleDirectory | Out-Null

$excludedFiles = if ($IncludeDebugSymbols) { @() } else { @("*.pdb") }
Write-Host "Copying the validated Pinocchio dependency prefix..."
Invoke-DirectoryCopy -Source $pinSource `
    -Destination (Join-Path $payloadRoot "pinocchio-cpp") `
    -ExcludedFiles $excludedFiles
Write-Host "Copying the validated ZeroMQ dependency prefix..."
Invoke-DirectoryCopy -Source $zmqSource `
    -Destination (Join-Path $payloadRoot "zeromq") `
    -ExcludedFiles $excludedFiles

# Conda packages are normally relocatable, but a few exported CMake files keep
# build-machine fallback paths. Rewrite only the known path forms while the
# relative package structure is still intact.
$pinPayload = Join-Path $payloadRoot "pinocchio-cpp\Library"
$assimpTargets = Join-Path $pinPayload "lib\cmake\assimp-6.0\assimpTargets.cmake"
if (Test-Path -LiteralPath $assimpTargets -PathType Leaf) {
    $content = Get-Content -LiteralPath $assimpTargets -Raw
    $content = [regex]::Replace(
        $content,
        '[A-Za-z]:/[^";>]*/Library/lib/z\.lib',
        '$${_IMPORT_PREFIX}/lib/z.lib')
    Set-Content -LiteralPath $assimpTargets -Value $content -Encoding utf8
}
$boostConfigs = Get-ChildItem (Join-Path $pinPayload "lib\cmake") `
    -Recurse -File -Filter "boost_*-config.cmake"
foreach ($config in $boostConfigs) {
    $content = Get-Content -LiteralPath $config.FullName -Raw
    $content = [regex]::Replace(
        $content,
        '[A-Za-z]:/[^"\r\n]*/lib/cmake',
        '$${_BOOST_CMAKEDIR}')
    Set-Content -LiteralPath $config.FullName -Value $content -Encoding utf8
}

$metadataRoot = Join-Path $payloadRoot "metadata"
New-Item -ItemType Directory -Force -Path $metadataRoot | Out-Null
if (-not [string]::IsNullOrWhiteSpace($CondaMetaDir)) {
    if (-not (Test-Path -LiteralPath $CondaMetaDir -PathType Container)) {
        throw "Conda metadata directory not found: $CondaMetaDir"
    }
    $packages = @(
        Get-ChildItem -LiteralPath $CondaMetaDir -Filter "*.json" -File |
            ForEach-Object {
                $item = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                [ordered]@{
                    name = $item.name
                    version = $item.version
                    build = $item.build
                    license = $item.license
                    channel = $item.channel
                }
            } | Sort-Object { $_.name }
    )
    $packages | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $metadataRoot "conda-package-inventory.json") -Encoding utf8
}

Copy-Item -LiteralPath (Join-Path $root "dependencies\THIRD_PARTY_NOTICES.md") `
    -Destination (Join-Path $metadataRoot "THIRD_PARTY_NOTICES.md") -Force

$payloadFiles = @(Get-ChildItem -LiteralPath $payloadRoot -Recurse -File)
$payloadBytes = ($payloadFiles | Measure-Object Length -Sum).Sum
$requiredFiles = @(
    [ordered]@{ path = "pinocchio-cpp/Library/lib/cmake/pinocchio/pinocchioConfig.cmake"; minimum_size = 1024 },
    [ordered]@{ path = "pinocchio-cpp/Library/include/pinocchio/fwd.hpp"; minimum_size = 256 },
    [ordered]@{ path = "pinocchio-cpp/Library/include/boost/any.hpp"; minimum_size = 1024 },
    [ordered]@{ path = "pinocchio-cpp/Library/include/eigen3/Eigen/Core"; minimum_size = 1024 },
    [ordered]@{ path = "pinocchio-cpp/Library/bin/pinocchio_default.dll"; minimum_size = 1048576 },
    [ordered]@{ path = "pinocchio-cpp/Library/bin/pinocchio_parsers.dll"; minimum_size = 1048576 },
    [ordered]@{ path = "zeromq/include/zmq.h"; minimum_size = 4096 },
    [ordered]@{ path = "zeromq/lib/libzmq-mt-4_3_5.lib"; minimum_size = 4096 },
    [ordered]@{ path = "zeromq/bin/libzmq-mt-4_3_5.dll"; minimum_size = 4096 },
    [ordered]@{ path = "zeromq/share/cmake/ZeroMQ/ZeroMQConfig.cmake"; minimum_size = 512 }
)

$bundleInfo = [ordered]@{
    schema_version = 1
    bundle_id = $bundleId
    platform_key = $platformKey
    architecture = "x86_64"
    compiler_abi = "msvc"
    revision = $Revision
    generated_utc = (Get-Date).ToUniversalTime().ToString("o")
    source_repository = "https://github.com/linzp16/net_package.git"
    source_commit = "64139f571698d91754e77a9624db1e863c43d955"
    policy = "compatibility_bundle"
    relocation_repairs = @(
        "assimp zlib imported link path",
        "Boost original CMake directory fallbacks"
    )
    excluded_patterns = $excludedFiles
    payload_file_count = $payloadFiles.Count
    payload_size_bytes = $payloadBytes
    components = @(
        [ordered]@{ name = "pinocchio"; version = "4.0.0" },
        [ordered]@{ name = "boost"; version = "1.88.0" },
        [ordered]@{ name = "eigen"; version = "3.4.0" },
        [ordered]@{ name = "zeromq"; version = "4.3.5" }
    )
    required_files = $requiredFiles
    licensing_status = "review_required_before_public_redistribution"
}
$bundleInfo | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $payloadRoot "bundle-info.json") -Encoding utf8

$tar = Get-Command tar.exe -ErrorAction SilentlyContinue
if ($null -eq $tar) {
    throw "tar.exe is required to create the dependency ZIP archive."
}
Write-Host "Creating dependency archive..."
& $tar.Source -a -c -f $archivePath -C $stagingRoot $bundleId
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Failed to create dependency archive: $archivePath"
}

$archiveItem = Get-Item -LiteralPath $archivePath
$archiveHash = Get-Sha256 -Path $archivePath
$manifest = [ordered]@{
    schema_version = 1
    project = "neuronbridge"
    default_platform = $platformKey
    bundles = @(
        [ordered]@{
            bundle_id = $bundleId
            platform_key = $platformKey
            architecture = "x86_64"
            compiler_abi = "msvc"
            revision = $Revision
            archive = "bundles/$platformKey/$bundleId.zip"
            archive_size_bytes = $archiveItem.Length
            archive_sha256 = $archiveHash
            extract_root = $bundleId
            install_subdir = "$platformKey/r$Revision"
            payload_file_count = $payloadFiles.Count + 1
            payload_size_bytes = $payloadBytes
            required_files = $requiredFiles
            licensing_status = "review_required_before_public_redistribution"
        }
    )
    status = "compatibility_bundle_validated"
}
$manifest | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $root "dependencies\manifest.json") -Encoding utf8

Write-Host "Dependency bundle created" -ForegroundColor Green
Write-Host "  Archive : $archivePath"
Write-Host "  Files   : $($payloadFiles.Count + 1)"
Write-Host ("  Size    : {0:N2} MiB" -f ($archiveItem.Length / 1MB))
Write-Host "  SHA-256 : $archiveHash"
