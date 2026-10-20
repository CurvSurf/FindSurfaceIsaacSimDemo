param(
    [ValidateSet('Start', 'Remove')][string]$Action = 'Start',
    [string]$InstallBase,
    [string]$IsaacRoot,
    [switch]$PrepareOnly,
    [switch]$LibraryOnly
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$RecordPath = Join-Path $ProjectRoot '.installation\manifest.json'
$RuntimeVersion = '6.0.1'

function Write-JsonAtomic($Value, [string]$Path) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $temporary = $Path + '.tmp'
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Read-Json([string]$Path) {
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-Sha256([string]$Path) {
    $stream = [IO.File]::OpenRead($Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose(); $stream.Dispose() }
}

function Assert-NoReparsePath([string]$Path) {
    $candidate = [IO.Path]::GetFullPath($Path)
    while ($candidate) {
        if (Test-Path -LiteralPath $candidate) {
            if ((Get-Item -LiteralPath $candidate -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing a symlink/junction path: $candidate"
            }
        }
        $candidate = Split-Path -Parent $candidate
    }
}

function Assert-PrivateRoot($Record, [switch]$InspectContents) {
    if ($Record.schema -ne 1 -or $Record.app -ne 'FindSurfaceIsaacSimDemo') { throw 'Unknown installation record.' }
    $id = [guid]::Parse($Record.id).ToString()
    $root = [IO.Path]::GetFullPath($Record.private_root).TrimEnd('\')
    if ((Split-Path -Leaf $root) -ne "fsis-$id") { throw 'Private root name does not match installation ID.' }
    if ([IO.Path]::GetFullPath($Record.project_root).TrimEnd('\') -ne $ProjectRoot) {
        throw 'This record belongs to another/moved checkout. Keep the old record; see INSTALL.md.'
    }
    Assert-NoReparsePath $root
    $marker = Read-Json (Join-Path $root '.fsis-owner.json')
    if ($marker.id -ne $Record.id -or $marker.project_root -ne $Record.project_root) { throw 'Ownership marker mismatch.' }
    if ($InspectContents) {
        Initialize-RuntimeZipExtractor
        if ([CurvSurf.Installation.RuntimeZipExtractor]::ContainsReparsePoints($root)) {
            throw 'Private installation contains symlinks/junctions; manual inspection required.'
        }
    }
    return $root
}

function New-InstallationRecord([string]$Base) {
    $id = [guid]::NewGuid().ToString()
    $root = Join-Path ([IO.Path]::GetFullPath($Base)) "fsis-$id"
    Assert-NoReparsePath $root
    if (Test-Path -LiteralPath $root) { throw 'Refusing to adopt an existing private directory.' }
    New-Item -ItemType Directory -Path $root | Out-Null
    $record = [pscustomobject]@{
        schema = 1; app = 'FindSurfaceIsaacSimDemo'; id = $id
        project_root = $ProjectRoot; private_root = $root
        runtime = $null; sdk = $null
    }
    Write-JsonAtomic @{ id = $id; project_root = $ProjectRoot } (Join-Path $root '.fsis-owner.json')
    Write-JsonAtomic $record $RecordPath
    return $record
}

function Test-Runtime([string]$Root) {
    if (-not $Root) { return $false }
    foreach ($item in @('kit\kit.exe', 'kit\python\python.exe', 'apps', 'VERSION')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root $item))) { return $false }
    }
    return (Get-Content -LiteralPath (Join-Path $Root 'VERSION') -Raw) -match '^\s*6\.0\.1(?:\D|$)'
}

function Find-PrivateRuntime([string]$Base) {
    if (-not $Base -or -not (Test-Path -LiteralPath $Base -PathType Container)) { return $null }
    try { Assert-NoReparsePath $Base } catch { return $null }
    # Only inspect this app's immediate installation folders, not the whole drive.
    foreach ($folder in @(Get-ChildItem -LiteralPath $Base -Directory -ErrorAction Stop | Sort-Object Name)) {
        if ($folder.Name -notmatch '^fsis-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$') { continue }
        $runtime = Join-Path $folder.FullName "isaacsim-$RuntimeVersion"
        try { Assert-NoReparsePath $runtime } catch { continue }
        if (Test-Runtime $runtime) { return [IO.Path]::GetFullPath($runtime) }
    }
    return $null
}

function Find-Runtime([string]$Requested, [string]$PrivateBase) {
    foreach ($root in @($Requested, $env:ISAACSIM_ROOT, 'C:\isaacsim-6.0.1', 'C:\isaacsim')) {
        if (Test-Runtime $root) { return [IO.Path]::GetFullPath($root) }
    }
    $defaultBase = if ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'CurvSurf\FindSurfaceIsaacSimDemo' }
    foreach ($base in @($PrivateBase, $defaultBase) | Where-Object { $_ } | Select-Object -Unique) {
        $runtime = Find-PrivateRuntime $base
        if ($runtime) {
            Write-Host "Reusing existing runtime (ownership not transferred): $runtime"
            return $runtime
        }
    }
    return $null
}

function Receive-VerifiedFile([string]$Url, [string]$Path, [string]$Hash) {
    if ($Url -notmatch '^https://' -or $Hash -notmatch '^[a-fA-F0-9]{64}$') { throw 'Download URL/hash is not pinned.' }
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
        $partial = $Path + '.partial'
        $curl = Get-Command curl.exe -ErrorAction Stop
        & $curl.Source --fail --location --proto '=https' --proto-redir '=https' --retry 3 --continue-at - --output $partial $Url
        if ($LASTEXITCODE -ne 0) { throw 'Download failed; partial file retained for retry.' }
        Write-Host 'Verifying downloaded file SHA-256...'
        if ((Get-Sha256 $partial) -ne $Hash) {
            throw "Hash mismatch; inspect/remove $partial before retrying."
        }
        Move-Item -LiteralPath $partial -Destination $Path
        return
    }
    Write-Host "Verifying cached download SHA-256: $Path"
    if ((Get-Sha256 $Path) -ne $Hash) { throw "Hash mismatch: $Path" }
}

function Initialize-RuntimeZipExtractor {
    if (-not ('CurvSurf.Installation.RuntimeZipExtractor' -as [type])) {
        Add-Type -AssemblyName System.IO.Compression
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        Add-Type -Path (Join-Path $PSScriptRoot 'RuntimeZipExtractor.cs') `
            -ReferencedAssemblies System.IO.Compression,System.IO.Compression.FileSystem
    }
}

function Expand-RuntimeZip([string]$Archive, [string]$Destination) {
    Assert-NoReparsePath $Destination
    Initialize-RuntimeZipExtractor
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $progressState = @{ lastTextSeconds = -5; phase = ''; extractingAt = 0.0 }
    $report = [Action[string,long,long,int,int]] {
        param($phase, $bytes, $totalBytes, $files, $totalFiles)
        $now = $clock.Elapsed.TotalSeconds
        $changedPhase = $phase -ne $progressState.phase
        if ($changedPhase) {
            $progressState.phase = $phase
            $progressState.extractingAt = $now
        }
        $elapsed = $now - $progressState.extractingAt
        if ($phase -eq 'Checking ZIP paths') {
            $percent = if ($totalFiles) { [int][Math]::Floor(100.0 * $files / $totalFiles) } else { 0 }
            $status = "Checking archive entries: $files / $totalFiles"
        } else {
            $percent = if ($totalBytes) { [int][Math]::Floor(100.0 * $bytes / $totalBytes) } elseif ($totalFiles) {
                [int][Math]::Floor(100.0 * $files / $totalFiles)
            } else { 100 }
            $remaining = if ($bytes -gt 0 -and $elapsed -ge 1) {
                '~{0:n0}s remaining' -f (($totalBytes - $bytes) * $elapsed / $bytes)
            } else { 'estimating remaining time' }
            $status = '{0}% | {1:n2}/{2:n2} GiB | {3}/{4} files | {5:n0}s elapsed | {6}' -f `
                $percent, ($bytes / 1GB), ($totalBytes / 1GB), $files, $totalFiles, $elapsed, $remaining
        }
        Write-Progress -Id 41 -Activity $phase -Status $status -PercentComplete $percent
        # Also leave periodic text for redirected output / terminals hiding progress bars.
        if ($changedPhase -or $now - $progressState.lastTextSeconds -ge 5 -or
            ($phase -eq 'Extracting ZIP' -and $files -eq $totalFiles)) {
            Write-Host "[$phase] $status"
            $progressState.lastTextSeconds = $now
        }
    }
    try {
        Write-Host 'Reading ZIP directory...'
        return [CurvSurf.Installation.RuntimeZipExtractor]::Extract($Archive, $Destination, $report)
    } finally { Write-Progress -Id 41 -Activity 'Extracting ZIP' -Completed }
}

function Install-PrivateRuntime($Record) {
    $root = Assert-PrivateRoot $Record
    $destination = Join-Path $root "isaacsim-$RuntimeVersion"
    if (Test-Runtime $destination) { return $destination }
    if (Test-Path -LiteralPath $destination) { throw 'Incomplete runtime exists; inspect it before retrying.' }
    $config = Read-Json (Join-Path $PSScriptRoot 'installation_dependencies.json')
    Write-Host "Isaac Sim download: approximately 10.6 GB. Destination: $destination"
    Write-Host 'Review https://docs.isaacsim.omniverse.nvidia.com/6.0.1/common/licenses-isaac-sim.html'
    if ((Read-Host 'Type YES to accept those terms and install') -cne 'YES') { throw 'Installation cancelled.' }
    $drive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($root))
    if ($drive.AvailableFreeSpace -lt 35GB) { throw 'At least 35 GB free space is required for download and extraction.' }
    $archive = Join-Path $root 'downloads\isaacsim.zip'
    Receive-VerifiedFile $config.isaac_url $archive $config.isaac_sha256
    $staging = Join-Path $root ('extract-' + [guid]::NewGuid().ToString())
    Write-Host 'Extracting verified Isaac Sim ZIP (Unicode and long paths supported)...'
    try {
        $files = Expand-RuntimeZip $archive $staging
        if (-not (Test-Runtime $staging)) { throw 'Required runtime files/version are missing.' }
    } catch {
        throw "Extraction failed: $($_.Exception.Message). Archive retained at $archive; staging: $staging"
    }
    Write-Host "Extraction complete: $files files."
    Write-Host 'Checking extracted runtime links...'
    if ([CurvSurf.Installation.RuntimeZipExtractor]::ContainsReparsePoints($staging)) {
        throw 'Runtime archive contains unexpected links; manual inspection required.'
    }
    Move-Item -LiteralPath $staging -Destination $destination
    Write-Host 'Runtime installation complete; checking dependencies before launch...'
    # Keep the verified archive; Remove_Demo removes it with the private runtime.
    return $destination
}

function Assert-NoKitProcess {
    # Conservative: do not delete while any Kit app might use the dependency.
    if (Get-Process -Name kit -ErrorAction SilentlyContinue) { throw 'Close all Kit/Isaac Sim apps before removing dependencies.' }
}

function Remove-Installation($Record, [switch]$Confirmed) {
    $root = Assert-PrivateRoot $Record -InspectContents
    Write-Host "App-private runtime/downloads only: $root"
    foreach ($entry in @($Record.runtime, $Record.sdk)) {
        if ($entry -and $entry.ownership -eq 'external') { Write-Host "PRESERVED external dependency: $($entry.path)" }
    }
    Write-Host 'App source, user data/settings and shared NVIDIA/Packman caches are NOT removed.'
    if (-not $Confirmed) { return }
    Assert-NoKitProcess
    # Revalidate exact owned target immediately before recursive removal.
    $root = Assert-PrivateRoot $Record -InspectContents
    [CurvSurf.Installation.RuntimeZipExtractor]::DeleteOwnedDirectory($root)
    Remove-Item -LiteralPath $RecordPath
}

function Start-Installation {
    $bootstrap = Join-Path $ProjectRoot 'source\extensions\com.findsurface.bootstrap'
    $needsBootstrap = -not (Test-Path -LiteralPath (Join-Path $bootstrap 'config\extension.toml'))
    if ($needsBootstrap) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot '.git'))) {
            throw 'Source ZIP lacks the extension. Use the thin release ZIP or a recursive Git clone.'
        }
        if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) {
            throw 'Git is needed to initialize the pinned extension. Use an assembled release ZIP or install Git.'
        }
    }
    if (Test-Path -LiteralPath $RecordPath) {
        $record = Read-Json $RecordPath
        $null = Assert-PrivateRoot $record
    } else {
        $base = $InstallBase
        if (-not $base) {
            $default = Join-Path $env:LOCALAPPDATA 'CurvSurf\FindSurfaceIsaacSimDemo'
            $base = Read-Host "Private dependency base folder (Enter for $default)"
            if (-not $base) { $base = $default }
        }
        $record = New-InstallationRecord $base
    }
    $root = Assert-PrivateRoot $record
    if ($IsaacRoot) {
        if (-not (Test-Runtime $IsaacRoot)) { throw 'Explicit IsaacRoot is not a complete Isaac Sim 6.0.1 installation.' }
        $runtime = [IO.Path]::GetFullPath($IsaacRoot)
    } elseif ($record.runtime -and (Test-Runtime $record.runtime.path)) {
        $runtime = $record.runtime.path
    } else { $runtime = Find-Runtime $null (Split-Path -Parent $root) }
    if (-not $runtime) { $runtime = Install-PrivateRuntime $record }
    $ownedRuntime = Join-Path $root "isaacsim-$RuntimeVersion"
    $ownership = if ($runtime -eq $ownedRuntime) { 'private' } else { 'external' }
    $record.runtime = @{ path = $runtime; version = $RuntimeVersion; ownership = $ownership }
    Write-JsonAtomic $record $RecordPath
    $python = Join-Path $runtime 'kit\python\python.exe'
    if ($needsBootstrap) {
        # Use the same SDK-preserving initializer as Ubuntu and the developer entrypoint.
        & $python -B (Join-Path $PSScriptRoot 'repo.py') init-dependencies
        if ($LASTEXITCODE -ne 0) { throw 'Pinned submodule initialization failed; local SDK files were preserved. Resolve Git/network access and rerun Start_Demo.bat.' }
    }
    & $python -B (Join-Path $PSScriptRoot 'repo.py') doctor --isaac-sim-root $runtime
    if ($LASTEXITCODE -ne 0) { throw 'Runtime structure/version check failed.' }
    $sdk = Join-Path $bootstrap 'data\native\windows-x86_64\FindSurface.dll'
    if ($record.sdk -and (Test-Path -LiteralPath $record.sdk.path -PathType Leaf)) {
        $sdk = $record.sdk.path
        if ($record.sdk.ownership -eq 'private') {
            if ($sdk -ne (Join-Path $root 'sdk\FindSurface.dll')) { throw 'Private SDK path differs from its installation record.' }
            $actualDigest = Get-Sha256 $sdk
            if ($actualDigest -ne $record.sdk.sha256) {
                $policy = Read-Json (Join-Path $PSScriptRoot 'sdk_releases.json')
                $approvedDigests = @($policy.releases | ForEach-Object { $_.assets.'windows-x86_64'.sha256 })
                if ($actualDigest -notin $approvedDigests) { throw 'Private SDK hash differs from its installation record.' }
                Write-Host 'Recognized an approved SDK replacement; refreshing its ownership digest.'
            }
        }
    } elseif (-not (Test-Path -LiteralPath $sdk -PathType Leaf)) {
        $sdk = Join-Path $root 'sdk\FindSurface.dll'
    }
    $privateSdk = Join-Path $root 'sdk\FindSurface.dll'
    $defaultSdk = Join-Path $bootstrap 'data\native\windows-x86_64\FindSurface.dll'
    if ($sdk -eq $privateSdk -or $sdk -eq $defaultSdk) {
        $updateArguments = @('-B', (Join-Path $PSScriptRoot 'update_sdk.py'), '--sdk-path', $sdk,
                            '--asset-key', 'windows-x86_64')
        if ($sdk -eq $privateSdk) { $updateArguments += '--managed' }
        & $python @updateArguments
        if ($LASTEXITCODE -ne 0) { throw 'FindSurface SDK preparation failed; existing files were preserved.' }
    } else { Write-Host "Preserving explicitly configured external SDK: $sdk" }
    $ownership = if ($sdk -eq (Join-Path $root 'sdk\FindSurface.dll')) { 'private' } else { 'external' }
    $record.sdk = @{ path = $sdk; ownership = $ownership; sha256 = (Get-Sha256 $sdk) }
    Write-JsonAtomic $record $RecordPath
    & $python -B -c 'import ctypes, sys; ctypes.CDLL(sys.argv[1])' $sdk
    if ($LASTEXITCODE -ne 0) { throw 'SDK load check failed; dependency files retained for diagnosis.' }
    Write-Host "Runtime ($($record.runtime.ownership)): $runtime"
    Write-Host "SDK ($ownership): $sdk"
    if ($PrepareOnly) { return }
    & $python -B (Join-Path $PSScriptRoot 'repo.py') launch --isaac-sim-root $runtime -- `
        "--/exts/com.curvsurf.findsurface_isaac_sim/findSurfaceLibraryPath=$sdk"
    if ($LASTEXITCODE -ne 0) { throw "Demo exited with code $LASTEXITCODE" }
}

if ($LibraryOnly) { return }
$installationLock = $null
try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw '64-bit Windows is required.' }
    Assert-NoReparsePath $RecordPath
    New-Item -ItemType Directory -Path (Split-Path -Parent $RecordPath) -Force | Out-Null
    $installationLock = [IO.File]::Open((Join-Path (Split-Path -Parent $RecordPath) 'operation.lock'),
        [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    if ($Action -eq 'Start') { Start-Installation } else {
        if (-not (Test-Path -LiteralPath $RecordPath)) { throw 'No ownership record. Nothing will be automatically removed.' }
        $record = Read-Json $RecordPath
        Remove-Installation $record
        if ((Read-Host 'Type REMOVE to delete the listed app-private folder permanently') -ceq 'REMOVE') {
            Remove-Installation $record -Confirmed
        } else { Write-Host 'Removal cancelled; nothing changed.' }
    }
} catch {
    Write-Error $_ -ErrorAction Continue
    exit 1
} finally {
    if ($installationLock) { $installationLock.Dispose() }
}
