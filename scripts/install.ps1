# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Windows (PowerShell) counterpart of scripts/install.sh: installs the locked
# Bun workspace and production .cmd launchers, then persists the launcher
# directory on the user PATH.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install.ps1

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoUrl = 'https://github.com/erfanzar/Xerxes-Agents.git'
$InstallDirectory = if ($env:XERXES_INSTALL_DIRECTORY) { $env:XERXES_INSTALL_DIRECTORY } else { Join-Path $HOME '.xerxes-bun' }
$BinDirectory = if ($env:XERXES_BIN_DIRECTORY) { $env:XERXES_BIN_DIRECTORY } else { Join-Path $HOME '.local\bin' }

function Info([string]$Message) { Write-Host "==> $Message" }
function Ok([string]$Message) { Write-Host "✓ $Message" }
function Die([string]$Message) { Write-Host "x $Message" -ForegroundColor Red; exit 1 }

function Resolve-LocalCheckoutRoot([string]$ScriptPath) {
    $scriptDirectory = Split-Path -Parent (Resolve-Path $ScriptPath).Path
    $repositoryRoot = Split-Path -Parent $scriptDirectory
    if ((Test-Path (Join-Path $repositoryRoot 'package.json')) -and
        (Test-Path (Join-Path $repositoryRoot 'bun.lock')) -and
        (Test-Path (Join-Path $repositoryRoot 'xerxes'))) {
        return $repositoryRoot
    }
    return $null
}

function Resolve-Source {
    if ($env:XERXES_SOURCE_DIRECTORY) {
        if (-not (Test-Path $env:XERXES_SOURCE_DIRECTORY)) { Die "XERXES_SOURCE_DIRECTORY does not exist: $env:XERXES_SOURCE_DIRECTORY" }
        return (Resolve-Path $env:XERXES_SOURCE_DIRECTORY).Path
    }
    $localRoot = Resolve-LocalCheckoutRoot $PSCommandPath
    if ($localRoot) { return $localRoot }

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Die 'required command not found: git' }
    $expectedRemote = if ($env:XERXES_REPOSITORY_URL) { $env:XERXES_REPOSITORY_URL } else { $RepoUrl }
    if (Test-Path $InstallDirectory) {
        $managedRoot = (Resolve-Path $InstallDirectory).Path
        $gitRoot = (git -C $managedRoot rev-parse --show-toplevel 2>$null)
        if ($LASTEXITCODE -ne 0) { Die "install directory is not a managed Git checkout: $managedRoot" }
        if ($gitRoot -ne $managedRoot) { Die "install directory is nested inside another Git checkout: $managedRoot" }
        $actualRemote = (git -C $managedRoot remote get-url origin 2>$null)
        if ($actualRemote -ne $expectedRemote) { Die "managed checkout origin does not match ${expectedRemote}: $actualRemote" }
        $managedBranch = (git -C $managedRoot symbolic-ref --quiet --short HEAD 2>$null)
        if ($managedBranch -ne 'main') { Die "managed checkout is not on main: $managedRoot" }
        if ((git -C $managedRoot status --porcelain --untracked-files=normal)) {
            Die "managed checkout has local changes; refusing to update: $managedRoot"
        }
        Info "updating native Bun source in $managedRoot"
        git -C $managedRoot pull --ff-only origin main
        if ($LASTEXITCODE -ne 0) { Die "managed checkout cannot be fast-forwarded: $managedRoot" }
        return $managedRoot
    }
    Info "cloning native Bun source into $InstallDirectory"
    git clone --depth 1 $expectedRemote $InstallDirectory
    if ($LASTEXITCODE -ne 0) { Die "could not clone native Bun source into $InstallDirectory" }
    return (Resolve-Path $InstallDirectory).Path
}

function Write-Launcher([string]$SourceRoot, [string]$LauncherName, [string]$CommandPrefix) {
    $launcher = Join-Path $BinDirectory "$LauncherName.cmd"
    $entry = Join-Path $SourceRoot 'xerxes\dist\cli.js'
    $temporary = "$launcher.tmp.$PID"
    $invocation = if ($CommandPrefix) { "bun `"$entry`" $CommandPrefix %*" } else { "bun `"$entry`" %*" }
    Set-Content -Path $temporary -Value "@echo off", $invocation -Encoding ascii
    Move-Item -Force $temporary $launcher
    Ok "installed native launcher at $launcher"
}

function Persist-BinPath {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $entries = @($userPath -split ';' | Where-Object { $_ })
    $alreadyOnPath = $entries | Where-Object { $_.TrimEnd('\') -ieq $BinDirectory.TrimEnd('\') }
    if (-not $alreadyOnPath) {
        [Environment]::SetEnvironmentVariable('Path', (@($entries) + $BinDirectory) -join ';', 'User')
        Ok "added $BinDirectory to the user PATH (open a new terminal to pick it up)"
    }

    # Keep an idempotent, managed block in the PowerShell profile as well so
    # interactive sessions work even when PATH was inherited from elsewhere.
    $profilePath = $PROFILE.CurrentUserAllHosts
    $profileDirectory = Split-Path -Parent $profilePath
    if (-not (Test-Path $profileDirectory)) { New-Item -ItemType Directory -Force $profileDirectory | Out-Null }
    if (-not (Test-Path $profilePath)) { New-Item -ItemType File -Force $profilePath | Out-Null }
    $content = Get-Content $profilePath -Raw
    $escaped = [regex]::Escape($BinDirectory)
    $block = "# >>> xerxes PATH >>>`n`$env:Path = `"$BinDirectory;`$env:Path`"`n# <<< xerxes PATH <<<"
    if ($content -match '(?ms)# >>> xerxes PATH >>>.*?# <<< xerxes PATH <<<') {
        $updated = [regex]::Replace($content, '(?ms)# >>> xerxes PATH >>>.*?# <<< xerxes PATH <<<', $block)
        if ($updated -ne $content) { Set-Content -Path $profilePath -Value $updated -Encoding utf8 }
    } elseif ($content -notmatch $escaped) {
        Add-Content -Path $profilePath -Value "`n$block" -Encoding utf8
    }
    Ok "persisted launcher directory in $profilePath"
}

function Warn-RunningXerxesProcesses([string]$SourceRoot) {
    $cliEntry = (Join-Path $SourceRoot 'xerxes\dist\cli.js').ToLowerInvariant()
    $uiEntry = (Join-Path $SourceRoot 'xerxes\dist\ui\entry.js').ToLowerInvariant()
    $running = @(Get-CimInstance Win32_Process -Filter "Name = 'bun.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine.ToLowerInvariant().Contains($cliEntry) -or $_.CommandLine.ToLowerInvariant().Contains($uiEntry)) })
    if ($running.Count -gt 0) {
        Write-Host "! $($running.Count) running Xerxes process(es) still have the previous build loaded."
        Write-Host '! Exit open Xerxes TUI/daemon processes, then launch xerxes again to use this install.'
    }
}

if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
    Die 'required command not found: bun (install it first: powershell -c "irm bun.sh/install.ps1 | iex")'
}
if ($BinDirectory -match '[\r\n]') { Die "XERXES_BIN_DIRECTORY cannot contain control characters: $BinDirectory" }
if (-not [System.IO.Path]::IsPathRooted($BinDirectory)) { Die "XERXES_BIN_DIRECTORY must be an absolute path: $BinDirectory" }
New-Item -ItemType Directory -Force $BinDirectory | Out-Null
$BinDirectory = (Resolve-Path $BinDirectory).Path

$sourceRoot = Resolve-Source
if (-not (Test-Path (Join-Path $sourceRoot 'package.json'))) { Die "native package manifest is missing: $sourceRoot/package.json" }
if (-not (Test-Path (Join-Path $sourceRoot 'bun.lock'))) { Die "native lockfile is missing: $sourceRoot/bun.lock" }

Info 'installing locked Bun workspace dependencies'
Push-Location $sourceRoot
try {
    bun install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) { Die 'bun install --frozen-lockfile failed' }
    bun run build
    if ($LASTEXITCODE -ne 0) { Die 'bun run build failed' }
} finally {
    Pop-Location
}
if (-not (Test-Path (Join-Path $sourceRoot 'xerxes\dist\cli.js'))) { Die "runtime build is missing: $sourceRoot/xerxes/dist/cli.js" }
if (-not (Test-Path (Join-Path $sourceRoot 'xerxes\dist\ui\entry.js'))) { Die "TUI build is missing: $sourceRoot/xerxes/dist/ui/entry.js" }

Write-Launcher $sourceRoot 'xerxes' ''
Write-Launcher $sourceRoot 'xerxes-acp' 'acp'
Persist-BinPath
Warn-RunningXerxesProcesses $sourceRoot
& (Join-Path $BinDirectory 'xerxes.cmd') --help | Out-Null
if ($LASTEXITCODE -ne 0) { Die 'installed launcher failed to run' }
Ok 'Xerxes Bun runtime is ready; open a new terminal to invoke xerxes'
