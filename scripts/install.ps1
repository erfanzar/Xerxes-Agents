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

# Windows counterpart to install.sh: install the locked Bun workspace and the
# production launchers.
#
# Deliberate differences from the shell installer, all of them Windows facts
# rather than preferences:
#   * Launchers are .cmd shims, because PATH entries on Windows are resolved
#     through PATHEXT and an extensionless file is not executable.
#   * PATH is persisted to the user environment via .NET rather than by appending
#     to a shell rc file; cmd and PowerShell do not share one.
#   * Running processes are found with Get-CimInstance, because there is no `ps`.

#Requires -Version 5.1
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepositoryUrl = 'https://github.com/erfanzar/Xerxes-Agents.git'
$InstallDirectory = if ($env:XERXES_INSTALL_DIRECTORY) {
  $env:XERXES_INSTALL_DIRECTORY
} else {
  Join-Path $env:USERPROFILE '.xerxes-bun'
}
$BinDirectory = if ($env:XERXES_BIN_DIRECTORY) {
  $env:XERXES_BIN_DIRECTORY
} else {
  Join-Path $env:LOCALAPPDATA 'Xerxes\bin'
}

# Presentation. Write-Host honours the host's colour capability and is a no-op for
# colour when output is redirected, so this needs no NO_COLOR branch of its own —
# but it is still respected, because a user who sets it means it everywhere.
$script:UseColor = -not $env:NO_COLOR
$script:StepIndex = 0
$script:TotalSteps = 6

function Write-Banner {
  $rule = '-' * 64
  if ($script:UseColor) {
    Write-Host 'Xerxes installer ' -ForegroundColor Cyan -NoNewline
    Write-Host $rule -ForegroundColor DarkGray
  } else {
    Write-Host "Xerxes installer $rule"
  }
}

# A numbered phase heading, so a reader watching a long install knows both where
# they are and how much is left.
function Write-Info {
  param([string] $Message)
  $script:StepIndex++
  $prefix = "> [$script:StepIndex/$script:TotalSteps]"
  if ($script:UseColor) {
    Write-Host $prefix -ForegroundColor Blue -NoNewline
    Write-Host " $Message"
  } else {
    Write-Host "$prefix $Message"
  }
}

function Write-Ok {
  param([string] $Message)
  if ($script:UseColor) {
    Write-Host '  ok' -ForegroundColor Green -NoNewline
    Write-Host " $Message"
  } else {
    Write-Host "  ok $Message"
  }
}

function Write-Note {
  param([string] $Message)
  if ($script:UseColor) {
    Write-Host "  $Message" -ForegroundColor DarkGray
  } else {
    Write-Host "  $Message"
  }
}

function Stop-WithError { param([string] $Message) throw $Message }

function Assert-Command {
  param([string] $Name)
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    Stop-WithError "required command not found: $Name"
  }
}

# Bun is the runtime everything else depends on, so bootstrap it rather than
# telling the user to go install it and come back. Auto-installing a language
# runtime is a real side effect, so it is announced and can be declined with
# XERXES_SKIP_BUN_INSTALL=1.
function Install-BunIfMissing {
  $existing = Get-Command bun -ErrorAction SilentlyContinue
  if ($existing) {
    Write-Ok "bun $(& bun --version 2>$null)"
    return
  }
  if ($env:XERXES_SKIP_BUN_INSTALL -eq '1') {
    Stop-WithError 'bun is not installed and XERXES_SKIP_BUN_INSTALL=1; install Bun from https://bun.sh and re-run'
  }

  Write-Note 'bun was not found; installing it from https://bun.sh'
  # The official Windows installer is a PowerShell script served over HTTPS.
  # Invoke it in a child powershell so its own $ErrorActionPreference and any
  # exit call cannot terminate this installer mid-run. Output is captured so a
  # successful install stays quiet and replayed on failure: this is the step most
  # likely to fail (network, proxy, TLS policy) and a bare "it failed" leaves the
  # user nothing to act on.
  # `$ErrorActionPreference = "Stop"` inside the child is load-bearing: with the
  # default Continue, a failed `irm` is non-terminating, `iex` receives nothing,
  # and the child exits 0 — so a network or TLS failure would be invisible here
  # and resurface later as a confusing "bun is not on PATH". The outer string is
  # single-quoted so the parent does not expand the variable itself.
  $bunLog = & powershell -NoProfile -Command '$ErrorActionPreference = "Stop"; irm bun.sh/install.ps1 | iex' 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Note '--- Bun installer output ---'
    $bunLog | ForEach-Object { Write-Host $_ }
    Stop-WithError 'the Bun installer failed; install Bun manually from https://bun.sh and re-run'
  }

  # The installer edits the persisted user PATH, which cannot affect this already
  # running process, so the new binary is added to this session explicitly.
  $bunRoot = if ($env:BUN_INSTALL) { $env:BUN_INSTALL } else { Join-Path $env:USERPROFILE '.bun' }
  $bunBin = Join-Path $bunRoot 'bin'
  if (Test-Path -LiteralPath $bunBin) {
    $env:Path = "$bunBin;$env:Path"
  }
  if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
    Stop-WithError "Bun installed to $bunRoot but is still not on PATH; open a new terminal and re-run"
  }
  Write-Ok "installed bun $(& bun --version 2>$null)"
  Write-Note 'Bun was added to PATH for this run; open a new terminal for other sessions to see it'
}

function Initialize-BinDirectory {
  if (-not $BinDirectory) { Stop-WithError 'XERXES_BIN_DIRECTORY cannot be empty' }
  if (-not [System.IO.Path]::IsPathRooted($BinDirectory)) {
    Stop-WithError "XERXES_BIN_DIRECTORY must be an absolute path: $BinDirectory"
  }
  # A launcher directory containing ';' would corrupt every PATH it is added to,
  # and one containing control characters cannot round-trip through the registry.
  if ($BinDirectory.Contains(';')) {
    Stop-WithError "XERXES_BIN_DIRECTORY cannot contain a semicolon: $BinDirectory"
  }
  if ($BinDirectory -match '[\x00-\x1f]') {
    Stop-WithError 'XERXES_BIN_DIRECTORY cannot contain control characters'
  }
  New-Item -ItemType Directory -Force -Path $BinDirectory | Out-Null
  return (Resolve-Path -LiteralPath $BinDirectory).ProviderPath
}

function Test-CheckoutRoot {
  param([string] $Candidate)
  return (Test-Path -LiteralPath (Join-Path $Candidate 'package.json')) `
    -and (Test-Path -LiteralPath (Join-Path $Candidate 'bun.lock')) `
    -and (Test-Path -LiteralPath (Join-Path $Candidate 'xerxes'))
}

function Resolve-Source {
  if ($env:XERXES_SOURCE_DIRECTORY) {
    if (-not (Test-Path -LiteralPath $env:XERXES_SOURCE_DIRECTORY)) {
      Stop-WithError "XERXES_SOURCE_DIRECTORY does not exist: $($env:XERXES_SOURCE_DIRECTORY)"
    }
    return (Resolve-Path -LiteralPath $env:XERXES_SOURCE_DIRECTORY).ProviderPath
  }

  $scriptRoot = Split-Path -Parent $PSCommandPath
  $localRoot = (Resolve-Path -LiteralPath (Join-Path $scriptRoot '..')).ProviderPath
  if (Test-CheckoutRoot $localRoot) { return $localRoot }

  Assert-Command git
  $expectedRemote = if ($env:XERXES_REPOSITORY_URL) { $env:XERXES_REPOSITORY_URL } else { $RepositoryUrl }

  if (Test-Path -LiteralPath $InstallDirectory) {
    $managedRoot = (Resolve-Path -LiteralPath $InstallDirectory).ProviderPath
    if (-not (Test-CheckoutRoot $managedRoot)) {
      Stop-WithError "managed checkout is incomplete: $managedRoot"
    }
    # Refuse to touch a checkout that is not the one this installer manages, is
    # not on main, or has local work. Updating it could destroy the user's edits.
    $actualRemote = (& git -C $managedRoot remote get-url origin 2>$null)
    if ($LASTEXITCODE -ne 0) { Stop-WithError "managed checkout has no origin remote: $managedRoot" }
    if ($actualRemote.Trim() -ne $expectedRemote) {
      Stop-WithError "managed checkout origin does not match ${expectedRemote}: $($actualRemote.Trim())"
    }
    $branch = (& git -C $managedRoot symbolic-ref --quiet --short HEAD 2>$null)
    if ($LASTEXITCODE -ne 0) { Stop-WithError "managed checkout is detached; refusing to update: $managedRoot" }
    if ($branch.Trim() -ne 'main') {
      Stop-WithError "managed checkout is on $($branch.Trim()), expected main: $managedRoot"
    }
    $status = (& git -C $managedRoot status --porcelain --untracked-files=normal)
    if ($status) { Stop-WithError "managed checkout has local changes; refusing to update: $managedRoot" }

    Write-Info "updating native Bun source in $managedRoot"
    & git -C $managedRoot pull --ff-only origin main
    if ($LASTEXITCODE -ne 0) { Stop-WithError "managed checkout cannot be fast-forwarded: $managedRoot" }
    return $managedRoot
  }

  Write-Info "cloning native Bun source into $InstallDirectory"
  & git clone --depth 1 $expectedRemote $InstallDirectory
  if ($LASTEXITCODE -ne 0) { Stop-WithError "could not clone native Bun source into $InstallDirectory" }
  return (Resolve-Path -LiteralPath $InstallDirectory).ProviderPath
}

function Write-Launcher {
  param(
    [string] $SourceRoot,
    [string] $LauncherName,
    [string] $CommandPrefix = ''
  )
  if ($CommandPrefix -notin @('', 'acp')) {
    Stop-WithError "unsupported launcher command prefix: $CommandPrefix"
  }
  $entry = Join-Path $SourceRoot 'xerxes\dist\cli.js'
  $launcher = Join-Path $script:ResolvedBinDirectory "$LauncherName.cmd"
  $prefix = if ($CommandPrefix) { " $CommandPrefix" } else { '' }
  # `@echo off` keeps the shim silent; %* forwards every argument unparsed.
  $lines = @(
    '@echo off',
    "bun `"$entry`"$prefix %*"
  )
  # ASCII avoids a UTF-8 BOM, which cmd.exe would try to execute as a command.
  Set-Content -LiteralPath $launcher -Value $lines -Encoding Ascii
  Write-Ok "installed native launcher at $launcher"
}

function Add-BinDirectoryToUserPath {
  $target = [System.EnvironmentVariableTarget]::User
  $current = [System.Environment]::GetEnvironmentVariable('Path', $target)
  $entries = @()
  if ($current) {
    $entries = $current.Split(';') | Where-Object { $_ -ne '' }
  }
  if ($entries -contains $script:ResolvedBinDirectory) {
    Write-Ok "$script:ResolvedBinDirectory is already on the user PATH"
    return
  }
  $updated = (@($script:ResolvedBinDirectory) + $entries) -join ';'
  [System.Environment]::SetEnvironmentVariable('Path', $updated, $target)
  # Also update this session so the verification call below can find the shim.
  $env:Path = "$script:ResolvedBinDirectory;$env:Path"
  Write-Ok "added $script:ResolvedBinDirectory to the user PATH"
}

function Warn-RunningXerxesProcesses {
  param([string] $SourceRoot)
  $cliEntry = Join-Path $SourceRoot 'xerxes\dist\cli.js'
  $uiEntry = Join-Path $SourceRoot 'xerxes\dist\ui\entry.js'
  # There is no `ps` on Windows; CIM is what reports a command line.
  $running = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and ($_.CommandLine.Contains($cliEntry) -or $_.CommandLine.Contains($uiEntry))
  })
  if ($running.Count -eq 0) { return }
  Write-Warning "$($running.Count) running Xerxes process(es) still have the previous build loaded."
  Write-Warning 'Exit open Xerxes TUI/daemon processes, then launch xerxes again to use this install.'
  Write-Warning 'The installer leaves active sessions running so it cannot destroy in-progress work.'
}

function Invoke-Install {
  Write-Banner

  Write-Info 'checking prerequisites'
  Install-BunIfMissing
  $script:ResolvedBinDirectory = Initialize-BinDirectory
  Write-Ok "launchers will be installed to $script:ResolvedBinDirectory"

  Write-Info 'resolving source'
  $sourceRoot = Resolve-Source
  if (-not (Test-CheckoutRoot $sourceRoot)) {
    Stop-WithError "native package manifest or lockfile is missing under: $sourceRoot"
  }
  Write-Ok $sourceRoot

  Write-Info 'installing locked workspace dependencies'
  Push-Location $sourceRoot
  try {
    & bun install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) { Stop-WithError 'bun install failed' }
  } finally {
    Pop-Location
  }
  Write-Ok 'dependencies installed from the lockfile'

  Write-Info 'building the runtime and terminal interface'
  Push-Location $sourceRoot
  try {
    & bun run build
    if ($LASTEXITCODE -ne 0) { Stop-WithError 'bun run build failed' }
  } finally {
    Pop-Location
  }
  foreach ($artifact in @('xerxes\dist\cli.js', 'xerxes\dist\ui\entry.js')) {
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot $artifact))) {
      Stop-WithError "build output is missing: $(Join-Path $sourceRoot $artifact)"
    }
  }
  Write-Ok 'runtime and TUI built'

  Write-Info 'installing launchers'
  Write-Launcher -SourceRoot $sourceRoot -LauncherName 'xerxes'
  Write-Launcher -SourceRoot $sourceRoot -LauncherName 'xerxes-acp' -CommandPrefix 'acp'
  Add-BinDirectoryToUserPath

  Write-Info 'verifying the installation'
  & (Join-Path $script:ResolvedBinDirectory 'xerxes.cmd') --help | Out-Null
  if ($LASTEXITCODE -ne 0) { Stop-WithError 'the installed launcher did not run successfully' }
  Write-Ok 'launcher runs'
  Warn-RunningXerxesProcesses -SourceRoot $sourceRoot

  Write-Host ''
  if ($script:UseColor) {
    Write-Host 'ok Xerxes is ready.' -ForegroundColor Green
  } else {
    Write-Host 'ok Xerxes is ready.'
  }
  Write-Note 'Open a new terminal, then run xerxes'
}

if ($env:XERXES_INSTALLER_SOURCE_ONLY -ne '1') {
  Invoke-Install
}
