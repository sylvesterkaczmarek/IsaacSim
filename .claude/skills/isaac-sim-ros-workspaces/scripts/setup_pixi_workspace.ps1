# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Clone and build the ROS 2 Jazzy workspace with Pixi on Windows 11 x64.

[CmdletBinding()]
param(
    [string]$RepoUrl = "https://github.com/isaac-sim/IsaacSim-ros_workspaces.git",
    [string]$RepoPath = "C:\IsaacSim-ros_workspaces",
    [string]$Branch = "main",
    [switch]$CleanReclone,
    [switch]$YesIKnow,
    [switch]$SkipBuild
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($YesIKnow -and -not $CleanReclone) {
    Write-Error "-YesIKnow is only valid together with -CleanReclone"
    exit 1
}

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Stop-WithError {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    $env:PATH = "$machinePath;$userPath"
}

function Install-WingetPackage {
    param(
        [string]$PackageId,
        [string]$DisplayName
    )

    Write-Step "Installing $DisplayName with winget"
    & winget install --id $PackageId --exact --silent `
        --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "winget failed to install $DisplayName (exit code $LASTEXITCODE)."
    }
    Refresh-Path
}

Write-Step "Checking Windows 11 x64"
$version = [Environment]::OSVersion.Version
if ($version.Build -lt 22000) {
    Stop-WithError "Windows 11 build 22000 or newer is required. Detected build $($version.Build)."
}

$architecture = [Runtime.InteropServices.RuntimeInformation]::OSArchitecture
if ($architecture -ne [Runtime.InteropServices.Architecture]::X64) {
    Stop-WithError "Windows x64 is required. Detected architecture: $architecture."
}

if (-not (Test-Command "winget")) {
    Stop-WithError "winget is required. Install App Installer from the Microsoft Store, then retry."
}

Write-Step "Checking MSVC Build Tools 2022"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    Stop-WithError "MSVC Build Tools 2022 with the Desktop development with C++ workload is required. Install it, then retry."
}

$msvcPath = & $vswhere `
    -products * `
    -version "[17.0,18.0)" `
    -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
    -property installationPath `
    -latest

if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace(($msvcPath | Out-String))) {
    Stop-WithError "MSVC Build Tools 2022 with the Desktop development with C++ workload was not found."
}

if (-not (Test-Command "git")) {
    Install-WingetPackage -PackageId "Git.Git" -DisplayName "Git for Windows"
}
if (-not (Test-Command "git")) {
    Stop-WithError "Git was installed but is not available in this terminal. Open a new PowerShell and retry."
}

if (-not (Test-Command "pixi")) {
    Install-WingetPackage -PackageId "prefix-dev.pixi" -DisplayName "Pixi"
}
if (-not (Test-Command "pixi")) {
    Stop-WithError "Pixi was installed but is not available in this terminal. Open a new PowerShell and retry."
}

Write-Step "Preparing IsaacSim-ros_workspaces"
if (-not (Test-Path $RepoPath)) {
    & git clone --branch $Branch $RepoUrl $RepoPath
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "Failed to clone '$RepoUrl' branch '$Branch'."
    }
} else {
    if (-not (Test-Path (Join-Path $RepoPath ".git"))) {
        Stop-WithError "Existing repository path is not a Git checkout: $RepoPath"
    }

    if ($CleanReclone) {
        $repoItem = Get-Item -LiteralPath $RepoPath -Force
        if (($repoItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            Stop-WithError "Refusing to clean a symbolic-link or reparse-point repository path: $RepoPath"
        }

        $resolvedPath = [IO.Path]::GetFullPath($repoItem.FullName)
        $pathRoot = [IO.Path]::GetPathRoot($resolvedPath)
        $userHome = [IO.Path]::GetFullPath($env:USERPROFILE).TrimEnd('\', '/')
        if (
            [string]::Equals($resolvedPath, $pathRoot, [StringComparison]::OrdinalIgnoreCase) -or
            [string]::Equals($resolvedPath.TrimEnd('\', '/'), $userHome, [StringComparison]::OrdinalIgnoreCase)
        ) {
            Stop-WithError "Refusing to clean unsafe repository path: $resolvedPath"
        }

        Write-Warning "About to permanently delete the Git checkout at:`n  $resolvedPath`nLocal and untracked changes will be lost."
        $stdinIsTty = $false
        try {
            $stdinIsTty = [Console]::IsInputRedirected -eq $false -and [Environment]::UserInteractive
        } catch {
            $stdinIsTty = $false
        }
        if ($stdinIsTty) {
            $reply = Read-Host "Type y to confirm deletion"
            if ($reply -ne 'y' -and $reply -ne 'Y') {
                Stop-WithError "Clean reclone cancelled (did not receive y)."
            }
        } elseif (-not $YesIKnow) {
            Stop-WithError "Non-interactive -CleanReclone requires -YesIKnow (path: $resolvedPath)."
        } else {
            Write-Step "Non-interactive confirmation accepted via -YesIKnow"
        }

        Write-Step "Removing existing checkout at $resolvedPath (-CleanReclone)"
        Remove-Item -LiteralPath $resolvedPath -Recurse -Force
        Write-Step "Cloning clean branch '$Branch' from $RepoUrl"
        & git clone --branch $Branch $RepoUrl $resolvedPath
        if ($LASTEXITCODE -ne 0) {
            Stop-WithError "Failed to clone '$RepoUrl' branch '$Branch'."
        }
        $RepoPath = $resolvedPath
    } else {
        $currentBranch = (& git -C $RepoPath branch --show-current).Trim()
        if ($LASTEXITCODE -ne 0) {
            Stop-WithError "Failed to inspect the existing repository at $RepoPath."
        }
        if ($currentBranch -and $currentBranch -ne $Branch) {
            Stop-WithError "Existing checkout is on branch '$currentBranch', but '$Branch' was requested. Reuse it with -Branch '$currentBranch', choose another -RepoPath, or use -CleanReclone only after the user confirms local changes may be deleted."
        }

        $status = & git -C $RepoPath status --porcelain --untracked-files=normal
        if ($LASTEXITCODE -ne 0) {
            Stop-WithError "Failed to inspect the existing repository status at $RepoPath."
        }
        if ($status) {
            Write-Warning "Reusing an existing checkout with local or untracked changes: $RepoPath"
        }
        Write-Step "Using existing repository at $RepoPath"
    }
}

& git -C $RepoPath submodule update --init --recursive
if ($LASTEXITCODE -ne 0) {
    Stop-WithError "Failed to initialize repository submodules."
}

$workspace = Join-Path $RepoPath "jazzy_ws"
$pixiToml = Join-Path $workspace "pixi.toml"
if (-not (Test-Path $pixiToml)) {
    Stop-WithError "Jazzy Pixi project not found: $pixiToml"
}

if ($RepoPath.Length -gt 40) {
    Write-Warning "The repository path is long and may exceed Windows MAX_PATH during the build. Prefer C:\IsaacSim-ros_workspaces."
}

if ($SkipBuild) {
    Write-Host "`nWorkspace prepared at $workspace (build skipped)." -ForegroundColor Green
    exit 0
}

Push-Location $workspace
try {
    Write-Step "Installing Jazzy workspace dependencies with Pixi"
    & pixi install
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "pixi install failed."
    }

    Write-Step "Building Jazzy workspace with Pixi and MSVC"
    & pixi run build
    if ($LASTEXITCODE -ne 0) {
        Stop-WithError "pixi run build failed."
    }
} finally {
    Pop-Location
}

Write-Host "`nPixi build complete: $workspace" -ForegroundColor Green
Write-Host "Use ROS commands with 'pixi run <command>' or enter 'pixi shell'."
