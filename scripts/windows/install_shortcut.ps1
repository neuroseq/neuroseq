<#
.SYNOPSIS
Create the NeuroSeq desktop shortcut (issue #617).

.DESCRIPTION
Run once from a prompt where `uv` works (e.g. the Anaconda Prompt with the
environment that provides uv active), in the NeuroSeq clone:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows\install_shortcut.ps1

The shortcut runs launch_neuroseq.ps1 from this clone, which updates the clone
from origin/main and starts NeuroSeq with the clone's .venv. An older
"neuroseq.exe" desktop shortcut, which starts a separately pip-installed copy,
is removed so only the up-to-date one remains.
#>
param(
    [string]$Uv = "",
    [string]$Name = "NeuroSeq",
    # Keep an existing neuroseq.exe shortcut instead of removing it.
    [switch]$KeepOld
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$repo = (Resolve-Path (Join-Path $here "..\..")).Path
$launcher = Join-Path $here "launch_neuroseq.ps1"
$icon = Join-Path $here "neuroseq.ico"

if (-not $Uv) {
    $found = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $found) {
        throw "uv was not found. Run this from a prompt where 'uv --version' works, or pass -Uv C:\path\to\uv.exe."
    }
    $Uv = $found.Source
}
$Uv = (Resolve-Path $Uv).Path
if (-not (Test-Path (Join-Path $repo "pyproject.toml"))) {
    throw "$repo does not look like a NeuroSeq clone."
}

$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell

if (-not $KeepOld) {
    foreach ($old in Get-ChildItem -Path $desktop -Filter "*.lnk") {
        $target = $shell.CreateShortcut($old.FullName).TargetPath
        if ($target -and ((Split-Path $target -Leaf) -ieq "neuroseq.exe")) {
            Remove-Item $old.FullName
            Write-Host "Removed old shortcut $($old.Name) -> $target"
        }
    }
}

$path = Join-Path $desktop "$Name.lnk"
$powershell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$link = $shell.CreateShortcut($path)
$link.TargetPath = $powershell
$link.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$launcher`" -Uv `"$Uv`""
$link.WorkingDirectory = $repo
$link.IconLocation = "$icon,0"
$link.Description = "Update NeuroSeq from GitHub (main) and start it"
$link.WindowStyle = 7  # minimized, so no console flashes up
$link.Save()

Write-Host "Created $path"
Write-Host "  clone: $repo"
Write-Host "  uv:    $Uv"
