<#
.SYNOPSIS
Update NeuroSeq from origin/main, then start it (issue #617).

.DESCRIPTION
Run by the desktop shortcut that install_shortcut.ps1 creates. Only a clone on
`main` without changes to tracked files is updated, and only by fast-forward;
otherwise, or when GitHub cannot be reached, the current version starts and a
short notice says why it was not updated. `uv run` keeps the clone's .venv in
sync with uv.lock before starting NeuroSeq.

Output of each start is written to %LOCALAPPDATA%\neuroseq\launcher.log.
Works with Windows PowerShell 5.1 and PowerShell 7.
#>
param(
    # uv.exe to use; the shortcut passes the one found at install time,
    # because a conda environment's uv is not on PATH outside its prompt.
    [string]$Uv = "",
    # Skip the update, e.g. during an experiment.
    [switch]$NoUpdate,
    # Show notices on the console instead of popups (for testing).
    [switch]$Console
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$logDir = Join-Path $env:LOCALAPPDATA "neuroseq"
if (-not $env:LOCALAPPDATA) { $logDir = Join-Path $repo "tmp" }
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "launcher.log"
Set-Content -Path $log -Value "NeuroSeq launcher $(Get-Date -Format s) in $repo" -Encoding UTF8

function Write-Log([string]$text) {
    Add-Content -Path $log -Value $text -Encoding UTF8
}

function Show-Notice([string]$text, [int]$seconds, [int]$icon) {
    Write-Log $text
    if ($Console) { Write-Host $text; return }
    # Closes by itself after $seconds; 48 = warning icon, 16 = error icon.
    $shell = New-Object -ComObject WScript.Shell
    $shell.Popup($text, $seconds, "NeuroSeq", $icon) | Out-Null
}

# Run git in the clone; returns its output lines and sets $script:gitOk.
function Invoke-Git([string[]]$arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & git -C $repo @arguments 2>&1 | ForEach-Object { "$_" }
    } finally {
        $ErrorActionPreference = $previous
    }
    $script:gitOk = ($LASTEXITCODE -eq 0)
    Write-Log ("git " + ($arguments -join " ") + " -> exit $LASTEXITCODE")
    $out | ForEach-Object { Write-Log "  $_" }
    return $out
}

function Update-Clone {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        return "Git was not found, so NeuroSeq was not updated."
    }
    $branch = (Invoke-Git @("branch", "--show-current")) -join ""
    if ($branch -ne "main") {
        return "NeuroSeq was not updated: the clone is on branch '$branch', not 'main'."
    }
    $changes = Invoke-Git @("status", "--porcelain", "--untracked-files=no")
    if ($changes) {
        return "NeuroSeq was not updated: the clone has local changes ($repo)."
    }
    # Never wait for a login prompt; the public repository needs none.
    $env:GIT_TERMINAL_PROMPT = "0"
    Invoke-Git @("fetch", "--quiet", "origin", "main") | Out-Null
    if (-not $script:gitOk) {
        return "NeuroSeq was not updated: GitHub could not be reached."
    }
    Invoke-Git @("merge", "--ff-only", "--quiet", "origin/main") | Out-Null
    if (-not $script:gitOk) {
        return "NeuroSeq was not updated: the local 'main' has commits that are not on GitHub."
    }
    return ""
}

function Find-Uv {
    if ($Uv -and (Test-Path $Uv)) { return $Uv }
    $found = Get-Command uv -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    if (-not $env:USERPROFILE) { return "" }
    foreach ($candidate in @(
            (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
            (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"))) {
        if (Test-Path $candidate) { return $candidate }
    }
    return ""
}

$notice = ""
if ($NoUpdate) {
    Write-Log "Update skipped (-NoUpdate)."
} else {
    try {
        $notice = Update-Clone
    } catch {
        $notice = "NeuroSeq was not updated: $($_.Exception.Message)"
    }
}
$head = (Invoke-Git @("log", "--oneline", "-1")) -join ""

$uvExe = Find-Uv
if (-not $uvExe) {
    Show-Notice "uv was not found. Re-run install_shortcut.ps1 from a prompt where 'uv' works.`n`nLog: $log" 0 16
    exit 1
}
Write-Log "Starting with $uvExe at $head"
$appOut = Join-Path $logDir "neuroseq.out.log"
$appErr = Join-Path $logDir "neuroseq.err.log"
$start = @{
    FilePath               = $uvExe
    ArgumentList           = @("run", "--project", "`"$repo`"", "neuroseq")
    WorkingDirectory       = $repo
    PassThru               = $true
    RedirectStandardOutput = $appOut
    RedirectStandardError  = $appErr
}
# Hidden is Windows-only; elsewhere (tests) there is no console to hide.
if ($env:OS -eq "Windows_NT") { $start.WindowStyle = "Hidden" }
$process = Start-Process @start
# Windows PowerShell 5.1 reports no ExitCode unless the handle is opened early.
$null = $process.Handle
if ($notice) { Show-Notice "$notice`n`nStarting the current version:`n$head" 10 48 }
$process.WaitForExit()
Write-Log "NeuroSeq exited with code $($process.ExitCode)"
if ($process.ExitCode -ne 0) {
    $tail = (Get-Content $appErr -Tail 15 -ErrorAction SilentlyContinue) -join "`n"
    Show-Notice "NeuroSeq stopped with an error (exit code $($process.ExitCode)).`n`n$tail`n`nFull output: $appErr" 0 16
}
exit $process.ExitCode
