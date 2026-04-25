$ErrorActionPreference = "Continue"

$machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "$machinePath;$userPath"

function Show-CommandStatus {
    param(
        [string]$Name,
        [string]$Expected = ""
    )

    $resolved = Get-Command $Name -ErrorAction SilentlyContinue
    if ($resolved) {
        if ($Name -eq "python" -and $resolved.Source -like "*WindowsApps*python.exe") {
            Write-Host "[WARN] $Name -> $($resolved.Source)"
            Write-Host "       This looks like the Microsoft Store alias, not a usable Python install."
        } else {
            Write-Host "[FOUND] $Name -> $($resolved.Source)"
            if ($Expected) {
                Write-Host "        Expected: $Expected"
            }
        }
    } else {
        Write-Host "[MISSING] $Name"
        if ($Expected) {
            Write-Host "          Expected: $Expected"
        }
    }
}

Write-Host "Checking command-line tools..."
Show-CommandStatus -Name "python" -Expected "Python 3.12"
Show-CommandStatus -Name "py" -Expected "Python Launcher"
Show-CommandStatus -Name "uv" -Expected "uv 0.11+"
Show-CommandStatus -Name "git" -Expected "Git 2.54+"

$uvWingetPath = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
if (-not (Get-Command uv -ErrorAction SilentlyContinue) -and (Test-Path $uvWingetPath)) {
    Write-Host "[FOUND] uv winget executable -> $uvWingetPath"
    Write-Host "        Restart PowerShell or use the full path if uv is not on PATH yet."
}

$buildToolsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
if (Test-Path $buildToolsPath) {
    Write-Host "[FOUND] Visual Studio Build Tools -> $buildToolsPath"
} else {
    Write-Host "[MISSING] Visual Studio Build Tools"
}

Write-Host ""
Write-Host "Suggested manual checks:"
Write-Host "python --version"
Write-Host "py --list"
Write-Host "git --version"
Write-Host "uv --version"
