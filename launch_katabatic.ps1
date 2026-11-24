param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$KatabaticArgs
)

# Resolve paths from this launcher so it can be called from any working folder.
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptsRoot = Join-Path $projectRoot "Scripts Pythons"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue

if ($null -eq $pythonCommand) {
    throw "Python was not found. Activate the katabatic-offline environment first."
}
if ($pythonCommand.Source -match '\\Microsoft\\WindowsApps\\python(?:3)?\.exe$') {
    throw "Only the Microsoft Store Python alias was found. Install or activate the katabatic-offline environment first."
}

# Make this copy's shared packages available without relying on the source tree.
$existingPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($existingPythonPath)) {
    $env:PYTHONPATH = $scriptsRoot
} else {
    $env:PYTHONPATH = "$scriptsRoot$([IO.Path]::PathSeparator)$existingPythonPath"
}

# Keep Cartopy's downloadable map assets inside this portable project by default.
if ([string]::IsNullOrWhiteSpace($env:CARTOPY_DATA_DIR)) {
    $env:CARTOPY_DATA_DIR = Join-Path $projectRoot "Data\cartopy"
}

Push-Location $scriptsRoot
try {
    & $pythonCommand.Source -m katabatic.run_katabatic @KatabaticArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

exit $exitCode
