param([Parameter(Mandatory = $true)][string]$Scenario)
$ErrorActionPreference = 'Stop'
$testState = @{ Started = $false; Messages = @() }

# Replace every OS interaction: these tests must never launch a real installer.
function Get-ItemProperty {
    if ($Scenario -eq 'existing' -or ($testState.Started -and $Scenario -ne 'missing-sdk')) {
        return @{ ImagePath = 'vjoy.sys' }
    }
    return @{}
}
function Test-Path { return $Scenario -ne 'missing-sdk' }
function Write-Host { $testState.Messages += [string]$args[0] }
function Start-Process {
    param($FilePath, $ArgumentList, $Verb, $WindowStyle, [switch]$Wait, [switch]$PassThru)
    if ($FilePath -ne 'C:\test path\vJoySetup.exe' -or $Verb -ne 'RunAs' -or $WindowStyle -ne 'Hidden' -or -not $Wait -or -not $PassThru) {
        throw 'Incorrect installer invocation'
    }
    if ('/NORESTART' -notin $ArgumentList -or '/RESTARTEXITCODE=3010' -notin $ArgumentList -or '/COMPONENTS="apps,apps\vJoyConf"' -notin $ArgumentList) {
        throw 'Missing required installer arguments'
    }
    $testState.Started = $true
    if ($Scenario -eq 'cancelled') { throw 'UAC cancelled' }
    $code = switch ($Scenario) { 'reboot3010' { 3010 }; 'reboot8' { 8 }; 'failed' { 5 }; default { 0 } }
    return @{ ExitCode = $code }
}

$failure = $null
try {
    & (Join-Path $PSScriptRoot '..\install-vjoy.ps1') -InstallerPath 'C:\test path\vJoySetup.exe'
} catch {
    $failure = $_.Exception.Message
}

$expectedFailure = switch ($Scenario) {
    'failed' { 'failed \(code 5\)' }
    'missing-sdk' { 'driver or SDK is missing' }
    'cancelled' { 'Administrator approval is required.*UAC cancelled' }
    default { $null }
}
if ($expectedFailure) {
    if (-not $failure -or $failure -notmatch $expectedFailure) { throw "Unexpected failure: $failure" }
} elseif ($failure) { throw $failure }
if ($Scenario -eq 'existing') {
    if ($testState.Started) { throw 'Existing driver must not be reinstalled' }
} elseif (-not $testState.Started) { throw 'Expected the driver installer to run' }
if ($Scenario -like 'reboot*' -and ($testState.Messages -join ' ') -notmatch 'Restart Windows') {
    throw 'A required restart must be reported'
}
