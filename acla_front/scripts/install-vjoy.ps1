param([Parameter(Mandatory = $true)][string]$InstallerPath)

$ErrorActionPreference = 'Stop'
$nativeArchitecture = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($nativeArchitecture -ne 'AMD64' -or [Environment]::OSVersion.Version.Major -lt 10) {
    throw 'The vJoy driver requires Windows 10 or newer on x64.'
}

$programFilesPath = if ($env:ProgramW6432) { $env:ProgramW6432 } else { $env:ProgramFiles }
$vjoyDirectory = Join-Path $programFilesPath 'vJoy'
$logPath = Join-Path $env:TEMP 'Kestrel-vJoy-setup.log'

function Test-VjoyInstalled {
    $service = Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Services\vjoy' -ErrorAction SilentlyContinue
    return ($service.ImagePath -and (Test-Path -LiteralPath (Join-Path $vjoyDirectory 'x64\vJoyInterface.dll')))
}

# Upstream setup removes an existing driver before reinstalling. Preserve working
# installations and device settings, just as the packaged app's installer does.
if (Test-VjoyInstalled) {
    Write-Host 'vJoy is already installed. Keeping the existing driver and device configuration.'
    return
}

Write-Host 'Installing vJoy. Approve the Windows administrator prompt to continue.'
$installerArguments = @(
    '/VERYSILENT', '/SUPPRESSMSGBOXES', '/SP-', '/NORESTART', '/RESTARTEXITCODE=3010',
    '/COMPONENTS="apps,apps\vJoyConf"', ('/DIR="{0}"' -f $vjoyDirectory), ('/LOG="{0}"' -f $logPath)
)
try {
    $installation = Start-Process -FilePath $InstallerPath -ArgumentList $installerArguments -Verb RunAs -WindowStyle Hidden -Wait -PassThru
} catch {
    throw "Unable to start vJoy setup. Administrator approval is required. $($_.Exception.Message)"
}

if ($installation.ExitCode -in @(3010, 8)) {
    Write-Host 'Restart Windows to finish installing vJoy before using the virtual wheel.'
    return
}
if ($installation.ExitCode -ne 0) {
    throw "vJoy installation failed (code $($installation.ExitCode)). See $logPath."
}
if (-not (Test-VjoyInstalled)) {
    throw "vJoy setup finished but the driver or SDK is missing. Restart Windows and rerun npm run setup:all. See $logPath."
}
Write-Host 'vJoy installed. Enable X, Y and Z axes on your device in Configure vJoy, then bind them in the simulator.'
