param(
    [Parameter(Mandatory = $true)]
    [string]$InstallerPath
)

$ErrorActionPreference = 'Stop'

function Get-ReadyControllerBus {
    Get-CimInstance -ClassName Win32_PnPEntity -Filter "Service = 'ViGEmBus'" |
        Where-Object { $_.ConfigManagerErrorCode -eq 0 }
}

try {
    # Keep an existing working driver, including one installed by another app.
    if (Get-ReadyControllerBus) {
        Write-Output 'The virtual Xbox controller driver is already installed.'
        exit 0
    }

    if (-not (Test-Path -LiteralPath $InstallerPath -PathType Leaf)) {
        throw 'The bundled ViGEmBus installer is missing. Download and run the application installer again.'
    }

    # Elevate only the driver setup; the app can remain a per-user installation.
    $driverSetup = Start-Process -FilePath $InstallerPath -ArgumentList '/exenoui', '/qn', '/norestart' `
        -Verb RunAs -WindowStyle Hidden -Wait -PassThru
    $driverExitCode = $driverSetup.ExitCode
    if ($driverExitCode -in @(3010, 1641)) {
        exit 3010
    }
    if ($driverExitCode -ne 0) {
        Write-Output "ViGEmBus setup failed with exit code $driverExitCode."
        exit $driverExitCode
    }
    if (-not (Get-ReadyControllerBus)) {
        throw 'ViGEmBus setup completed but the controller driver is not ready. Restart Windows and rerun application setup.'
    }
    Write-Output 'The virtual Xbox controller driver is ready.'
    exit 0
} catch {
    Write-Output $_.Exception.Message
    # Win32 error 1223 indicates the user declined Windows administrator approval.
    if ($_.Exception.NativeErrorCode -eq 1223) { exit 1223 }
    exit 1
}
