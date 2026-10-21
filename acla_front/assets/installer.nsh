!macro customInstall
  DetailPrint "Installing the virtual Xbox controller required by AI Self-Driving Training..."
  InitPluginsDir
  File /oname=$PLUGINSDIR\ViGEmBus.exe "${PROJECT_DIR}\.venv\controller-driver\ViGEmBus_1.22.0_x64_x86_arm64.exe"
  File /oname=$PLUGINSDIR\install-controller-driver.ps1 "${BUILD_RESOURCES_DIR}\install-controller-driver.ps1"
  nsExec::ExecToLog '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$PLUGINSDIR\install-controller-driver.ps1" -InstallerPath "$PLUGINSDIR\ViGEmBus.exe"'
  Pop $0
  ${If} $0 == 3010
    SetRebootFlag true
    MessageBox MB_OK|MB_ICONINFORMATION "Restart Windows before using AI Self-Driving Training to finish installing the virtual Xbox controller." /SD IDOK
    ; Preserve the reboot result for silent deployment and skip automatic app launch.
    SetErrorLevel 3010
    Quit
  ${ElseIf} $0 != 0
    MessageBox MB_OK|MB_ICONSTOP "The virtual Xbox controller driver could not be installed (code $0). Rerun application setup and approve the Windows administrator prompt. If the problem persists, restart Windows and try again." /SD IDOK
    SetErrorLevel 1
    Abort
  ${EndIf}
!macroend
