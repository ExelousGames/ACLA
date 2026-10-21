!include "LogicLib.nsh"
!include "x64.nsh"
!include "WinVer.nsh"
!define MUI_FINISHPAGE_REBOOTLATER_DEFAULT

!macro customInit
  ${IfNot} ${IsNativeAMD64}
    MessageBox MB_OK|MB_ICONSTOP "Kestrel and its bundled vJoy driver require Windows x64." /SD IDOK
    SetErrorLevel 1633
    Quit
  ${EndIf}
  ${IfNot} ${AtLeastWin10}
    MessageBox MB_OK|MB_ICONSTOP "The bundled vJoy driver requires Windows 10 or newer." /SD IDOK
    SetErrorLevel 1633
    Quit
  ${EndIf}
!macroend

!macro customInstall
  ; Match the SDK location used by the Python wheel bridge. Keep existing devices
  ; and newer drivers: upstream setup uninstalls an existing vJoy before reinstalling.
  SetRegView 64
  ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Services\vjoy" "ImagePath"
  ${If} $0 != ""
  ${AndIf} ${FileExists} "$PROGRAMFILES64\vJoy\x64\vJoyInterface.dll"
    DetailPrint "vJoy is already installed. Keeping the existing driver and device configuration."
  ${Else}
    DetailPrint "Installing the vJoy steering wheel driver..."
    ClearErrors
    ExecWait '"$INSTDIR\resources\vjoy\vJoySetup.exe" /VERYSILENT /SUPPRESSMSGBOXES /SP- /NORESTART /RESTARTEXITCODE=3010 /COMPONENTS="apps,apps\vJoyConf" /DIR="$PROGRAMFILES64\vJoy" /LOG="$TEMP\Kestrel-vJoy-setup.log"' $0
    ${If} ${Errors}
      StrCpy $0 1
    ${EndIf}
    ; vJoy also uses code 8 for a delayed driver installation requiring restart.
    ${If} $0 == 3010
    ${OrIf} $0 == 8
      SetRebootFlag true
      SetErrorLevel 3010
      DetailPrint "Restart Windows to finish installing vJoy."
    ${ElseIf} $0 != 0
      MessageBox MB_OK|MB_ICONSTOP "vJoy installation failed (code $0). See $TEMP\Kestrel-vJoy-setup.log, then run Kestrel setup again." /SD IDOK
      SetErrorLevel $0
      Abort
    ${Else}
      ; Do not report success if setup exits without installing the driver/SDK.
      ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Services\vjoy" "ImagePath"
      ${If} $0 == ""
      ${OrIfNot} ${FileExists} "$PROGRAMFILES64\vJoy\x64\vJoyInterface.dll"
        MessageBox MB_OK|MB_ICONSTOP "vJoy setup finished but the driver or SDK is missing. Restart Windows and run Kestrel setup again. See $TEMP\Kestrel-vJoy-setup.log." /SD IDOK
        SetErrorLevel 1
        Abort
      ${EndIf}
    ${EndIf}
  ${EndIf}
!macroend

; vJoy is a shared system driver. App updates/uninstallation must not remove it.
