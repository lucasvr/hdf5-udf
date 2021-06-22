; MSIS installation script for HDF5-UDF
; Process this file with makensis or makensisw to create an installer program

!define MUI_PRODUCT "HDF5-UDF"
!define MUI_VERSION "2.1"
!define MUI_BRANDINGTEXT "User-Defined Functions for HDF5"
!define MUI_FINISHPAGE_NOAUTOCLOSE
!define ROOT ".."
CRCCheck on

; General options
Name "${MUI_PRODUCT} ${MUI_VERSION}"
OutFile "HDF5-UDF Installer.exe"
ShowInstDetails show
ShowUninstDetails show
SetDateSave on
Unicode true

; Target directory
InstallDir "$PROGRAMFILES\${MUI_PRODUCT}\${MUI_VERSION}"

; UI configuration
!include "MUI2.nsh"
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Language and activation of UI system
!insertmacro MUI_LANGUAGE "English"

; Actions we want to perform once
Section "-hidden section"
    InitPluginsDir
    SetOverwrite on
    SetOutPath $PLUGINSDIR\PowerShell
    File install_helpers.ps1
    File install_base.ps1
    File install_cpp.ps1
    File install_cpython.ps1
    File install_luajit.ps1
SectionEnd

; Core HDF5-UDF files. Note that we don't ship MSYS2 libraries that libhdf5-udf depends on,
; as they're useless without a proper Python/LuaJIT/C++ runtime environment. An accompanying
; MSYS2 environment is required for the proper usage of HDF5-UDF.
Section "HDF5-UDF" SectionCore
    SectionIn RO

    SetOverwrite on
    SetOutPath "$INSTDIR"
    File "${ROOT}\install\bin\hdf5-udf.exe"
    File "${ROOT}\install\bin\libhdf5-udf-0.dll"
    File "${ROOT}\install\hdf5\lib\plugin\libhdf5-udf-iofilter.dll"
    File "${ROOT}\LICENSE"

    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${MUI_PRODUCT}" "DisplayName" "${MUI_PRODUCT} (remove only)"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${MUI_PRODUCT}" "UninstallString" "$INSTDIR\uninstall.exe"
    WriteUninstaller "$INSTDIR\uninstall.exe"
SectionEnd

; Provide a method to allow the user to install MSYS2. All required packages are downloaded
; from the MSYS2 repository on-the-fly. MSYS2 is installed as a subdirectory of HDF5-UDF, so
; no conflicts with existing installations are expected to happen.
Section "MSYS2 runtime" SectionMsys2Runtime
    AddSize 500000 ; kb
    nsExec::ExecToLog 'powershell -InputFormat none -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\PowerShell\install_base.ps1" "$INSTDIR" "$PLUGINSDIR\PowerShell"'
    Pop $0
SectionEnd

; GCC/LuaJIT/Python runtime sections follow
Section "C/C++ runtime" SectionCppRuntime
    AddSize 490000 ; kb
    nsExec::ExecToLog 'powershell -InputFormat none -NoProfile -ExecutionPolicy Bypass -File $PLUGINSDIR\PowerShell\install_cpp.ps1 "$INSTDIR" "$PLUGINSDIR\PowerShell"'
    Pop $0
SectionEnd

Section "LuaJIT runtime" SectionLuaJITRuntime
    AddSize 6000 ; kb
    nsExec::ExecToLog 'powershell -InputFormat none -NoProfile -ExecutionPolicy Bypass -File $PLUGINSDIR\PowerShell\install_luajit.ps1 "$INSTDIR" "$PLUGINSDIR\PowerShell"'
    Pop $0
SectionEnd

Section "CPython runtime" SectionCPythonRuntime
    AddSize 420000 ; kb
    nsExec::ExecToLog 'powershell -InputFormat none -NoProfile -ExecutionPolicy Bypass -File $PLUGINSDIR\PowerShell\install_cpython.ps1 "$INSTDIR" "$PLUGINSDIR\PowerShell"'
    Pop $0
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${SectionCore} "HDF5-UDF core library and I/O filter (required)."
    !insertmacro MUI_DESCRIPTION_TEXT ${SectionMsys2Runtime} "MSYS2 runtime (required). May be disabled if you already have a MSYS2 environment and would like to use that instead."
    !insertmacro MUI_DESCRIPTION_TEXT ${SectionCppRuntime} "Support for writing and reading UDFs written in C/C++. This option installs the GCC compiler suite."
    !insertmacro MUI_DESCRIPTION_TEXT ${SectionLuaJITRuntime} "Support for writing and reading UDFs written in Lua. This option installs the LuaJIT engine."
    !insertmacro MUI_DESCRIPTION_TEXT ${SectionCPythonRuntime} "Support for writing and reading UDFs written in Python. This option installs the Python interpreter."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Function .onInstSuccess
    ; Add the installation directory to $PATH and $HDF5_PLUGIN_PATH.
    ; EnVar::Check returns 3 if the value doesn't exist in the given variable.
    EnVar::Check "PATH" "$INSTDIR\msys64\mingw64\bin"
    Pop $0
    IntCmp $0 3 addVariables
    addVariables:
        EnVar::AddValueEx "PATH" "$INSTDIR\msys64\mingw64\bin"
        EnVar::AddValueEx "HDF5_PLUGIN_PATH" "$INSTDIR"
        EnVar::AddValueEx "PATH" "$INSTDIR"
        goto done
    done:
        Delete "$PLUGINSDIR\PowerShell\install_helpers.ps1"
        Delete "$PLUGINSDIR\PowerShell\install_base.ps1"
        Delete "$PLUGINSDIR\PowerShell\install_cpp.ps1"
        Delete "$PLUGINSDIR\PowerShell\install_cpython.ps1"
        Delete "$PLUGINSDIR\PowerShell\install_luajit.ps1"
        RMDir "$PLUGINSDIR\PowerShell"
FunctionEnd

Section "Uninstall"
    RMDir /r "$INSTDIR\*.*"
    RMDir "$PROGRAMFILES\${MUI_PRODUCT}\${MUI_VERSION}"
    RMDir "$PROGRAMFILES\${MUI_PRODUCT}"

    DeleteRegKey HKLM "Software\${MUI_PRODUCT}"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${MUI_PRODUCT}"

    EnVar::DeleteValue "PATH" "$INSTDIR\msys64\mingw64\bin"
    EnVar::DeleteValue "HDF5_PLUGIN_PATH" "$INSTDIR"
    EnVar::DeleteValue "PATH" "$INSTDIR"
SectionEnd
