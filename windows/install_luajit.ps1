# Based on https://github.com/actions/virtual-environments/blob/main/images/win/scripts/Installers/Install-Msys2.ps1

$instdir = $args[0]
$pluginsdir = $args[1]
. $pluginsdir\install_helpers.ps1

$env:PATH += ";$instdir\msys64\mingw64\bin;$instdir\msys64\usr\bin"

#############################
# Install LuaJIT interpreter
#############################

pacman --noconfirm -r "$instdir/msys64" -S mingw-w64-x86_64-luajit
