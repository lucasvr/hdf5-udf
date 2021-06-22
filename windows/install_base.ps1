# Based on https://github.com/actions/virtual-environments/blob/main/images/win/scripts/Installers/Install-Msys2.ps1

$instdir = $args[0]
$pluginsdir = $args[1]
. .\install_helpers.ps1

Write-Host "=> instdir=$instdir"
Write-Host "=> pluginsdir=$pluginsdir"


################################
# Download and install XZ-Utils
################################

$url = "https://tukaani.org/xz/xz-5.2.5-windows.zip"
Write-Host "Downloading XZ-Utils $($url.split('/')[-1])"
$archive = Start-DownloadWithRetry -Url $url

$xz = "$instdir\xz-utils\bin_x86-64\xz.exe"
Write-Host "Testing if $xz exists"
$env:PATH += ";$instdir\xz-utils\bin_x86-64"
if (-Not (Test-Path -Path $xz)) {
    Write-Host "=> Creating $instdir/xz-utils"
    mkdir "$instdir\xz-utils"
    Expand-Archive -Path "$archive" -DestinationPath "$instdir\xz-utils"
    Remove-Item -Path $archive -Force
}

#################
# Download MSYS2
#################

$msys2_baseurl = "https://api.github.com/repos/msys2/msys2-installer/releases/latest"
$url = ((Invoke-RestMethod $msys2_baseurl).assets | Where-Object {
    $_.name -match "x86_64" -and $_.name.EndsWith(".tar.xz") }).browser_download_url

Write-Host "Downloading MSYS2 $($url.split('/')[-1])"
$archive = Start-DownloadWithRetry -Url $url
$tarfile = "$archive".replace('.tar.xz', '.tar')

if (-Not (Test-Path -Path "$instdir\msys64")) {
    Write-Host "Uncompressing MSYS2"
    & $xz -T 0 -v -d "$archive"
    tar -C "$instdir" -xf "$tarfile"
    Remove-Item -Path "$tarfile" -Force
}

# We no longer need xz-utils around
Remove-Item -Path "$instdir/xz-utils" -Recurse -Force

##############################
# Initialize Pacman and MSYS2
##############################

$env:PATH += ";$instdir\msys64\mingw64\bin;$instdir\msys64\usr\bin"

bash -c "pacman-key --init 2>&1"
bash -c "pacman-key --populate msys2 2>&1"
pacman --noconfirm -r "$instdir\msys64" -Syu
pacman --noconfirm -r "$instdir\msys64" -Syu # twice

######################################
# Basic HDF5-UDF runtime dependencies
######################################

pacman --noconfirm -r "$instdir\msys64" -S mingw-w64-x86_64-hdf5
pacman --noconfirm -r "$instdir\msys64" -S mingw-w64-x86_64-dlfcn
pacman --noconfirm -r "$instdir\msys64" -S mingw-w64-x86_64-pcre
pacman --noconfirm -r "$instdir\msys64" -S mingw-w64-x86_64-libsodium
