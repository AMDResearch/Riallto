$username = [System.Environment]::UserName
$desktopPath = [Environment]::GetFolderPath("Desktop")
$userStartMenuPath = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"

$dataPath = "C:\Users\$username\AppData\Local\Riallto"
$shortcutDest = "$desktopPath\Launch Riallto.lnk"
$menuShortcutDest = "$userStartMenuPath\Launch Riallto.lnk"

# we won't delete what's in the roaming folder
# $notebookDest = "C:\Users\$username\AppData\Roaming\riallto_notebooks"

if (py -3.9 -m pip --disable-pip-version-check list --format freeze | Where-Object { $_ -like "npu=*" }) {
    try {
        py -3.9 -m pip uninstall -y npu
		$pythonLibDir = (py -3.9 -m pip show pip | Select-String "Location:").Line.Split(" ")[1]
		if (Test-Path -Path $pythonLibDir\npu){
			try {
				Remove-Item -Recurse -Force $pythonLibDir\npu
			} catch {
				Write-Host "Couldn't delete previous npu site-packages folder: $_"
			}
		}
    } catch {
        Write-Host "Failed to uninstall npu: $_"
    }
}

# Delete shortcuts
if (Test-Path $menuShortcutDest -PathType Leaf) {
    try {
        Remove-Item -Recurse -Force $menuShortcutDest
        Write-Host "Deleted $menuShortcutDest"
    } catch {
        Write-Host "Couldn't delete $menuShortcutDest : $_"
    }
}

if (Test-Path $shortcutDest -PathType Leaf) {
    try {
        Remove-Item -Recurse -Force $shortcutDest
        Write-Host "Deleted $shortcutDest"
    } catch {
        Write-Host "Couldn't delete $shortcutDest : $_"
    }
}

# Delete Riallto data folder
if (Test-Path -Path $dataPath){
    try {
        Remove-Item -Recurse -Force $dataPath
        Write-Host "Deleted $dataPath"
    } catch {
        Write-Host "Couldn't delete $dataPath : $_"
    }
}