# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

param(
  [Parameter(Mandatory=$false)]
  [string]$notebookSource = "Riallto\notebooks",
  [string]$scriptSource = "Riallto\scripts\utils"
)

# Get user info and set notebooks dir
$username = [System.Environment]::UserName
$desktopPath = [Environment]::GetFolderPath("Desktop")
$userStartMenuPath = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"

$notebookDest = "C:\Users\$username\AppData\Roaming\riallto_notebooks"
$scriptDest = "C:\Users\$username\AppData\Local\Riallto"
$shortcutDest = "$desktopPath\Launch Riallto.lnk"
$menuShortcutDest = "$userStartMenuPath\Launch Riallto.lnk"

# In case riallto_notebooks folder already exists, make sure to create a bakup
function Backup-Notebooks {
    if (Test-Path -Path $notebookDest) {
        Write-Output "riallto_notebooks folder already exists."

        $timestamp = (Get-Date).ToString("yyyyMMddHHmmss")
        $backupPath = "${notebookDest}_${timestamp}"

        try {        
            # Rename the original folder to the backup name
            Rename-Item -Path $notebookDest -NewName ([System.IO.Path]::GetFileName($backupPath))
            Write-Output "Existing notebooks folder backed up as: $backupPath"   
        } catch {
            Write-Output "Failed to rename $notebookDest, please delete or rename that folder."
            Exit 1
        }
    }
}

function Install-Notebooks {
    Write-Output "Delivering notebooks to $notebookDest"
    try {
        Copy-Item -Path $notebookSource -Destination $notebookDest -Recurse
    } catch {
        Write-Output "Failed to copy notebooks to $notebookDest"
        Exit 1
    }	
}

function Install-Shortcut {
    # Copy utility scripts
    if (-Not (Test-Path $scriptDest)) {
        $null = mkdir $scriptDest
    }

    if (Test-Path "$scriptDest\utils") {
        Remove-Item -Path "$scriptDest\utils" -Recurse -Force
    }

    Copy-Item -Path $scriptSource -Destination "$scriptDest\utils" -Recurse -Force

    $powershellLoc = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    $Arguments = "-ExecutionPolicy Bypass -NoProfile -File `"" + $scriptDest + "\utils\run_jupyter.ps1" + "`""

    # Create new shortcut object
    $WScriptShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WScriptShell.CreateShortcut($shortcutDest)

    $Shortcut.TargetPath = $powershellLoc
    $Shortcut.Arguments = $Arguments
    $Shortcut.IconLocation =  $notebookDest + "\images\ico\riallto.ico"

    # Save shortcut to desktop
    $Shortcut.Save()

    # Create new shortcut object for Start Menu
    $StartMenuShortcut = $WScriptShell.CreateShortcut($menuShortcutDest) 

    $StartMenuShortcut.TargetPath = $powershellLoc
    $StartMenuShortcut.Arguments = $Arguments
    $StartMenuShortcut.IconLocation = $notebookDest + "\images\ico\riallto.ico"

    # Save shortcut to Start Menu
    $StartMenuShortcut.Save()
}

Backup-Notebooks
Install-Notebooks
Install-Shortcut
