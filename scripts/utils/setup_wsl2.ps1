# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

function Install-WSL() {
	$wsl_enabled = (Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux).State -eq 'Enabled'
	
	# Additional check, in case script was re-run without a reboot
	try {
		wsl --set-default-version 2 > $null 2>&1
		$wsl_pass = $true
	} catch {
		$wsl_pass = $false
	}

	$wsl_status = ($wsl_enabled -and $wsl_pass)

	# If WSL not installed
	if (-not $wsl_status) {
		
		try {
			Write-Output "WSL is not installed. Beginning the installation..."
			
			# Enable WSL
			Write-Host "Enabling Microsoft-Windows-Subsystem-Linux Windows feature"
			Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart
			
			# Enable Virtual Machine Platform
			Write-Host "Enabling VirtualMachinePlatform Windows feature"
			Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart
			
			# Download and install WSL2 kernel package update
			try {
				Write-Host "Downloading wsl_update_x64.msi kernel package update"
				Invoke-WebRequest -Uri https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi -OutFile ./wsl_update_x64.msi
			} catch {
				Write-Host "Failed to download kernel update from https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi"
				Exit 1
			}
			
			try {
				Start-Process msiexec.exe -Wait -ArgumentList '/I ./wsl_update_x64.msi /quiet /norestart'
				wsl.exe --update
			} catch {
				Write-Host "Failed to install WSL2 kernel"
				Exit 1
			}
			
			# For WSL features to be properly activated we need to reboot
			Write-Host "WSL installed successfully. Please reboot your system, then run this script again"
			$reboot = Read-Host "Would you like to reboot now? (yes/no)"
			if ($reboot -eq "yes" -or $reboot -eq "") {
				Write-Output "Restarting now..."
				Restart-Computer -Force
			} else {
				Write-Output "Please don't forget to restart your computer before running this script again."
			}

		} catch {
			Write-Host "WSL failed to install" -ForegroundColor red -BackgroundColor black
			Exit 1
		}
		
	} else {
		Write-Output "WSL already installed"
	}
}

Install-WSL