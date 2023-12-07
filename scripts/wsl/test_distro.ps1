# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

$wslName = "Riallto"

## Exit if distro already installed
function Test-Distro() {
	$wslInfo = wsl -l -v
	if (($null -eq ($wslInfo -replace "`0" | Select-String -Pattern $wslName))){
		Write-Output "Installing $wslName WSL instance..."
	} 
	else {
		Write-Output "Unregistering existing $wslName..."
		try {
			$null = wsl --unregister $wslName
		} catch {
			Write-Output "Failed to unregister existing WSL instance $wslName"
			Exit 1
		}
	}
}

Test-Distro