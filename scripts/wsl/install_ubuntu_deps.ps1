# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

$wsl_name = "Riallto"

function Install-UbuntuDependencies {
	if ($verbose) {
		wsl -d $wsl_name -u root bash -ic "cd ~/build && ./ubuntu_deps.sh 2>&1 | tee wsl.log; exit `${PIPESTATUS[0]}"
	} else {
		wsl -d $wsl_name -u root bash -ic "cd ~/build && ./ubuntu_deps.sh >> wsl.log 2>&1; exit `${PIPESTATUS[0]}"
	}

	if ($LASTEXITCODE -ne 0) { 
		Write-Host 'Failed to install Ubuntu packages. Please check the log in /root/build/wsl.log'
		Exit 1
	}
}

Write-Output "Installing Ubuntu packages..."
Install-UbuntuDependencies