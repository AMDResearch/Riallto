# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

$wslName = "Riallto"

function Install-PythonDependencies {
	if ($verbose) {
		wsl -d $wslName -u root bash -ic "python3 -m pip install -r ~/build/requirements.txt 2>&1 | tee wsl.log; exit `${PIPESTATUS[0]}"
	} else {
		wsl -d $wslName -u root bash -ic "python3 -m pip install -r ~/build/requirements.txt >> wsl.log 2>&1; exit `${PIPESTATUS[0]}"
	}

	if ($LASTEXITCODE -ne 0) { 
		Write-Host 'Failed to install Python packages. Please check the log in /root/build/wsl.log'
		Exit 1
	}
}

Write-Output "Installing Python3 packages..."
Install-PythonDependencies