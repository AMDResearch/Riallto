# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Our custom wsl instance name and install directory
$wslName = "Riallto"

$xilinx_tarball = "xilinx_tools_latest.tar.gz"
$mlir_tarball =  "pynqMLIR-AIE_latest.tar.gz"

function Initialize-Environment {
	if ($verbose) {
		$null = wsl -d $wslName -u root bash -ic "cd ~/build && ./setup_env.sh $xilinx_tarball $mlir_tarball 2>&1 | tee wsl.log; exit `${PIPESTATUS[0]}"
	} else {
		$null = wsl -d $wslName -u root bash -ic "cd ~/build && ./setup_env.sh $xilinx_tarball $mlir_tarball >> wsl.log 2>&1; exit `${PIPESTATUS[0]}"
	}

	if ($LASTEXITCODE -ne 0) { 
		Write-Host 'Failed to setup environment. Please check the log in /root/build/wsl.log'
		Exit 1
	}
}

# Extract tarballs, move tools to /opt and setup paths
Write-Output "Extracting tools and setting up the environment..."
Initialize-Environment