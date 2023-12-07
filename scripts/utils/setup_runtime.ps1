# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

param(
  [Parameter(Mandatory=$true)]
  [string]$package_dir = "."
)

# Check if -Verbose was used when calling script
$verbose = $PSBoundParameters.ContainsKey('Verbose')

# $username = [System.Environment]::UserName
# $venvPath = "C:\Users\$username\AppData\Local\Riallto\riallto_venv"

# Installs from this github repo setup.py
function Install-Runtime($package_dir) {
	
	Write-Output "Installing Riallto NPU Python package..."

	# Enter venv
	$venvPath = Join-Path -Path $PSScriptRoot -ChildPath "activate_venv.ps1"
	. $venvPath

	# Check if NPU runtime is installed
	if (py -m pip --disable-pip-version-check list --format freeze | Where-Object { $_ -like "npu=*" }) {
		Start-Process -FilePath py -ArgumentList "-m pip uninstall -y npu" -Wait -NoNewWindow -RedirectStandardOutput stdout_log.txt -RedirectStandardError stderr_log.txt
		$pythonLibDir = (py -m pip show pip | Select-String "Location:").Line.Split(" ")[1]
		if (Test-Path -Path $pythonLibDir\npu){
			try {
				Remove-Item -Recurse -Force $pythonLibDir\npu
			} catch {
				Write-Host "Couldn't delete previous npu site-packages folder: $_"
			}
		}
	}
	Push-Location $package_dir
	try {
		if ($verbose) {
			Start-Process -FilePath py -ArgumentList "-m pip install ." -Wait -NoNewWindow 
		} else {
			Start-Process -FilePath py -ArgumentList "-m pip install ." -Wait -NoNewWindow -RedirectStandardOutput stdout_log.txt -RedirectStandardError stderr_log.txt
		}
		Write-Output "Successfully installed Riallto runtime"
	} catch {
		Write-Output "Failed to install Riallto runtime"
	}
	Pop-Location

	# Add driver dir to system path so we can access xbutil
	try {
		$driverPath = "C:\Windows\System32\AMD"
		if (-not ($env:Path -split ';' -contains $driverPath)) {
			$env:Path += ";$driverPath"
			[System.Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)
		}
	} catch {
		Write-Host "Failed to add $driverPath to system path. $_"
		Exit 1
	}
}

Install-Runtime($package_dir)