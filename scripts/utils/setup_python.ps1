# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

param(
  [Parameter(Mandatory=$false)]
  [version]$version = "3.9.2"
)

# Check if -Verbose was used when calling script
$verbose = $PSBoundParameters.ContainsKey('Verbose')

# Check if required Python version is installed
function Check-PythonInstalled {
	
	# Depending on installation python can be launched with py.exe or python.exe
	$commands = 'py', 'python'
	
	Write-Host "Detecting Python installation..."
	
	foreach ($command in $commands) {
		try {
			$python_version = (& $command --version 2>null) -replace "Python ", ""
			
			# Because of windows app execution aliases, if python isn't installed the process
			# can still run without error because it tries to launch the microsoft store, in which case
			# e.g. python.exe runs successfully but python_version will be an empty string
			if ($python_version){
				# For pybind11 compiled pyd Python has to be 3.9 major version, patch version doesn't matter
				if ($python_version -like "3.9.*") {
					Write-Host "Python $python_version is installed"
					return $true
				} else {
					Write-Host "Found Python $python_version, expected Python 3.9.*" -ForegroundColor red -BackgroundColor black
					return $false
				}
			} else {
				Write-Host "Could not get a version number, Python may not be installed"
			}
		} catch {
			Write-Host "$command.exe not found"
		}
	}
	
	#Write-Host "Could not find a Python 3.9.* installation, please install Python 3.9.* and run this script again" -ForegroundColor red -BackgroundColor black
	return $false
}

# This function downloads the executable and installs Python3.9.12-amd64
function Install-Python($version) {

	if (-Not (Check-PythonInstalled)) {
		Write-Host "Installing Python $version"

		# Download Python installer executable
		if (-Not (Test-Path "python-$version-amd64.exe" -PathType Leaf)) {
			try {
				Invoke-WebRequest -Uri "https://www.python.org/ftp/python/$version/python-$version-amd64.exe" -OutFile "$env:TEMP/python-$version-amd64.exe"
			} catch {
				Write-Host "Failed to download python-$version-amd64.exe, try again later or install python manually" -ForegroundColor red -BackgroundColor black
			}
		}
		
		# Install Python without UI
		try {
			Start-Process -FilePath "$env:TEMP\python-$version-amd64.exe" -ArgumentList "/passive InstallAllUsers=1 PrependPath=1 Include_launcher=1" -Wait -NoNewWindow
		} catch {
			Write-Host "Failed to install Python, make sure you are running this as administrator or install Python manually" -ForegroundColor red -BackgroundColor black
		}

	}
}

function New-VirtualEnv {
	$username = [System.Environment]::UserName
	$venvPath = "C:\Users\$username\AppData\Local\Riallto\riallto_venv"

	if (-Not (Test-Path $venvPath)) {
		try {
			Write-Host "Creating virtual environment: riallto_venv"
			py -3.9 -m venv Riallto $venvPath
		} catch {
			Write-Host "Failed to create the Riallto virtualenv"
			Exit 1
		}
	}

	# Add Riallto utils folder to path if not already on there
	# more convenient to run activate_venv.ps1
	try {
		$utilsPath = "C:\Users\$username\AppData\Local\Riallto\utils"
		if (-not ($env:Path -split ';' -contains $utilsPath)) {
			$env:Path += ";$utilsPath"
			[System.Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)
		}
	} catch {
		Write-Host "Failed to add $utilsPath to system path."
	}

	try {
		Write-Host "Executing activation script $venvPath\Scripts\activate"
		. "$venvPath\Scripts\activate.ps1"
	} catch {
		Write-Host "Failed to enter riallto_venv virtual environment"
		Exit 1
	}

	Write-Host "Upgrading pip"
	try {
		Start-Process -FilePath py -ArgumentList "-m pip install --upgrade pip" -Wait -NoNewWindow
	} catch {
		Write-Host "Failed to upgrade pip."
		Write-Host "Try to manually run py -m pip install --upgrade pip"
		Exit 1
	}

}

Install-Python($version)
New-VirtualEnv