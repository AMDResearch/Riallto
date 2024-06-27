# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Downloads a base WSL Ubuntu 20.04 image and imports it into WSL.
# Reason this is used instead of --install Ubuntu-20.04 is this allows
# setting a custom instance name.

param(
  [Parameter(Mandatory=$false)] 
  [string]$downloadPath = $env:TEMP
)

# Get user info 
$username = [System.Environment]::UserName
$installPath = "C:\Users\$username\AppData\Local\AMD\Riallto"

$wslName = "Riallto"

function New-Ubuntu {
	
	if (Test-Path -Path $installPath\ext4.vhdx) {
		Write-Output "Install directory $installPath already in use"
		Write-Output "Try running 'wsl --unregister $wslName' then run this script again"
		Exit 1
	}

	if ( (Test-Path -Path "Riallto\downloads\ubuntuLTS.appx" -PathType Leaf) -Or (Test-Path -Path "Riallto\downloads\ubuntuLTS.zip" -PathType Leaf) ) {
		$downloadPath = "Riallto\downloads"
	}
	
	if (-Not (Test-Path -Path $downloadPath)) {mkdir $downloadPath }
	if (-Not (Test-Path -Path $installPath)) {mkdir $installPath}
	
	# only re-download if not already downloaded & extracted
	if (-Not ((Test-Path -Path $downloadPath\ubuntuLTS.appx) -Or (Test-Path -Path $downloadPath\ubuntuLTS.zip))) {
		try {
			curl.exe -L -o $downloadPath\ubuntuLTS.appx https://aka.ms/wslubuntu2004
		} catch {
			Write-Output "Failed to download the base Ubuntu WSL image from https://aka.ms/wslubuntu2004: $_"
			Write-Output "Please check your connection."
			Exit 1
		}
	}

	# Confirm download succeeded
	Write-Output "Checking if $downloadPath\ubuntuLTS.appx successfully downloaded"
	if (-Not ((Test-Path -Path $downloadPath\ubuntuLTS.appx))) {
		Write-Output "Failed to download the base Ubuntu WSL image"
		Write-Output "Please make sure you have access to https://aka.ms/wslubuntu2004"
		Exit 1
	}

	try {
		Write-Output "Creating new Ubuntu WSL instance"
		if (-Not (Test-Path -Path $downloadPath\ubuntuLTS.zip)){
			Move-Item $downloadPath\ubuntuLTS.appx $downloadPath\ubuntuLTS.zip -Force
		}

		try {
			Write-Host "Extracting $downloadPath\ubuntuLTS.zip"
			Expand-Archive $downloadPath\ubuntuLTS.zip $downloadPath\ubuntuLTS -Force
		} catch {
			Write-Host "Failed to extract $downloadPath\ubuntuLTS.zip, retrying download."
			try {
				curl.exe -L -o $downloadPath\ubuntuLTS.appx https://aka.ms/wslubuntu2004
			} catch {
				Write-Output "Failed to download the base Ubuntu WSL image from https://aka.ms/wslubuntu2004: $_"
				Write-Output "Please check your connection."
				Exit 1
			}
			Write-Host "Moving $downloadPath\ubuntuLTS.appx to $downloadPath\ubuntuLTS.zip"
			Move-Item $downloadPath\ubuntuLTS.appx $downloadPath\ubuntuLTS.zip -Force

			Write-Host "Extracting $downloadPath\ubuntuLTS.zip"
			Expand-Archive $downloadPath\ubuntuLTS.zip $downloadPath\ubuntuLTS -Force
		}
	
		Move-Item $downloadPath\ubuntuLTS\Ubuntu_2004.2021.825.0_x64.appx $downloadPath\ubuntuLTS\Ubuntu_2004.2021.825.0_x64.zip -Force
		Expand-Archive $downloadPath\ubuntuLTS\Ubuntu_2004.2021.825.0_x64.zip $downloadPath\ubuntuLTS\Ubuntu_2004.2021.825.0_x64 -Force
	} catch {
		Write-Output "Failed to extract the Ubuntu WSL image: $_"
		Exit 1
	}
	
	# WSL2 is much more performant than 1, should always use by default
	try {
		$null = wsl --set-default-version 2
		# Start-Process wsl -ArgumentList "--set-default-version 2" -Wait -WindowStyle Hidden
		Write-Output "Using WSL version 2"
	} catch {
		Write-Output "Warning: WSL version 1 will be significantly slower than WSL2"
	}
	
	try {
        Write-Output "Registering new WSL instance $wslName, saving disk to $installPath"
		$null = wsl --import $wslName $installPath $downloadPath\ubuntuLTS\Ubuntu_2004.2021.825.0_x64\install.tar.gz
		# Start-Process wsl -ArgumentList "--import $wslName $installPath $downloadPath\ubuntuLTS\Ubuntu_2004.2021.825.0_x64\install.tar.gz" -Wait -WindowStyle Hidden
	} catch {
		Write-Output "Failed to create $wslName distro: $_"
		Exit 1
	}

	$wslInfo = wsl -l -v
	if (($null -eq ($wslInfo -replace "`0" | Select-String -Pattern $wslName))){
		Write-Output "Could not find $wslName in wsl distro list - WSL setup was not successful."
		Exit 1
	} else {
		Write-Output "Confirmed WSL $wslName instance successfully setup."
	}
}

function Update-Kernel {
	try {
		Write-Output "Updating WSL kernel"
		$null = wsl --update
	} catch {
		Write-Output "Failed to update WSL kernel: $_"
		Exit 1
	}
}

Update-Kernel
New-Ubuntu