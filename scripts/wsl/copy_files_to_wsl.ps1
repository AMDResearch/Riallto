# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

param(
  [Parameter(Mandatory=$false)]
  [string]$repoPath = "Riallto"
)

# Replace slashes and spaces
$repoPathWSL = "/mnt/" + ($repoPath -replace ":", "").Replace("\", "/").Replace(" ", "\ ").ToLower()

# Replace special characters
$repoPathWSL = $repoPathWSL -replace '\(', '\(' -replace '\)', '\)'

$wslName = "Riallto"

$tarballs = "xilinx_tools_latest.tar.gz", "pynqMLIR-AIE_latest.tar.gz"

# Copy tarballs and necessary scripts into virtual WSL hard drive
# this is done because untar'ing in Windows filesystem will take days
function Copy-Files() {
	try {
		wsl -d $wslName -u root bash -ic "mkdir ~/build"
		wsl -d $wslName -u root bash -ic "rsync $repoPathWSL/downloads/*.tar.gz ~/build/"
		wsl -d $wslName -u root bash -ic "cp -r $repoPathWSL/scripts/wsl/requirements.txt ~/build/"
		wsl -d $wslName -u root bash -ic "cp $repoPathWSL/scripts/wsl/*.sh ~/build"
	} catch {
		Write-Output "Failed to copy files"
		Exit 1
	}	
}

# Tarballs are copied using rsync, but just in case sanity check md5sums
function Test-Hashes() {
    foreach ($tarball in $tarballs) {
        $md5 = wsl -d $wslName -u root bash -ic "md5sum ~/build/$tarball | cut -d ' ' -f1"
        
        try {
            $md5file = "$repoPath/downloads/"+$tarball.replace("tar.gz", "md5")
            $hash = Get-Content ($md5file)
        } catch {
            Write-Output "Failed to get md5sum from $md5file"
        }
        
        if (-Not ($hash.Equals($md5))){
            Write-Output "Invalid checksum: $tarball"
            Write-Output "Got: $md5, Expected: $hash"
            Exit 1
        }
    }
}

Write-Host "Copying scripts and tarballs to WSL..."
Copy-Files

Write-Host "Validating tarball checksums..."
Test-Hashes