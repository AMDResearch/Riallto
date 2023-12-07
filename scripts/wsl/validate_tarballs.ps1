# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

param(
  [Parameter(Mandatory=$false)]
  [string]$tarballsPath = "Riallto\downloads"
)

# Tarball variables
$xilinx_tarball = "xilinx_tools_latest.tar.gz"
$mlir_tarball = "pynqMLIR-AIE_latest.tar.gz"
$xilinx_md5 = "xilinx_tools_latest.md5"
$mlir_md5 = "pynqMLIR-AIE_latest.md5"

$file_list = $xilinx_tarball, $mlir_tarball, $xilinx_md5, $mlir_md5

function Test-FilesExist {
    $missed_files = @()

    foreach ($file in $file_list) {
        if (-Not (Test-Path -Path $tarballsPath\$file)) {
            $missed_files += $file
        }
    }

    if ($missed_files.Count -gt 0) {
        foreach ($missed_file in $missed_files) {
            Write-Output "Missing file: $tarballsPath\$missed_file"
        }
        Exit 1
    }
}

Write-Host "Checking if tarballs and md5 sums exist..."
Test-FilesExist