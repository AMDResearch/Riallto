# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

param(
  [Parameter(Mandatory=$true)]
  [string]$licensePath = "Xilinx.lic"
)

$absPath = (Resolve-Path $licensePath).Path
$wslPath = "/mnt/" + ($absPath -replace ":", "").Replace("\", "/").Replace(" ", "\ ").ToLower()

$wslName = "Riallto"

$license_txt = Get-Content $absPath -Raw
$license_txt -match 'HOSTID=(?<content>.*);'

try {
    # Extract MAC as string aabbccddeeff
    $mac = $matches['content']
    # Convert to aa:bb:cc:dd:ee:ff format
    $mac = $mac -split '(..)' -ne '' -join ':'

    Write-Output "Found MAC: $mac"
} catch {
    Write-Output "Couldn't extract MAC, is $license a valid license file?"
    Exit 1
}


Write-Output "Copying $absPath to /opt/Xilinx.lic"
wsl -d $wslName -u root bash -ic "cp $wslPath /opt/Xilinx.lic"
wsl -d $wslName -u root bash -ic "echo export XILINXD_LICENSE_FILE=/opt/Xilinx.lic" `>`> /opt/mlir_settings.sh

Write-Output "Setting up a vmnic network interface"
wsl -d $wslName -u root bash -ic "echo 'ip link add vmnic0 type dummy || true'" `>`> /opt/mlir_settings.sh
wsl -d $wslName -u root bash -ic "echo 'ip link set vmnic0 addr $mac || true'" `>`> /opt/mlir_settings.sh