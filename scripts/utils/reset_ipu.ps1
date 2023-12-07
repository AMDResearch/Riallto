# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

$deviceName = "AMD IPU Device"
$device = Get-WmiObject Win32_PnPEntity | Where-Object { $_.Name -eq $deviceName }

# Disable IPU
$result = $device.Disable()
if ($result.ReturnValue -eq 0) {
    Write-Host "Disabled $deviceName"
} else {
    Write-Host "Failed to disable $deviceName"
    exit 1
}

# Need to wait a bit so it doesn't go into a bad state and then we have to restart...
Start-Sleep -Seconds 5

# Enable IPU
$result = $device.Enable()
if ($result.ReturnValue -eq 0) {
    Write-Host "Enabled $deviceName"
} else {
    Write-Host "Failed to enable $deviceName"
    exit 1
}