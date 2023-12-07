# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

$wslName = "Riallto"

try {
    Write-Host "Setting $wslName as default WSL instance"
    $null = wsl --setdefault $wslName
    # Start-Process wsl -ArgumentList "--setdefault $wslName" -Wait -WindowStyle Hidden
    Exit 0
} catch {
    Write-Host "Failed to set $wslName as default distro. Error: $_"
    Exit 1
}