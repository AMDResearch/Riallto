# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

param (
    [string]$zipPath
)

& uninstall_onnx.ps1

# Enter venv
$venvPath = Join-Path -Path $PSScriptRoot -ChildPath "activate_venv.ps1"
. $venvPath

# Unzip and cd into the RyzenAI-SW package
$zipFolderName = [System.IO.Path]::GetFileNameWithoutExtension($zipPath)
if (-Not (Test-Path $zipFolderName)) {
    Expand-Archive -Path $zipPath -DestinationPath .
}
cd ".\$zipFolderName"

# Install the required Python packages
try {
    py -m pip install .\voe-4.0-win_amd64\onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl
    py -m pip install .\voe-4.0-win_amd64\voe-0.1.0-cp39-cp39-win_amd64.whl
    py -m pip install .\vai_q_onnx-1.16.0+69bc4f2-py2.py3-none-any.whl
} catch {
    Write-Output "Failed to install RyzenAI-SW wheels"
    Exit 1
}

# Our python env site-packages directory
$siteDir = (py -m pip show pip | Select-String "Location:").Line.Split(" ")[1]

# Copy the DLL files required by ONNX Runtime capi
$dllFiles = @(
    "C:\Windows\System32\AMD\xrt_core.dll",
    "C:\Windows\System32\AMD\xrt_coreutil.dll",
    "C:\Windows\System32\AMD\amd_xrt_core.dll",
    "C:\Windows\System32\AMD\xdp_ml_timeline_plugin.dll",
    "C:\Windows\System32\AMD\xdp_core.dll"
)

try {
    foreach ($dll in $dllFiles) {
        Copy-Item -Path $dll -Destination "$siteDir\onnxruntime\capi\"
    }
} catch {
    Write-Output "Failed to copy dlls to $siteDir\onnxruntime\capi"
}


# Copy xclbin and config to Riallto notebooks folder
$destinationDir = "C:\users\$Env:UserName\AppData\Roaming\riallto_notebooks\onnx\xclbins"
try {
    Copy-Item -Path .\voe-4.0-win_amd64\vaip_config.json -Destination $destinationDir
    Copy-Item -Path .\voe-4.0-win_amd64\1x4.xclbin -Destination $destinationDir
} catch {
    Write-Output "Failed to copy files to $destinationDir"
}
