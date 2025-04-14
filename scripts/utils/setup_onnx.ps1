# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

& uninstall_onnx.ps1

# Enter venv
$venvPath = Join-Path -Path $PSScriptRoot -ChildPath "activate_venv.ps1"
. $venvPath

$raiInstallPath = "C:\Program Files\RyzenAI\1.2.0"

try {
    cd $raiInstallPath
} catch {
    Write-Output "Couldn't change dir to $raiInstallPath, have you installed RyzenAI-SW 1.2?"
}


# Install the required Python packages
try {
    py -m pip install .\voe-4.0-win_amd64\onnxruntime_vitisai-1.17.0-cp310-cp310-win_amd64.whl
    py -m pip install .\voe-4.0-win_amd64\voe-1.2.0-cp310-cp310-win_amd64.whl
    py -m pip install .\vai_q_onnx-1.2.0-py2.py3-none-win_amd64.whl
} catch {
    Write-Output "Failed to install RyzenAI-SW wheels"
    Exit 1
}

# Our python env site-packages directory
$siteDir = (py -m pip show pip | Select-String "Location:").Line.Split(" ")[1]

# Copy the DLL files required by ONNX Runtime capi
$dllFiles = @(
    "C:\Program Files\RyzenAI\1.2.0\onnxruntime\bin\onnxruntime.dll"
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

if (-Not (Test-Path $destinationDir)) {
        mkdir $destinationDir
}

try {
    Write-Output "Copying xclbin files to $destinationDir"

    Copy-Item -Path .\voe-4.0-win_amd64\vaip_config.json -Destination $destinationDir
    Copy-Item -Path .\voe-4.0-win_amd64\xclbins\phoenix\1x4.xclbin -Destination $destinationDir
} catch {
    Write-Output "Failed to copy files to $destinationDir"
}
