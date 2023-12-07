# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

$pythonLibDir = (py -m pip show pip | Select-String "Location:").Line.Split(" ")[1]

Write-Host "Checking for existing ONNX Runtime installations"

# Enter venv
$venvPath = Join-Path -Path $PSScriptRoot -ChildPath "activate_venv.ps1"
. $venvPath

py -m pip uninstall -y onnxruntime_vitisai
if (Test-Path -Path $pythonLibDir\onnxruntime){
    try {
        Remove-Item -Recurse -Force $pythonLibDir\onnxruntime
    } catch {
        Write-Host "Couldn't delete onnxruntime site-packages folder: $_"
    }
}

py -m pip uninstall -y voe
if (Test-Path -Path $pythonLibDir\voe){
    try {
        Remove-Item -Recurse -Force $pythonLibDir\voe
    } catch {
        Write-Host "Couldn't delete voe site-packages folder: $_"
    }
}

py -m pip uninstall -y vai_q_onnx
if (Test-Path -Path $pythonLibDir\vai_q_onnx){
    try {
        Remove-Item -Recurse -Force $pythonLibDir\vai_q_onnx
    } catch {
        Write-Host "Couldn't delete vai_q_onnx site-packages folder: $_"
    }
}