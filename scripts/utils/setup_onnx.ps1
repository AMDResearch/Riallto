# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Get user info
$username = [System.Environment]::UserName
$notebookDest = "C:\Users\$username\AppData\Roaming\riallto_notebooks"
$wheelDest = "$notebookDest\onnx\wheels"

# Enter venv
$venvPath = Join-Path -Path $PSScriptRoot -ChildPath "activate_venv.ps1"
. $venvPath
function Get-Wheels() {

    if (-Not (Test-Path $wheelDest)) {
        $null = mkdir $wheelDest
    }

    Write-Host "Downloading ONNX Runtime wheels"
    try {
        Invoke-WebRequest -Uri "https://github.com/amd/RyzenAI-SW/raw/c9d3db1418c0f7ae15a617fa0b79f12d8dbf6e24/demo/cloud-to-client/wheels/onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl" -OutFile "$wheelDest\onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl"
        Invoke-WebRequest -Uri "https://github.com/amd/RyzenAI-SW/raw/c9d3db1418c0f7ae15a617fa0b79f12d8dbf6e24/demo/cloud-to-client/wheels/voe-0.1.0-cp39-cp39-win_amd64.whl" -OutFile "$wheelDest\voe-0.1.0-cp39-cp39-win_amd64.whl"
        Invoke-WebRequest -Uri "https://github.com/amd/RyzenAI-SW/raw/c9d3db1418c0f7ae15a617fa0b79f12d8dbf6e24/tutorial/RyzenAI_quant_tutorial/onnx_example/pkgs/vai_q_onnx-1.16.0+60e82ab-py2.py3-none-any.whl" -OutFile "$wheelDest\vai_q_onnx-1.16.0+60e82ab-py2.py3-none-any.whl"
    } catch {
        Write-Host "Failed to download ONNX Runtime wheels: $_"
    }
}

function Install-ONNXRuntime() {
    Write-Host "Installing ONNX Runtime package with Vitis AI EP"
    try {
        if ($verbose) {
            Start-Process -FilePath py -ArgumentList "-m pip install $wheelDest\onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl" -Wait -NoNewWindow 
            Start-Process -FilePath py -ArgumentList "-m pip install $wheelDest\voe-0.1.0-cp39-cp39-win_amd64.whl" -Wait -NoNewWindow 
            Start-Process -FilePath py -ArgumentList "-m pip install $wheelDest\vai_q_onnx-1.16.0+60e82ab-py2.py3-none-any.whl" -Wait -NoNewWindow 
        } else {
            Start-Process -FilePath py -ArgumentList "-m pip install $wheelDest\onnxruntime_vitisai-1.15.1-cp39-cp39-win_amd64.whl" -Wait -NoNewWindow -RedirectStandardOutput onnx_setup_stdout_log.txt -RedirectStandardError onnx_setup_stderr_log.txt
            Start-Process -FilePath py -ArgumentList "-m pip install $wheelDest\voe-0.1.0-cp39-cp39-win_amd64.whl" -Wait -NoNewWindow -RedirectStandardOutput onnx_setup_stdout_log.txt -RedirectStandardError onnx_setup_stderr_log.txt
            Start-Process -FilePath py -ArgumentList "-m pip install $wheelDest\vai_q_onnx-1.16.0+60e82ab-py2.py3-none-any.whl" -Wait -NoNewWindow -RedirectStandardOutput onnx_setup_stdout_log.txt -RedirectStandardError onnx_setup_stderr_log.txt
        }
        Write-Host "Successfully installed ONNX Runtime"
    } catch {
        Write-Host "Failed to install ONNX Runtime" -ForegroundColor red -BackgroundColor black
        Return 1
    }

    # Final check if voe is installed:
    $checkInstall = (py -m pip show voe | Select-String 0.1.0) -match "0.1.0"
    if ($checkInstall) {
        Write-Host "VOE confirmed"
    } else {
        Write-Host "Retrying VOE installation"
        $checkInstall = (py -m pip show voe | Select-String 0.1.0) -match "0.1.0"
        if (-Not ($checkInstall)) {
            Write-Host "Failed VOE installation"
            Write-Host "Please make sure wheel exists in $wheelDest\voe-0.1.0-cp39-cp39-win_amd64.whl and try to install again"
            Exit 1
        }
    }
}

function Copy-DLLs() {

    # (py.exe -m site --user-site) would be more succinct but stupid windows gives stupid wrong path
	$site_dir = (py -m pip show pip | Select-String "Location:").Line.Split(" ")[1]
    Write-Host "Copying dlls into $site_dir/onnxruntime/capi"

    try {
        Copy-Item C:\Windows\System32\AMD\xrt_core.dll $site_dir\onnxruntime\capi\
        Copy-Item C:\Windows\System32\AMD\xrt_coreutil.dll $site_dir\onnxruntime\capi\
        Copy-Item C:\Windows\System32\AMD\amd_xrt_core.dll $site_dir\onnxruntime\capi\
        # cp onnx\dlls\onnxruntime.dll $site_dir\onnxruntime\capi\
        # cp onnx\dlls\onnxruntime_vitisai_ep.dll $site_dir\onnxruntime\capi\
    } catch {
        Write-Host "Failed to copy dlls: $_" -ForegroundColor red -BackgroundColor black
        Exit 1
    }
	Write-Host "Successfully copied dlls, ONNX Runtime all set!"
}

function Get-Binaries() {

    Write-Host "Downloading binaries"

    $xclbinDest = "$notebookDest\onnx\xclbins"

    if (-Not (Test-Path $xclbinDest)) {
        $null = mkdir $xclbinDest
    }
    
    try {
        # Download xclbins and jsons
        Invoke-WebRequest -Uri "https://github.com/amd/RyzenAI-SW/raw/c9d3db1418c0f7ae15a617fa0b79f12d8dbf6e24/demo/cloud-to-client/xclbin/1x4.xclbin" -OutFile "$xclbinDest\1x4.xclbin"
        Invoke-WebRequest -Uri "https://github.com/amd/RyzenAI-SW/raw/c9d3db1418c0f7ae15a617fa0b79f12d8dbf6e24/demo/cloud-to-client/models/vaip_config.json" -OutFile "$xclbinDest\vaip_config.json"

    } catch {
        Write-Host "Failed to download xclbins from https://github.com/amd/RyzenAI-SW: $_"
        Exit 1
    }
}

function Install-Prerequisites() {
    Write-Host "Installing ONNX Runtime python requirements"
    try {
        $requirementsPath = Join-Path -Path $PSScriptRoot -ChildPath "requirements.txt"
        Start-Process -FilePath py -ArgumentList "-m pip install -r $requirementsPath" -Wait -NoNewWindow -RedirectStandardOutput onnx_setup_stdout_log.txt -RedirectStandardError onnx_setup_stderr_log.txt
    } catch {
        Write-Host "Failed to download VOE prerequisites from requirements.txt: $_"
        Exit 1
    }
}

function Remove-Cache() {
    $cacheDir = "C:\temp\$username\vaip\.cache"
    if (Test-Path $cacheDir){
        Write-Host "Clearing NPU model cache"
        try {
            Remove-Item -Path $cacheDir -Recurse -Force
        } catch {
            Write-Host "WARNING: Failed to clear model cache. $_"
            Write-Host "You can do this manually by deleting: $cacheDir"
        }
    }
}

Get-Wheels
Get-Binaries
Install-Prerequisites
Install-ONNXRuntime
Copy-DLLs
Remove-Cache
