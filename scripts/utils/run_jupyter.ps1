# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

$username = [System.Environment]::UserName
$notebooksPath = "C:\Users\$username\AppData\Roaming\riallto_notebooks"

$venvPath = Join-Path -Path $PSScriptRoot -ChildPath "activate_venv.ps1"
. $venvPath

Write-Output "Launching jupyter server..."

Set-Location $notebooksPath
# py -m jupyterlab one_notebook_mlir.ipynb
Start-Process py -ArgumentList "-m", "jupyterlab", "1_0_Introduction.ipynb" -WindowStyle Minimized