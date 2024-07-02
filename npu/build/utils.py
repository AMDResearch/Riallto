# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import platform
import subprocess

def is_win()->bool:
    """ Returns true if we are running this on Windows."""
    return platform.system() == 'Windows'

def is_win_path(path:str)->bool:
    """ Returns true if the path above is a Windows path """
    newpath = path.split('\\')
    return newpath[0].endswith(':')

def is_wsl_win_path(path:str)->bool:
    """ Returns true if this is a windows path into WSL """
    return path.startswith("\\\\wsl.localhost")

def wsl_prefix()->str:
    """ if we are running this on windows return the appropriate wsl prefix."""
    if is_win():
        return "wsl -d Riallto "
    else:
        return ""

def check_wsl_install()->None:
    try:
        output = subprocess.check_output(['wsl', '-d', 'Riallto', 'cat', '/proc/version'])
    except subprocess.CalledProcessError as e:
        print("Failed to detect Riallto WSL instance. Please see https://riallto.ai/install-riallto.html for installation instructions.")
        print(f"{e.output.decode()}")
        raise e
