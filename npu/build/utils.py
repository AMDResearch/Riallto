# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import subprocess

def is_win()->bool:
    """ Returns true if we are running this on windows."""
    return os.name == "nt"

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