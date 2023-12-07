# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import subprocess

def version_to_tuple(version):
    """ Convert a version string to a tuple of integers"""
    return tuple(map(int, version.split('.')))

def get_device_status():
    """ Returns states of the NPU device, when operating normally should return the string 'OK'."""
    command = f'''
    $device = Get-PnpDevice | Where-Object {{ $_.FriendlyName -eq "AMD IPU Device" }}
    if ($device -eq $null) {{
        return $null
    }} else {{
        return $device.Status
    }}
    '''

    result = subprocess.run(['powershell', '-Command', command], capture_output=True, text=True)
    return result.stdout.strip()

def get_driver_version():
    command = f'''
    $device = Get-WmiObject -Class Win32_PnPSignedDriver | Where-Object {{ $_.DeviceName -eq "AMD IPU Device" }}
    return $device.DriverVersion
    '''

    result = subprocess.run(['powershell', '-Command', command], capture_output=True, text=True)
    return result.stdout.strip()

def test_device_status():
    """ Checks if device exists, and if the status is expected 'OK'."""
    status = get_device_status()

    assert status is not None, "Device not found: AMD IPU Device"
    assert status == "OK", f"Device status: {status}"


def test_driver_version(min_driver_version="10.1109.8.100"):
    """ Asserts if the driver version matches a minimum specified version."""

    min_version = version_to_tuple(min_driver_version)
    version = version_to_tuple(get_driver_version())

    assert version >= min_version, f"Current driver version: {version}, expected minimum: {min_version}"

def reset_npu():
    """ This is a utility function that wraps a powershell script to 
    reset the IPU device.
    Note: this requires admin privileges.
    """
    
    command = '''
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
    '''

    result = subprocess.run(['powershell', '-Command', command], capture_output=True, text=True)
    return result