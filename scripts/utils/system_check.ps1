# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

param(
  [Parameter(Mandatory=$false)]
  [string]$lightInstall = "false"
)

# Need to do conversion if script called from C#
$lightInstallBool = [System.Boolean]::Parse($lightInstall)

# "Whitelist" of processors
# $knownProcessors = "AMD Ryzen 5 7640HS", "AMD Ryzen 7 7840H", "AMD Ryzen 7 7840HS", "AMD Ryzen 7 7840U", "AMD Ryzen 9 7940H", "AMD Ryzen 9 7940HS"

# IPU device
$deviceName = "AMD IPU Device"

# Minimum version, casting to special version type very useful - can compare directly without string mannpulations
$minimumVersion = [version]"10.1109.8.100"

# Required disk space (looking at C: drive)
$requiredSpaceGB = 20

# Recommended RAM
$requiredMemoryGB = 8

# Max line length for padding
$maxLineLength = 80

# Helper test message formatter
# Usage: Show-Status -message "My test message" -isPass $testResult -failMessage "Printed below the main message if testResult=false"
function Show-Status {
    param (
        [string]$message,
        [bool]$isPass,
        [string]$failMessage = ""
    )

    $status = if ($isPass) { "OK" } else { "FAIL" }
    $color = if ($isPass) { "Green" } else { "Red" }
    $message = $message.PadRight($maxLineLength, ".")
    Write-Host -NoNewline $message
    Write-Host $status -ForegroundColor $color
    if (!$isPass -and $failMessage) {
        Write-Host $failMessage -ForegroundColor "Red"
        $script:allTestsPass = $false
    }
}

# This will be set to false if any test fails
$allTestsPass = $true

Write-Host "--- System Information ---"

# Operating System check
$osVersion = (Get-WmiObject -Class Win32_OperatingSystem).Caption
$osApproved = $osVersion -match "Windows 11"
Show-Status -message "Operating System: $osVersion" -isPass $osApproved -failMessage "- Only Windows 11 is supported."

# CPU name
# $processorName = (Get-WmiObject -class Win32_Processor).Name.trim()
# $cpuApproved = $knownProcessors | Where-Object { $processorName -match $_ }
# Show-Status -message "CPU: $processorName" -isPass ($null -ne $cpuApproved) -failMessage "- $processorName is not on list of supported devices with an IPU."

# Find our device
$allDevices = Get-WmiObject Win32_PnPEntity
$device = $allDevices | Where-Object {$_.Name -eq $deviceName}

# Device and version check
if ($device) {

    # If device exists automatic pass
    Show-Status -message "$($device.Name) exists" -isPass $true
    
    # if device exists we can also check status
    $deviceStatusIsOk = $device.Status -eq "OK"
    Show-Status -message "Device status" -isPass $deviceStatusIsOk -failMessage "- IPU status not OK, make sure $deviceName is enabled."

    # Check if device version is less than expected
    $driver = Get-WmiObject -Class Win32_PnPSignedDriver | Where-Object { $_.DeviceName -eq $deviceName }
    $driverVersion = $driver.DriverVersion
    $isVersionOk = ([version]$driverVersion -ge $minimumVersion)
    Show-Status -message "Driver version $driverVersion" -isPass $isVersionOk -failMessage "- Minimum driver version is $minimumVersion."
} else {
    Show-Status -message "$deviceName exists" -isPass $false -failMessage "Device $deviceName not found."
}

# Get the hard drive free spacea
$drive = Get-PSDrive -Name C
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)
$isSpaceOk = ($freeSpaceGB -ge $requiredSpaceGB) # greater than or equal
Show-Status -message "Drive C: $freeSpaceGB GB free" -isPass $isSpaceOk -failMessage "- Insufficient space - installation requires at least $requiredSpaceGB GB"

# RAM, using Win32_PhysicalMemory to get the installed physical memory (16GB)
# alternatively use (Get-WmiObject -class Win32_ComputerSystem).TotalPhysicalMemory to get actual mem, which would show something like 15.2GB
$memoryBytes = (Get-WmiObject -Class Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum
$memoryGB = [Math]::Round(($memoryBytes /1GB), 2)
$isMemoryOk = ($memoryGB -ge $requiredMemoryGB) # greater than or equal
Show-Status -message "Memory: $memoryGB GB" -isPass $isMemoryOk -failMessage "- Required memory is $requiredMemoryGB GB"

# # Do a network test to make sure our servers are reachable
# $server = "www.amd.com"
# try {
#     Test-Connection -ComputerName $server -Count 1 -ErrorAction Stop | Out-Null
#     $networkReachable = $true
# }
# catch {
#     $networkReachable = $false
# }

# Show-Status -message "Network connectivity" -isPass $networkReachable -failMessage "Did not get response from $server"

if (-Not $lightInstallBool){
    # WSL feature check
    $wsl_enabled = (Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux).State -eq 'Enabled'
    Show-Status -message "Microsoft-Windows-Subsystem-Linux feature enabled:" -isPass $wsl_enabled -failMessage "- Windows Subsystem for Linux feature is not enabled. See https://riallto.ai/prerequisites-wsl.html for help."

    # WSL2
    # if ($wsl_enabled) {
    #     if ((wsl -l -v 2>&1) -match '2') { $wsl2_enabled = $true } else { $wsl2_enabled = $false }
    #     Show-Status -message "WSL2 Kernel:" -isPass $wsl2_enabled -failMessage "WSL2 isn't being used, try to set it with wsl --set-default-version 2"
    # }

    if ($wsl_enabled) {

        $virtualMachineEnabled = (Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State -eq 'Enabled'
        Show-Status -message "VirtualMachinePlatform feature enabled:" -isPass $virtualMachineEnabled -failMessage "- Virtual Machine Platform feature not enabled. See https://riallto.ai/prerequisites-wsl.html for help."    

        $virtualizationEnabled = Get-WmiObject -Class Win32_ComputerSystem | Select-Object -ExpandProperty HypervisorPresent
        Show-Status -message "Virtualization enabled:" -isPass $virtualizationEnabled -failMessage "- Hypervisor not found, virtualization may not be enabled in your BIOS/UEFI settings."

        wsl --set-default-version 2 >$null 2>&1

        if ($LastExitCode -eq 0) {
            $wsl2_enabled = $true
        } else {
            $wsl2_enabled = $false
        }
        Show-Status -message "WSL2 Kernel installed:" -isPass $wsl2_enabled -failMessage "- Couldn't verify WSL2 kernel installation, run wsl --update or please check https://aka.ms/wsl2kernel to manually install WSL2 kernel."
    }
}

# Check the final status
if ($allTestsPass) {
    exit 0
} else {
    Exit 1
}