#!/bin/bash

# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

DRIVER_TARBALL=ubuntu24.04_npu_drivers.tar.gz
MIN_KERNEL_VERSION="6.8.8+"
NPU_FIRMWARE="/lib/firmware/amdnpu/1502_00/npu.sbin"

# Check that we are on Ubuntu24.04
distro_info=$(lsb_release -d)
if [[ $distro_info != *"Ubuntu 24.04"* ]]; then
	echo "Riallto is only currently supported on Ubuntu 24.04"
	exit 1
fi

# Check that docker is installed
if command -v docker >/dev/null 2>&1; then
	echo "Docker has been found."
else
	echo "Docker could not be found on this system."
       	echo "Unable to continue the installation."
        echo "Please configure docker using the instructions found here:"
	echo "https://docs.docker.com/engine/install/ubuntu/"
	echo "And then rerun the script"	
	exit 1
fi

############### License file check ###################
# Check to make sure that a license file has been provided and that 
# a MAC address can be extracted from it for adding into the docker
# image
if [ "$#" -ne 1 ]; then
	echo "Usage $0 <Xilinx license file>"
	exit 1
fi
LIC_FILE="$1"

# Check to make sure that the license file exists
if [ ! -f "$LIC_FILE" ]; then
	echo "Unable to open the license file $LIC_FILE"
	exit 1
fi

MAC=$(grep -oP 'HOSTID=\K[^;]+' $1 | head -n1 | sed 's/\(..\)/\1:/g; s/:$//')
echo "Found a License file associated with MAC address $MAC"
####################################################


######### Kernel and NPU driver check / install ###########
# Check to see if the kernel version and NPU driver is already installed
build_kernel_and_xrt=0
kernel_version=$(uname -r)

if [[ "$kernel_version" == "$MIN_KERNEL_VERSION" ]]; then
	echo "Kernel version is okay, is NPU available?"	
	if [ -f "${NPU_FIRMWARE}" ]; then	
		echo "NPU is available, just setting up Riallto"
		build_kernel_and_xrt=0;			
	else
		build_kernel_and_xrt=1
	fi
else
	echo "Kernel version is not the correct version for running Riallto"	
	echo "A non mainline linux kernel will have to be installed"
	echo "Kernel version=$kernel_version  need at least  $MIN_KERNEL_VERSION"
	build_kernel_and_xrt=1
fi

if [ $build_kernel_and_xrt -eq 1 ]; then
	# Building the driver and kernel version and installing it
	# First check to make sure that secure boot is disabled.
	if mokutil --sb-state | grep -q "enabled"; then
		echo "Secure boot is currently enabled."
		echo "To install Riallto on Linux currently requires a" 
		echo "non-mainline kernel version ${MIN_KERNEL_VERSION}."
	       	echo "If you would like to continue with the installation "
	        echo "please disable secure boot in your bios settings and rerun this script."
		exit 1	
	fi
	
	if [ ! -f "./xdna-driver-builder/${DRIVER_TARBALL}" ]; then
		echo "xdna-driver-builder/${DRIVER_TARBALL} is missing, building it from scratch"
		pushd xdna-driver-builder
		./build.sh
		popd
	else
		echo "Kernel and driver tarball already exists."
	fi


	if [[ "$kernel_version" != "$MIN_KERNEL_VERSION" ]]; then
		echo "To install Riallto requires upgrading your kernel to ${MIN_KERNEL_VERSION}"
		echo "After upgrading you will have to restart your machine and rerun this script"
		while true; do
			read -p "Are you happy to continue? [Y/N]  " answer
			case $answer in
				[Yy]* ) echo "You chose yes, attempting to update kernel"; break;;
				[Nn]* ) echo "Exiting"; exit 1;;
				* ) echo "Please chose Y or N.";;
			esac
		done
			
		kernel_bump_tmp_dir=$(mktemp -d)
		tar -xzvf "./xdna-driver-builder/${DRIVER_TARBALL}" -C "${kernel_bump_tmp_dir}"
		pushd $kernel_bump_tmp_dir/root/debs
			sudo dpkg -i linux-headers*_amd64.deb
			sudo dpkg -i linux-image*_amd64.deb
			sudo dpkg -i linux-libc*_amd64.deb
		popd
		echo -e "\033[31mPlease now restart your machine and rerun the script.\033[0m"
		exit 1
	fi
fi

# Install the NPU drivers (xdna-driver)
if [ ! -f "${NPU_FIRMWARE}" ]; then	
	npu_install_tmp_dir=$(mktemp -d)
	tar -xzvf "./xdna-driver-builder/${DRIVER_TARBALL}" -C "${npu_install_tmp_dir}"
	pushd $npu_install_tmp_dir/root/debs
		sudo apt -y --fix-broken install 
		sudo -E dpkg -i xrt_*-amd64-xrt.deb
		sudo -E dpkg -i xrt_plugin*-amdxdna.deb 
	popd
fi
#########################################################

########### Riallto Docker image construction ###########
echo "Building Riallto docker image" 
build_tmp=./_work
rm -rf $build_tmp
mkdir -p $build_tmp

## Checks to make sure that all the required tarballs and license are in the directory 
if [ ! -f "./pynqMLIR-AIE.tar.gz" ]; then
	echo "Error! pynqMLIR-AIE.tar.gz is missing, downloading from opendownloads..."
	wget -O $build_tmp/pynqMLIR-AIE.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=pynqMLIR_AIE_py312_v0.9.tar.gz 
else
	cp pynqMLIR-AIE.tar.gz $build_tmp
fi

if [ ! -f "./xilinx_tools.tar.gz" ]; then
	echo "xilinx_tools.tar.gz is missing, downloading it from opendownloads..."
	wget -O $build_tmp/riallto_installer.zip https://www.xilinx.com/bin/public/openDownload?filename=Riallto-v1.0.zip
	pushd $build_tmp
		unzip riallto_installer.zip
        	mv Riallto_v1.0/Riallto/downloads/xilinx_tools_latest.tar.gz ./xilinx_tools.tar.gz	
	popd
else
	cp xilinx_tools.tar.gz $build_tmp
fi

cp $LIC_FILE $build_tmp/Xilinx.lic

tar -xzvf ./xdna-driver-builder/${DRIVER_TARBALL} -C $build_tmp/

docker build \
	--build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
	--build-arg LIC_MAC=$MAC \
	--build-arg BUILD_TEMPDIR=$build_tmp \
       	-t riallto:latest \
	./

rm -rf $build_tmp
#####################################################

