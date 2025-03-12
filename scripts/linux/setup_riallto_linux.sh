#!/bin/bash

# Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -e

DRIVER_TARBALL=ubuntu24.04_npu_drivers.tar.gz
NPU_FIRMWARE="/lib/firmware/amdnpu/1502_00/npu.sbin"

MLIR_FILE="https://www.xilinx.com/bin/public/openDownload?filename=pynqMLIR_AIE_py312_v0.9.tar.gz"
RIALLTO_FILE="https://www.xilinx.com/bin/public/openDownload?filename=Riallto-v1.1.zip"

############# CHECKS ##################################
# Check to ensure key URLs are accessible
URLS=(
$MLIR_FILE
$RIALLTO_FILE
"https://github.com/amd/xdna-driver.git"
"https://github.com/AMDResearch/Riallto.git"
"https://docker.io/library/ubuntu"
)

for URL in "${URLS[@]}"; do
        if curl --output /dev/null --silent --head --fail "$URL"; then
                echo "[CHECK OK] URL is reachable: $URL"
        else
                echo "WARNING: [CHECK FAILED] URL is not reachable: $URL"
                echo "Some parts of the installation might not work correctly."
                while true; do
                        read -p "Are you happy to continue? [Y/N]  " answer
                        case $answer in
                                [Yy]* ) echo "Continuing..."; break;;
                                [Nn]* ) echo "Exiting"; exit 1;;
                                * ) echo "Please chose Y or N.";;
                        esac
                done

        fi
done

# Check that we are on Ubuntu24.04.2
distro_info=$(lsb_release -d)
if [[ $distro_info != *"Ubuntu 24.04.2"* ]]; then
	echo "Riallto is only currently supported on Ubuntu 24.04.2"
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
if [ "$#" -lt 1 ]; then
	echo "Usage $0 <Xilinx license file> <username (optional)>"
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
build_xrt=0

if [ -f "./xdna-driver-builder/${DRIVER_TARBALL}" ]; then
	echo "NPU driver is available, just setting up Riallto"
	build_xrt=0;
else
	build_xrt=1
fi

if [ $build_xrt -eq 1 ]; then
	# Building the NPU driver
	if [ ! -f "./xdna-driver-builder/${DRIVER_TARBALL}" ]; then
		echo "xdna-driver-builder/${DRIVER_TARBALL} is missing, building it from scratch"
		pushd xdna-driver-builder
		./build.sh
		popd
	else
		echo "Driver tarball already exists."
	fi

fi

# Install the NPU drivers in the host (xdna-driver)
if [ ! -f "${NPU_FIRMWARE}" ]; then
	npu_install_tmp_dir=$(mktemp -d)
	tar -xzvf "./xdna-driver-builder/${DRIVER_TARBALL}" -C "${npu_install_tmp_dir}"
	pushd $npu_install_tmp_dir/root/debs
		sudo apt -y --fix-broken install
		sudo apt install -y ocl-icd-libopencl1 dkms
		sudo -E dpkg -i xrt_*-amd64-base.deb
		sudo -E dpkg -i xrt_*-amd64-npu.deb || true
		sudo -E dpkg -i xrt_plugin*-amdxdna.deb || true
	popd
fi
#########################################################

########### Riallto Docker image construction ###########
echo "Building Riallto docker image"
build_tmp=./_work
rm -rf $build_tmp
mkdir -p $build_tmp

USER_NAME="$USER"

if [ $# -eq 2 ]; then
    USER_NAME="$2"
fi

USER_ID=`id -u $USER_NAME`
GROUP_ID=`id -g $USER_NAME`
GROUP_NAME=`id -g -n $USER_NAME`

## Checks to make sure that all the required tarballs and license are in the directory
if [ ! -f "./build_tmp/pynqMLIR-AIE.tar.gz" ]; then
	echo "Error! pynqMLIR-AIE.tar.gz is missing, downloading from opendownloads..."
	wget -O $build_tmp/pynqMLIR-AIE.tar.gz $MLIR_FILE
else
	cp pynqMLIR-AIE.tar.gz $build_tmp
fi

if [ ! -f "./build_tmp/xilinx_tools.tar.gz" ]; then
	echo "xilinx_tools.tar.gz is missing, downloading it from opendownloads..."
	wget -O $build_tmp/riallto_installer.zip $RIALLTO_FILE
	pushd $build_tmp
		unzip riallto_installer.zip
        mv Riallto-v1.1/Riallto/downloads/xilinx_tools_latest.tar.gz ./xilinx_tools.tar.gz
	popd
	cp $build_tmp/Riallto-v1.1/eula.txt ./
else
	cp xilinx_tools.tar.gz $build_tmp
fi

while true; do
	read -p "Do you agree to the terms in ./eula.txt and wish to proceed [y/n]? " answer
	case $answer in
		[Yy]* ) echo "Terms accepted"; break;;
		[Nn]* ) echo "Exiting"; exit 1;;
		* ) echo "Please chose Y or N.";;
	esac
done

cp $LIC_FILE $build_tmp/Xilinx.lic

tar -xzvf ./xdna-driver-builder/${DRIVER_TARBALL} -C $build_tmp/

docker build \
	--build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
	--build-arg LIC_MAC=$MAC \
        --build-arg USER_ID=${USER_ID} \
        --build-arg GROUP_ID=${GROUP_ID} \
        --build-arg GROUP_NAME=${GROUP_NAME} \
	--build-arg BUILD_TEMPDIR=$build_tmp \
       	-t riallto:latest \
	./

rm -rf $build_tmp
#####################################################

