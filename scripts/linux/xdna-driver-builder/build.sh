#!/bin/bash
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

directory=./linux

if [ ! -d "$directory" ]; then
	git clone https://github.com/AMD-SW/linux.git -b v6.8-iommu-sva-part4-v7 --depth=1 
else
	echo "already cloned the linux kernel"
fi

# Build a container that creates the appropriate linux kernel version
docker build \
  -t xdna_deb_builder:latest \
  ./

docker kill xdna_deb_builder_container || true

# Lauch an image with that container
docker run -dit --rm --name xdna_deb_builder_container \
  -v $(pwd):/workspace \
  -w /workspace/ \
  xdna_deb_builder:latest \
  /bin/bash

docker exec xdna_deb_builder_container bash -c "tar -zcvf driver.tar.gz /root/debs && mv driver.tar.gz /workspace/ubuntu24.04_npu_drivers.tar.gz"

## cleanup
#docker kill xdna_deb_builder_container || true
#docker image rm --force xdna_deb_builder:latest
