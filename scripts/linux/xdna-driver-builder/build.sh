#!/bin/bash
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

KERNEL_HEADERS=linux-headers-6.10.0-061000rc2_6.10.0-061000rc2.202406022333_all.deb
KERNEL_HEADERS_GENERIC=linux-headers-6.10.0-061000rc2-generic_6.10.0-061000rc2.202406022333_amd64.deb
KERNEL_MODULES=linux-modules-6.10.0-061000rc2-generic_6.10.0-061000rc2.202406022333_amd64.deb
KERNEL_IMAGE=linux-image-unsigned-6.10.0-061000rc2-generic_6.10.0-061000rc2.202406022333_amd64.deb

rm -rf _work
mkdir -p _work
wget -P _work https://kernel.ubuntu.com/mainline/v6.10-rc2/amd64/$KERNEL_HEADERS_GENERIC
wget -P _work https://kernel.ubuntu.com/mainline/v6.10-rc2/amd64/$KERNEL_HEADERS
wget -P _work https://kernel.ubuntu.com/mainline/v6.10-rc2/amd64/$KERNEL_IMAGE
wget -P _work https://kernel.ubuntu.com/mainline/v6.10-rc2/amd64/$KERNEL_MODULES

# Build a container that creates the appropriate linux kernel version
docker build \
  -t xdna_deb_builder:latest \
  --build-arg KERNEL_HEADERS=$KERNEL_HEADERS \
  --build-arg KERNEL_HEADERS_GENERIC=$KERNEL_HEADERS_GENERIC \
  --build-arg KERNEL_MODULES=$KERNEL_MODULES \
  --build-arg KERNEL_IMAGE=$KERNEL_IMAGE \
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
docker kill xdna_deb_builder_container || true
docker image rm --force xdna_deb_builder:latest
