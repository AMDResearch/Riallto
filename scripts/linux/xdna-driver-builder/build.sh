#!/bin/bash
#
# Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


# Build a container that creates the appropriate linux kernel version
docker build \
  -t npubase:latest \
  ./

docker kill npubase_build_container || true

# Lauch an image with that container
docker run -dit --rm --name npubase_build_container \
  -v $(pwd):/workspace \
  -w /workspace/ \
  npubase:latest \
  /bin/bash

# Execute container to get the driver and plugin
docker exec npubase_build_container bash -c "tar -zcvf driver.tar.gz /root/debs && mv driver.tar.gz /workspace/ubuntu24.04_npu_drivers.tar.gz"

## cleanup
docker kill npubase_build_container || true
