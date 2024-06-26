#!/bin/bash
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

docker container stop riallto_ci || true

docker run -dit --rm --name riallto_ci \
        --cap-add=NET_ADMIN \
        -v $(pwd):/workspace \
        --device=/dev/accel/accel0:/dev/accel/accel0 \
        -w /workspace \
        riallto:latest \
        /bin/bash 

docker exec -i riallto_ci /bin/bash -c "source ~/.bashrc && cd /workspace/ && python3 -m pip install . && python3 -m pytest ./tests"

