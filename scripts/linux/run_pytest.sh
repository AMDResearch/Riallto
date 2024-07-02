#!/bin/bash
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

docker container stop riallto_pytest || true
docker container wait riallto_pytest || true

docker run -dit --rm --name riallto_pytest \
        --cap-add=NET_ADMIN \
        -v $(pwd):/workspace \
        --device=/dev/accel/accel0:/dev/accel/accel0 \
        -w /workspace \
        riallto:latest \
        /bin/bash 

docker exec -it riallto_pytest /bin/bash -c "source ~/.bashrc && cd /home/riallto/Riallto && python3 -m pytest ./tests"

docker container stop riallto_pytest || true
