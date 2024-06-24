#!/bin/bash
#
# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

docker container stop riallto_jupyter || true

webcams=$(ls /dev/video*)
cmd="docker run -dit --rm --name riallto_jupyter"
cmd+=" --cap-add=NET_ADMIN --device=/dev/accel/accel0:/dev/accel/accel0"
cmd+=" -p 8888:8888 "
cmd+=" -v ./notebooks:/notebooks "
for cam in $webcams; do
        cmd+=" --device=$cam:$cam"
done
cmd+=" -w /workspace riallto:latest /bin/bash"

echo " running $cmd"
eval $cmd

docker exec -it riallto_jupyter /bin/bash -c " sudo chmod 666 /dev/video* && source ~/.bashrc && cd /notebooks && python3 -m jupyterlab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=''"

docker container stop riallto_jupyter || true
