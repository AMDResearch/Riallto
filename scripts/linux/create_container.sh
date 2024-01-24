#!/bin/bash

docker run -dit --rm --name riallto_docker \
        --cap-add=NET_ADMIN \
        -v $(pwd):/workspace \
        --device=/dev/accel/accel0:/dev/accel/accel0 \
        -v /lib/firmware/amdnpu/1502:/lib/firmware/amdnpu/1502 \
        -w /workspace \
        riallto:2024_03_12 \
        /bin/bash

docker exec -it riallto_docker /bin/bash
