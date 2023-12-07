// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

void gray2rgba_aie_scalar(uint8_t *in_buffer, uint8_t *out_buffer, const int32_t nbytes) {
    for (int i = 0; i < nbytes; i++){
        uint8_t value = in_buffer[i];
        out_buffer[i*4] = value;
        out_buffer[i*4 + 1] = value;
        out_buffer[i*4 + 2] = value;
        out_buffer[i*4 + 3] = 255;
    }
}

extern "C" {
    void gray2rgba_scalar(uint8_t *in_buffer, uint8_t *out_buffer, int32_t nbytes){
        gray2rgba_aie_scalar(in_buffer, out_buffer, nbytes);
    }
}
