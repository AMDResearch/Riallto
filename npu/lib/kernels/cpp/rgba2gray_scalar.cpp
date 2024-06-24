// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

void rgba2gray_aie_scalar(uint8_t *in_buffer, uint8_t *out_buffer, const uint32_t nbytes) {
    ///Y=0.299*R + 0.587*G + 0.114*B (BT.470)
    const int colorMatrix[4] = {(int)round(0.299*65536),(int)round(0.587*65536),(int)round(0.114*65536), (65536/2)};
    for(int i = 0; i < nbytes; i++) {
        int r = (int) in_buffer[i*4];
        int g = (int) in_buffer[i*4 + 1];
        int b = (int) in_buffer[i*4 + 2];
        int tmpSum = (colorMatrix[0]*r + colorMatrix[1]*g + colorMatrix[2]*b + colorMatrix[3]) >> 16;
        out_buffer[i] = (uint8_t)tmpSum;
    }
}

extern "C" {
    void rgba2gray_scalar(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes) {
        rgba2gray_aie_scalar(in_buffer, out_buffer, nbytes);
    }
}
