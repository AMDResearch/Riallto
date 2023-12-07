// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

const int32_t SRS_SHIFT = 12;

void filter2d_3lines_aie_scalar(uint8_t *lineIn0, uint8_t *lineIn1, uint8_t *lineIn2, uint8_t *out_buffer,
                    const int32_t nbytes,
                    int16_t *kernel)
{
    int32_t acc;

    // left of line, border extension by mirroring
    acc = 0;
    acc +=  ((int32_t)lineIn0[0]) * kernel[0 * 3 + 0];
    acc +=  ((int32_t)lineIn1[0]) * kernel[1 * 3 + 0];
    acc +=  ((int32_t)lineIn2[0]) * kernel[2 * 3 + 0];

    for (int ki = 1; ki < 3; ki++) {
        acc +=  ((int32_t)lineIn0[0 + ki - 1]) * kernel[0 * 3 + ki];
        acc +=  ((int32_t)lineIn1[0 + ki - 1]) * kernel[1 * 3 + ki];
        acc +=  ((int32_t)lineIn2[0 + ki - 1]) * kernel[2 * 3 + ki];
    }
    acc = ((acc + (1<<(SRS_SHIFT-1))) >> SRS_SHIFT);
    acc = (acc > UINT8_MAX) ? UINT8_MAX : (acc < 0) ? 0 : acc; //saturate
    out_buffer[0] = (uint8_t)acc;


    // middle of line, no border extension needed
    for (int i = 1; i < nbytes-1; i++) {
        acc = 0;
        for (int ki = 0; ki < 3; ki++) {
            acc +=  ((int32_t)lineIn0[i + ki - 1]) * kernel[0 * 3 + ki];
            acc +=  ((int32_t)lineIn1[i + ki - 1]) * kernel[1 * 3 + ki];
            acc +=  ((int32_t)lineIn2[i + ki - 1]) * kernel[2 * 3 + ki];
        }
        acc = ((acc + (1<<(SRS_SHIFT-1))) >> SRS_SHIFT);
        acc = (acc > UINT8_MAX) ? UINT8_MAX : (acc < 0) ? 0 : acc; //saturate
        out_buffer[i] = (uint8_t)acc;
    }

    // right of line, border extension by mirroring
    acc = 0;
    for (int ki = 0; ki < 2; ki++) {
        acc += ((int32_t)lineIn0[nbytes + ki - 2]) * kernel[0 * 3 + ki];
        acc += ((int32_t)lineIn1[nbytes + ki - 2]) * kernel[1 * 3 + ki];
        acc += ((int32_t)lineIn2[nbytes + ki - 2]) * kernel[2 * 3 + ki];
    }

    acc += ((int32_t)lineIn0[nbytes-1]) * kernel[0 * 3 + 2];
    acc += ((int32_t)lineIn1[nbytes-1]) * kernel[1 * 3 + 2];
    acc += ((int32_t)lineIn2[nbytes-1]) * kernel[2 * 3 + 2];
    acc = ((acc + (1<<(SRS_SHIFT-1))) >> SRS_SHIFT);
    acc = (acc > UINT8_MAX) ? UINT8_MAX : (acc < 0) ? 0 : acc; //saturate
    out_buffer[nbytes-1] = (uint8_t)acc;
}

