// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

const int32_t SRS_SHIFT = 14;

template <typename T, int N, int MAX>
void addweighted_aie_scalar(const T* in_buffer1, const T*  in_buffer2, T* out_buffer,
                            const uint32_t nbytes,
                            const int16_t alpha, const int16_t beta, const T gamma) {
    for (int i = 0; i < nbytes; i++) {
        int tmpIn1 = in_buffer1[i]*alpha;
        int tmpIn2 = in_buffer2[i]*beta;
        int tmp = ((tmpIn1 + tmpIn2 + (1<<(SRS_SHIFT-1))) >> SRS_SHIFT) + gamma;
        tmp = (tmp > MAX) ? MAX : (tmp < 0) ? 0 : tmp; //saturate
        out_buffer[i] = (T)tmp;
    }

}

extern "C" {
    void addweighted_scalar(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, int32_t nbytes, int16_t alpha, int16_t beta, uint8_t gamma) {
        addweighted_aie_scalar<uint8_t, 32, UINT8_MAX>(in_buffer1, in_buffer2, out_buffer, nbytes, alpha, beta, gamma);
    }
}
