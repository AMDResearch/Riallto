// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>


template <typename T, int N>
void bitwiseOR_aie_scalar(const T* in_buffer1, const T*  in_buffer2, T* out_buffer,
                        const uint32_t nbytes) {
    for (int i = 0; i < nbytes; i++){
        out_buffer[i] = in_buffer1[i] | in_buffer2[i];
    }

}

extern "C" {
    void bitwiseor_scalar(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, int32_t nbytes) {
        bitwiseOR_aie_scalar<uint8_t, 64>(in_buffer1, in_buffer2, out_buffer, nbytes);
    }
}
