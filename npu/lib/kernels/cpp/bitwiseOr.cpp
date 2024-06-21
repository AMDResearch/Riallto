// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>


template <typename T, int N>
void bitwiseOR_aie(const T* in_buffer1, const T*  in_buffer2, T* out_buffer,
                        const uint32_t nbytes) {

    for (int j = 0; j < nbytes; j += N)
        chess_prepare_for_pipelining chess_loop_range(14, ) // loop_range(14) - loop : 1 cycle
        {
            ::aie::vector<T, N> in1 = ::aie::load_v<N>(in_buffer1);
            in_buffer1 += N;
            ::aie::vector<T, N> in2 = ::aie::load_v<N>(in_buffer2);
            in_buffer2 += N;
            ::aie::vector<T, N> out;

            out = ::aie::bit_or(in1,in2);

            ::aie::store_v(out_buffer, out);
            out_buffer += N;
        }
}

extern "C" {
    void bitwiseOr(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, int32_t nbytes) {
        bitwiseOR_aie<uint8_t, 64>(in_buffer1, in_buffer2, out_buffer, nbytes);
    }
}