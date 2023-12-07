// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

::aie::vector<uint8,64> vector_broadcast(::aie::vector<uint8,16> e)
{
    v64uint8 lli = e.template grow<64>();
    lli = shuffle(lli, lli, T8_2x64_lo);
    lli = shuffle(lli, lli, T8_2x64_lo);
    return ::aie::vector<uint8,64>(lli);
}


void gray2rgba_aie(uint8_t *in_buffer, uint8_t *out_buffer, const int32_t nbytes)
{
    // Initialize alpha vector
    ::aie::vector<uint8,64> alpha255 = ::aie::zeros<uint8,64>();
    for(int i=0; i<16; i++) {
        alpha255[i*4+3] = 255;
    }

    for(int i = 0; i < nbytes; i+=16){
            ::aie::vector<uint8, 16> data_buf = ::aie::load_v<16>(in_buffer);
            in_buffer += 16;

            // vector shuffle
            ::aie::vector<uint8,64> out = vector_broadcast(data_buf);

            // bitwise OR with alpha value
            v64uint8 fout = bor(out, alpha255);

            ::aie::store_v(out_buffer, ::aie::vector<uint8,64>(fout));
            out_buffer += 64;
        }
}

extern "C" {
    void gray2rgba(uint8_t *in_buffer, uint8_t *out_buffer, int32_t nbytes){
        gray2rgba_aie(in_buffer, out_buffer, nbytes);
    }
}
