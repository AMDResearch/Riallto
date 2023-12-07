// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

const int32_t SRS_SHIFT = 15;
__attribute__((inline)) void xf_extract_rgb(uint8_t* in_buffer,
                                            ::aie::vector<uint8_t, 32>& r,
                                            ::aie::vector<uint8_t, 32>& g,
                                            ::aie::vector<uint8_t, 32>& b) {
    ::aie::vector<uint8_t, 32> rgba_channel0, rgba_channel1, rgba_channel3, rgba_channel2;
    rgba_channel0 = ::aie::load_v<32>(in_buffer);
    in_buffer += 32;
    rgba_channel1 = ::aie::load_v<32>(in_buffer);
    in_buffer += 32;
    rgba_channel2 = ::aie::load_v<32>(in_buffer);
    in_buffer += 32;
    rgba_channel3 = ::aie::load_v<32>(in_buffer);
    in_buffer += 32;

    // Unzip the interleaved channels
    auto[rg_temp, ba_temp] = ::aie::interleave_unzip(::aie::concat(rgba_channel0, rgba_channel1),
                                                     ::aie::concat(rgba_channel2, rgba_channel3), 2);
    r = ::aie::filter_even(rg_temp, 1);
    g = ::aie::filter_odd(rg_temp, 1);
    b = ::aie::filter_even(ba_temp, 1);
}

__attribute__((noinline)) void rgba2gray_aie(uint8_t *in_buffer, uint8_t *out_buffer, const uint32_t nbytes) {
    //::aie::vector<int16_t, 16> WT(66, 129, 25, 128); //Y=0.299*R + 0.587*G + 0.114*B (BT.470)
    //::aie::vector<int16_t, 16> WT(25, 129, 66, 128); //Y=0.299*R + 0.587*G + 0.114*B (BT.470)
    ::aie::vector<int16_t, 16> WT((int16_t)round(0.299*(1<<SRS_SHIFT)), (int16_t)round(0.587*(1<<SRS_SHIFT)), (int16_t)round(0.114*(1<<SRS_SHIFT)), (1<<(SRS_SHIFT-1))); //Y=0.299*R + 0.587*G + 0.114*B (BT.470)
    ::aie::vector<uint8_t, 32> c1 = ::aie::broadcast<uint8_t, 32>(1);
    ::aie::vector<uint8_t, 32> r, g, b;
    ::aie::vector<uint8_t, 32> y;

    for (int j = 0; (j < nbytes / (32*4)); j += 1) chess_prepare_for_pipelining {
        xf_extract_rgb(in_buffer, r, g, b);

        ::aie::accum<acc32, 32> acc;
        acc = ::aie::accumulate<32>(WT, 0, r, g, b, c1);
        y = acc.template to_vector<uint8_t>(SRS_SHIFT);

        ::aie::store_v(out_buffer, y);
        in_buffer += 128;
        out_buffer += 32;
    }
}

extern "C" {
    void rgba2gray(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes){
        rgba2gray_aie(in_buffer, out_buffer, nbytes);
    }
}
