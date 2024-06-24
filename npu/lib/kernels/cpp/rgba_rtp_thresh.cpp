// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

void rgba_rtp_thresh_aie(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes, uint8_t r_thresh, uint8_t g_thresh, uint8_t b_thresh) {
    uint8_t pixel_rtps[4];
    pixel_rtps[0] = r_thresh;
    pixel_rtps[1] = g_thresh;
    pixel_rtps[2] = b_thresh;
    pixel_rtps[3] = 0;

    ::aie::vector<uint8_t, 64> t_vec = ::aie::broadcast<uint32_t, 16>(*(uint32_t*)pixel_rtps).cast_to<uint8_t>();
    auto sat_vec = ::aie::broadcast<uint8_t, 64>(255);
    auto zero_vec = ::aie::zeros<uint8_t, 64>();

    uint16_t loop_count = (nbytes) >> 6;
    for(int j=0; j< loop_count; j++){
        auto i_buf = ::aie::load_v<64>(in_buffer);
        auto lt_mask = ::aie::lt(t_vec, i_buf);
        auto o_buf = ::aie::select(zero_vec, sat_vec, lt_mask);
        ::aie::store_v(out_buffer, o_buf);

        in_buffer += 64;
        out_buffer += 64;
    }
}

extern "C" {

void rgba_rtp_thresh(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes, uint8_t r_thresh, uint8_t g_thresh, uint8_t b_thresh) {
    rgba_rtp_thresh_aie(in_buffer, out_buffer, nbytes, r_thresh, g_thresh, b_thresh);
}

} // extern C
