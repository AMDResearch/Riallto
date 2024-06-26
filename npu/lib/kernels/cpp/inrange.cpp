// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>


void in_range_aie(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes,
                  uint8_t range_low, uint8_t range_high) {

    uint16_t loop_count = nbytes >> 6; // Divide by 32 as we're operating 64 pixels at a time

    auto zeros_buf = aie::zeros<uint8_t, 64>();
    auto ones_buf = aie::broadcast<uint8_t, 64>((uint8_t)255);

    for(int j=0; j<loop_count; j++) {
        auto buffer = ::aie::load_v<64>(in_buffer); // Read 64 grayscale pixels into a vector from data memory
        in_buffer += 64; // Increment the 64 positions the input pointer to data memory

        auto mask_low  = ::aie::ge(buffer, range_low); // Generate boolean mask indicating weather the pixel is greater or equal than the threshold
        auto mask_high = ::aie::le(buffer, range_high); // Generate boolean mask indicating weather the pixel is less or equal than the threshold
        auto mask = mask_low & mask_high;

        auto data_out = ::aie::select(zeros_buf, ones_buf, mask); // Generate pixel out based on the mask

        ::aie::store_v(out_buffer, data_out); // Write the result vector to data memory
        out_buffer += 64; // Increment the 64 positions the output pointer to data memory
    }
}

extern "C" {

void in_range(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes, uint8_t range_low, uint8_t range_high) {

    in_range_aie(in_buffer, out_buffer, nbytes, range_low, range_high);
}

} // extern C
