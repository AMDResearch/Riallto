// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>


void plusone_aie(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes) {
    ::aie::vector<uint8_t, 32> buffer;
    ::aie::vector<uint8_t, 32> inverted_buffer;
    uint16_t loop_count = (nbytes) >> 5;
    for(int j=0; j<loop_count; j++) {
        buffer = ::aie::load_v<32>(in_buffer);
        inverted_buffer = ::aie::add((uint8_t)1, buffer);
        in_buffer += 32;
        ::aie::store_v((uint8_t*)out_buffer, inverted_buffer);
        out_buffer += 32;
    }
}

extern "C" {

void plusone(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes) {
    plusone_aie(in_buffer, out_buffer, nbytes);
}

}