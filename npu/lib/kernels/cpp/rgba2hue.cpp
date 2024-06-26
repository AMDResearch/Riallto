// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>
#include "lut_inv_8b.h"


const int32_t SRS_SHIFT = 12;

__attribute__((inline)) void xf_extract_rgb(uint8_t* ptr_rgba,
                                            ::aie::vector<uint8_t, 32>& r,
                                            ::aie::vector<uint8_t, 32>& g,
                                            ::aie::vector<uint8_t, 32>& b) {
    ::aie::vector<uint8_t, 32> rgba_channel0, rgba_channel1, rgba_channel3, rgba_channel2;
    rgba_channel0 = ::aie::load_v<32>(ptr_rgba);
    ptr_rgba += 32;
    rgba_channel1 = ::aie::load_v<32>(ptr_rgba);
    ptr_rgba += 32;
    rgba_channel2 = ::aie::load_v<32>(ptr_rgba);
    ptr_rgba += 32;
    rgba_channel3 = ::aie::load_v<32>(ptr_rgba);
    ptr_rgba += 32;

    // Unzip the interleaved channels
    auto[rg_temp, ba_temp] = ::aie::interleave_unzip(::aie::concat(rgba_channel0, rgba_channel1),
                                                     ::aie::concat(rgba_channel2, rgba_channel3), 2);
    r = ::aie::filter_even(rg_temp, 1);
    g = ::aie::filter_odd(rg_temp, 1);
    b = ::aie::filter_even(ba_temp, 1);
}

__attribute__((inline)) void comp_divisor_16b(::aie::vector<uint8_t, 32> divisor,
                                             ::aie::vector<uint16_t, 32>& divisor_select) {
    const int step = 0;
    using lut_type_uint16 = aie::lut<4, uint16, uint16>;
    lut_type_uint16 inv_lut_16b(num_entries_lut_inv_16b,
                        lut_inv_16b_ab,lut_inv_16b_cd);
    aie::parallel_lookup<uint8, lut_type_uint16, aie::lut_oor_policy::truncate>
        lookup_inv_16b(inv_lut_16b, step);

    aie::vector<uint8,16> input1, input2;
    aie::vector<uint16,16> res1, res2;
    input1 = divisor.extract<16>(0);
    input2 = divisor.extract<16>(1);
    res1 = lookup_inv_16b.fetch(input1.cast_to<uint8>());
    res2 = lookup_inv_16b.fetch(input2.cast_to<uint8>());
    divisor_select = aie::concat(res1, res2);
}

__attribute__((noinline)) void rgba2hue_aie(uint8_t *in_buffer, uint8_t *out_buffer, const uint32_t nbytes) {
    ::aie::vector<uint8_t, 32> r, g, b;
    ::aie::vector<uint8_t, 32> hue;

    ::aie::vector<uint8_t, 32> rgbMin, rgbMax;

    ::aie::vector<uint8_t, 32> zero32         = aie::zeros<uint8_t, 32>();

    ::aie::vector<int16_t, 32> eightFive      = aie::zeros<int16_t, 32>();
    eightFive[0] = 85; eightFive[1] = -85;
    ::aie::vector<int16_t, 32> one            = aie::broadcast<int16_t, 32>(1);
    ::aie::vector<int16_t, 32> twoEightFive   = aie::broadcast<int16_t, 32>(171); // 170 + 1
    ::aie::vector<int16_t, 32> fourEightFive  = aie::broadcast<int16_t, 32>(341); // 340 + 1

    for (int j = 0; (j < nbytes / (32*4)); j += 1) chess_prepare_for_pipelining {
        xf_extract_rgb(in_buffer, r, g, b);

        // Get rgbMin and rgbMax
        rgbMin = ::aie::min(::aie::min(r, g), b);
        rgbMax = ::aie::max(::aie::max(r, g), b);

        // Get divisor and select the fixed point divisor to multiply by
        auto divisor = ::aie::sub(rgbMax, rgbMin);
        ::aie::vector<uint16, 32> divisor_sel;
        comp_divisor_16b(divisor, divisor_sel);

        // Initialize accum with value since 340 is larger than uint8
        aie::accum<acc32,32> hr_partial(one,9);
        aie::accum<acc32,32> hg_partial(twoEightFive,9);
        aie::accum<acc32,32> hb_partial(fourEightFive,9);

        // Performa uin8*int16 vector multiply
        hr_partial = aie::mac(hr_partial, g, divisor_sel);
        hg_partial = aie::mac(hg_partial, b, divisor_sel);
        hb_partial = aie::mac(hb_partial, r, divisor_sel);

        hr_partial = aie::msc(hr_partial, b, divisor_sel);
        hg_partial = aie::msc(hg_partial, r, divisor_sel);
        hb_partial = aie::msc(hb_partial, g, divisor_sel);

        auto hr = hr_partial.to_vector<uint8>(10); // Q7.9 shift + 1 (div 2)
        auto hg = hg_partial.to_vector<uint8>(10); // Q7.9 shift + 1 (div 2)
        auto hb = hb_partial.to_vector<uint8>(10); // Q7.9 shift + 1 (div 2)

        aie::mask<32> sel1 = aie::eq(rgbMax, r);
        auto          tmp1 = aie::select(hb, hr, sel1);
        aie::mask<32> sel2 = aie::eq(rgbMax, g);
        auto          tmp2 = aie::select(tmp1, hg, sel2);
        aie::mask<32> sel3 = aie::eq(divisor, zero32);
        hue                = aie::select(tmp2, zero32, sel3);

        ::aie::store_v(out_buffer, hue);
        in_buffer += 128;
        out_buffer += 32;
    }
}

extern "C" {
    void rgba2hue(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes){
        rgba2hue_aie(in_buffer, out_buffer, nbytes);
    }
}