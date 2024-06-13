// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aie_api/aie.hpp>

const int32_t ACC_SHIFT = 12;

#define KERNEL_WIDTH 3

    constexpr unsigned VecFactor    = 32;

    constexpr unsigned Lanes        = 32; // Parallel vector output lanes
    constexpr unsigned Points       = 8;  // Columns where data in summed together
    constexpr unsigned CoeffStep    = 1;
    constexpr unsigned DataStepXY   = 1;

    using mul_ops = aie::sliding_mul_xy_ops<Lanes, Points, CoeffStep, DataStepXY, int8, uint8>;


void filter2d_3lines_aie(uint8_t *lineIn0, uint8_t *lineIn1, uint8_t *lineIn2, uint8_t *out_buffer,
                    const uint32_t nbytes,
                    int16_t *kernel)
{

    set_sat(); // Needed for int16 to saturate properly to uint8

    aie::vector<uint8, 64> data_buf1, data_buf2, data_buf3;
    aie::vector<uint8, 64> prev_buf1, prev_buf2, prev_buf3;
    aie::vector<uint8, 64> zero_buf = ::aie::zeros<uint8, 64>();
    aie::vector<int8,32> kernel_vec;

    const uint32_t kernel_side = KERNEL_WIDTH/2;

    for(int j=0; j<KERNEL_WIDTH; j++) {
        for(int i=0; i<KERNEL_WIDTH; i++) {
            kernel_vec[j*Points+i] = (int8_t)((*kernel)>>8); // int16 to int8 shift
            kernel++;
        }
        for(int i2=0; i2<Points-KERNEL_WIDTH; i2++) {
            kernel_vec[j*Points+KERNEL_WIDTH+i2] = 0;
        }
    }

    // left of line, border extension by mirroring
    // first kernel row
    data_buf1.insert(0, aie::load_v<32>(lineIn0)); lineIn0+=VecFactor;
    data_buf1.insert(1, aie::load_v<32>(lineIn0));
    prev_buf1.insert(1, data_buf1.template extract<32>(0));
    data_buf1 = ::aie::shuffle_up_replicate(data_buf1, kernel_side);
    auto acc  = mul_ops::mul(     kernel_vec, 0, data_buf1, 0);

    // second kernel row
    data_buf2.insert(0, aie::load_v<32>(lineIn1)); lineIn1+=VecFactor;
    data_buf2.insert(1, aie::load_v<32>(lineIn1));
    prev_buf2.insert(1, data_buf2.template extract<32>(0));
    data_buf2 = ::aie::shuffle_up_replicate(data_buf2, kernel_side);
    acc       = mul_ops::mac(acc, kernel_vec, Points, data_buf2, 0);

    // third kernel row
    data_buf3.insert(0, aie::load_v<32>(lineIn2)); lineIn2+=VecFactor;
    data_buf3.insert(1, aie::load_v<32>(lineIn2));
    prev_buf3.insert(1, data_buf3.template extract<32>(0));
    data_buf3 = ::aie::shuffle_up_replicate(data_buf3, kernel_side);
    acc       = mul_ops::mac(acc, kernel_vec, 2*Points, data_buf3, 0);

    // Store result
    ::aie::store_v(out_buffer, acc.to_vector<uint8>(ACC_SHIFT-8)); out_buffer+=VecFactor;

    // middle of line, no border extension needed
    for (int i = 2*VecFactor; i < nbytes-1; i+=VecFactor) {
        // first kernel row
        data_buf1.insert(0, aie::load_v<32>(lineIn0)); lineIn0+=VecFactor;
        data_buf1.insert(1, aie::load_v<32>(lineIn0));
        data_buf1 = ::aie::shuffle_up_fill(data_buf1, prev_buf1, kernel_side);
        prev_buf1.insert(1, data_buf1.template extract<32>(0));
        acc       = mul_ops::mul(     kernel_vec, 0, data_buf1, 0);

        // second kernel row
        data_buf2.insert(0, aie::load_v<32>(lineIn1)); lineIn1+=VecFactor;
        data_buf2.insert(1, aie::load_v<32>(lineIn1));
        data_buf2 = ::aie::shuffle_up_fill(data_buf2, prev_buf2, kernel_side);
        prev_buf2.insert(1, data_buf2.template extract<32>(0));
        acc       = mul_ops::mac(acc, kernel_vec, Points, data_buf2, 0);

        // third kernel row
        data_buf3.insert(0, aie::load_v<32>(lineIn2)); lineIn2+=VecFactor;
        data_buf3.insert(1, aie::load_v<32>(lineIn2));
        data_buf3 = ::aie::shuffle_up_fill(data_buf3, prev_buf3, kernel_side);
        prev_buf3.insert(1, data_buf3.template extract<32>(0));
        acc       = mul_ops::mac(acc, kernel_vec, 2*Points, data_buf3, 0);

        // Store result
        ::aie::store_v(out_buffer, acc.to_vector<uint8>(ACC_SHIFT-8)); out_buffer+=VecFactor;
    }

    // right of line, border extension by mirroring
    // first kernel row
    data_buf1.insert(1, aie::load_v<32>(lineIn0));
    data_buf1 = ::aie::shuffle_down_replicate(data_buf1, 32);
    data_buf1 = ::aie::shuffle_up_fill(data_buf1, prev_buf1, kernel_side);
    acc       = mul_ops::mul(     kernel_vec, 0, data_buf1, 0);

    // second kernel row
    data_buf2.insert(1, aie::load_v<32>(lineIn1));
    data_buf2 = ::aie::shuffle_down_replicate(data_buf2, 32);
    data_buf2 = ::aie::shuffle_up_fill(data_buf2, prev_buf2, kernel_side);
    acc       = mul_ops::mac(acc, kernel_vec, Points, data_buf2, 0);

    // third kernel row
    data_buf3.insert(1, aie::load_v<32>(lineIn2)); lineIn2+=VecFactor;
    data_buf3 = ::aie::shuffle_down_replicate(data_buf3, 32);
    data_buf3 = ::aie::shuffle_up_fill(data_buf3, prev_buf3, kernel_side);
    acc       = mul_ops::mac(acc, kernel_vec, 2*Points, data_buf3, 0);

    // Store result
    ::aie::store_v(out_buffer, acc.to_vector<uint8>(ACC_SHIFT-8)); out_buffer+=VecFactor;
}

