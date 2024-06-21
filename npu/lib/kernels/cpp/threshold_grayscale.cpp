// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

enum _threshold_type {
    XF_THRESHOLD_TYPE_BINARY = 0,
    XF_THRESHOLD_TYPE_BINARY_INV = 1,
    XF_THRESHOLD_TYPE_TRUNC = 2,
    XF_THRESHOLD_TYPE_TOZERO = 3,
    XF_THRESHOLD_TYPE_TOZERO_INV = 4,
};

template <typename T, int N>
__attribute__((noinline)) void threshold_grayscale_aie(
    T* in_buffer, T* out_buffer, const int32_t nbytes, const T& thresh_val, const T& max_val, const uint8_t threshold_type) {
    ::aie::vector<T, N> constants;
    ::aie::vector<T, N> data_out;
    ::aie::mask<N> temp_val;
    constants[0] = 0;          // updating constant zero_val value
    constants[1] = thresh_val; // updating constant threshold value
    constants[2] = max_val;    // updating constant max_val value

    switch (threshold_type) {
        case XF_THRESHOLD_TYPE_TRUNC:
            for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                data_out = ::aie::min(constants[1], data_buf1);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
            break;
        case XF_THRESHOLD_TYPE_BINARY:
            for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                temp_val = ::aie::lt(constants[1], data_buf1);
                data_out = ::aie::select(constants[0], constants[2], temp_val);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
            break;
        case XF_THRESHOLD_TYPE_BINARY_INV:
            for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                temp_val = ::aie::lt(constants[1], data_buf1);
                data_out = ::aie::select(constants[2], constants[0], temp_val);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
            break;
        case XF_THRESHOLD_TYPE_TOZERO:
            for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                temp_val = ::aie::lt(constants[1], data_buf1);
                data_out = ::aie::select(constants[0], data_buf1, temp_val);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
            break;
        case XF_THRESHOLD_TYPE_TOZERO_INV:
           for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                temp_val = ::aie::lt(constants[1], data_buf1);
                data_out = ::aie::select(data_buf1, constants[0], temp_val);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
            break;
        default:
            for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                data_out = ::aie::min(constants[1], data_buf1);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
    }
}


extern "C" {
    void threshold_grayscale(uint8_t *in_buffer, uint8_t *out_buffer, int32_t nbytes, uint8_t thresh_val, uint8_t max_val, uint8_t threshold_type) {
        threshold_grayscale_aie<uint8_t, 64>(in_buffer, out_buffer, nbytes, thresh_val, max_val, threshold_type);
    }
}
