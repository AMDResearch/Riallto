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
__attribute__((noinline)) void threshold_rgba_aie(
    T* in_buffer, T* out_buffer, const int32_t nbytes, const T& thresh_val1, const T& thresh_val2, const T& thresh_val3, const T& thresh_val4, const T& max_val1, const T& max_val2, const T& max_val3, const T& max_val4, const uint8_t threshold_type) {
    ::aie::vector<T, N> data_out;
    ::aie::mask<N> temp_val;

    ::aie::vector<T, N> mask_zeros  = ::aie::zeros<T, N>();
    ::aie::vector<T, N> mask_thresh;
    ::aie::vector<T, N> mask_max;
    for(int i=0; i<N/4; i++) {
        mask_thresh[i*4]   = thresh_val1;
        mask_thresh[i*4+1] = thresh_val2;
        mask_thresh[i*4+2] = thresh_val3;
        mask_thresh[i*4+3] = thresh_val4;
        mask_max[i*4]      = max_val1;
        mask_max[i*4+1]    = max_val2;
        mask_max[i*4+2]    = max_val3;
        mask_max[i*4+3]    = max_val4;
    }

    switch (threshold_type) {
        case XF_THRESHOLD_TYPE_TRUNC:
            for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                data_out = ::aie::min(mask_thresh, data_buf1);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
            break;
        case XF_THRESHOLD_TYPE_BINARY:
            for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                temp_val = ::aie::lt(mask_thresh, data_buf1);
                data_out = ::aie::select(mask_zeros, mask_max, temp_val);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
            break;
        case XF_THRESHOLD_TYPE_BINARY_INV:
            for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                temp_val = ::aie::lt(mask_thresh, data_buf1);
                data_out = ::aie::select(mask_zeros, mask_max, temp_val);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
            break;
        case XF_THRESHOLD_TYPE_TOZERO:
            for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                temp_val = ::aie::lt(mask_thresh, data_buf1);
                data_out = ::aie::select(mask_zeros, data_buf1, temp_val);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
            break;
        case XF_THRESHOLD_TYPE_TOZERO_INV:
           for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                temp_val = ::aie::lt(mask_thresh, data_buf1);
                data_out = ::aie::select(data_buf1, mask_zeros, temp_val);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
            break;
        default:
            for (int j = 0; j < nbytes; j += N) // 16x samples per loop
            chess_prepare_for_pipelining chess_loop_range(14, ) {
                ::aie::vector<T, N> data_buf1 = ::aie::load_v(in_buffer); // in:00++15|_________|_________|_________
                in_buffer += N;
                data_out = ::aie::min(mask_thresh, data_buf1);
                ::aie::store_v(out_buffer, data_out);
                out_buffer += N;
            }
    }
}


extern "C" {
    void thresholdRgba(uint8_t *in_buffer, uint8_t *out_buffer, int32_t nbytes, uint8_t thresholdValue1, uint8_t thresholdValue2, uint8_t thresholdValue3, uint8_t thresholdValue4, uint8_t maxValue1, uint8_t maxValue2, uint8_t maxValue3, uint8_t maxValue4, uint8_t threshold_type) {
        threshold_rgba_aie<uint8_t, 64>(in_buffer, out_buffer, nbytes, thresholdValue1, thresholdValue2, thresholdValue3, thresholdValue4, maxValue1, maxValue2, maxValue3, maxValue4, threshold_type);
    }
}
