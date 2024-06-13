// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

const int32_t SHIFT = 14;

template <typename T, int N, int MAX>
void addweighted_aie(const T* in_buffer1, const T*  in_buffer2, T* out_buffer,
                        const uint32_t nbytes,
                        const int16_t alphaFixedPoint, const int16_t betaFixedPoint, const T gamma) {

    ::aie::set_saturation(aie::saturation_mode::saturate); // Needed to saturate properly to uint8

    ::aie::vector<int16_t, N> coeff(alphaFixedPoint, betaFixedPoint);
    ::aie::vector<T, N> gamma_coeff;
    ::aie::accum<acc32, N> gamma_acc;
    for (int i = 0; i < N; i++) {
        gamma_coeff[i] = gamma;
    }
    gamma_acc.template from_vector(gamma_coeff, 0);
    for (int j = 0; j < nbytes; j += N)             // N samples per loop
        chess_prepare_for_pipelining chess_loop_range(14, ) // loop_range(14) - loop : 1 cycle
        {
            ::aie::vector<T, N> data_buf1 = ::aie::load_v<N>(in_buffer1);
            in_buffer1 += N;
            ::aie::vector<T, N> data_buf2 = ::aie::load_v<N>(in_buffer2);
            in_buffer2 += N;
            ::aie::accum<acc32, N> acc = ::aie::accumulate<N>(
                gamma_acc, coeff, 0, data_buf1, data_buf2); // weight[0] * data_buf1 + weight[1] * data_buf2
            ::aie::store_v(out_buffer, acc.template to_vector<T>(SHIFT));
            out_buffer += N;
        }
}

extern "C" {
    void addWeighted(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, uint32_t nbytes, int16_t alpha, int16_t beta, uint8_t gamma) {
        addweighted_aie<uint8_t, 32, UINT8_MAX>(in_buffer1, in_buffer2, out_buffer, nbytes, alpha, beta, gamma);
    }
}
