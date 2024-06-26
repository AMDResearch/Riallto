// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "filter2d_scalar.h"
#include "linebuffer.h"
#include <aie_api/aie.hpp>

void filter2d_720p_aie_scalar(uint8_t *in_buffer, uint8_t *out_buffer,
        int16_t coeff_0_0, int16_t coeff_0_1, int16_t coeff_0_2,
        int16_t coeff_1_0, int16_t coeff_1_1, int16_t coeff_1_2,
        int16_t coeff_2_0, int16_t coeff_2_1, int16_t coeff_2_2) {

     int16_t filter[3][3];
     filter[0][0] = coeff_0_0;
     filter[0][1] = coeff_0_1;
     filter[0][2] = coeff_0_2;
     filter[1][0] = coeff_1_0;
     filter[1][1] = coeff_1_1;
     filter[1][2] = coeff_1_2;
     filter[2][0] = coeff_2_0;
     filter[2][1] = coeff_2_1;
     filter[2][2] = coeff_2_2;
     int16_t *filter_ptr = &filter[0][0];

     linebuffer_t lb = linebuffer<1280>(in_buffer, 719);
     filter2d_3lines_aie_scalar(lb.line0, lb.line1, lb.line2, out_buffer, 1280, filter_ptr);
  }


extern "C" {

void filter2d_720p(uint8_t *in_buffer, uint8_t *out_buffer,
        int16_t coeff_0_0, int16_t coeff_0_1, int16_t coeff_0_2,
        int16_t coeff_1_0, int16_t coeff_1_1, int16_t coeff_1_2,
        int16_t coeff_2_0, int16_t coeff_2_1, int16_t coeff_2_2) {

     filter2d_720p_aie_scalar(in_buffer, out_buffer, coeff_0_0, coeff_0_1,
                              coeff_0_2, coeff_1_0, coeff_1_1, coeff_1_2,
                              coeff_2_0, coeff_2_1, coeff_2_2);
  }

}
