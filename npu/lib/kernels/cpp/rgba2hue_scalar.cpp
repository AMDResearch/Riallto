// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>
#include "lut_inv_8b.h"


void rgba2hue_aie_scalar(uint8_t *in_buffer, uint8_t *out_buffer, const uint32_t nbytes) {
    for(int i = 0; i < nbytes; i++) {
        int r = (int) in_buffer[i*4];
        int g = (int) in_buffer[i*4 + 1];
        int b = (int) in_buffer[i*4 + 2];
        int h;
        uint8_t rgbMin, rgbMax;

        rgbMin = r < g ? (r < b ? r : b) : (g < b ? g : b);
        rgbMax = r > g ? (r > b ? r : b) : (g > b ? g : b);

        if (rgbMax == 0 || rgbMax == rgbMin)
            h = 0;
        else if (rgbMax == r)
            h = 0 + 85 * (g - b) / (rgbMax - rgbMin); // h = 0 + 42.5*(g - b) / (rgbMax - rgbMin);
        else if (rgbMax == g)
            h = 85*2 + 85 * (b - r) / (rgbMax - rgbMin); // h = 85 + 42.5*(b - r) / (rgbMax - rgbMin);
        else
            h = 170*2 + 85 * (r - g) / (rgbMax - rgbMin); // h = 170 + 42.5*(r - g) / (rgbMax - rgbMin);

        h = (h+1) >> 1;
        out_buffer[i] = (uint8_t)h;
    }
}

extern "C" {
    void rgba2hue_scalar(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes){
        rgba2hue_aie_scalar(in_buffer, out_buffer, nbytes);
    }
}
