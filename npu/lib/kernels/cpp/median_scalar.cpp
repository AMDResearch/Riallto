// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

constexpr int kernelSize = 3;

// FROM: https://www.softwaretestinghelp.com/insertion-sort/
template <typename T>
static void insertion_sort(T *a, int32_t size) {
  for(int k=1; k<size; k++)
    {
        T temp = a[k];
        int j= k-1;
        while(j>=0 && temp <= a[j])
        {
            a[j+1] = a[j];
            j = j-1;
        }
        a[j+1] = temp;
    }
}

void median1D_aie_scalar(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes)
{
  uint8_t in[kernelSize];

  // left
  in[0] = in_buffer[0];
  for (int ki = 1; ki < kernelSize; ki++)
    in[ki] = in_buffer[ki - 1];

  insertion_sort<uint8_t>(in,kernelSize);
  out_buffer[0] = in[kernelSize/2];

  // middle
  for (int i = 1; i < nbytes-1; i++) {
    for (int ki = 0; ki < kernelSize; ki++)
      in[ki] = in_buffer[i + ki - 1];

    insertion_sort<uint8_t>(in,kernelSize);
    out_buffer[i] = in[kernelSize/2];
  }

  // right
  for (int ki = 0; ki < (kernelSize-1); ki++)
    in[ki] = in_buffer[nbytes + ki - (kernelSize-1)];
  in[kernelSize-1] = in_buffer[nbytes - 1];

  insertion_sort<uint8_t>(in,kernelSize);
  out_buffer[nbytes-1] = in[kernelSize/2];

}

extern "C" {

  void median_scalar(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes) {
      median1D_aie_scalar(in_buffer, out_buffer, nbytes);
  }

}
