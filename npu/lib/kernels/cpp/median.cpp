// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>

constexpr unsigned VecFactor = 32;

__attribute__((always_inline))
aie::vector<uint8, 32> vsort3(aie::vector<uint8,32> &line0,
                              aie::vector<uint8,32> &line1,
                              aie::vector<uint8,32> &line2)
{
  aie::vector<uint8,32> tmp1 = aie::min(line0, line1);
  aie::vector<uint8,32> tmp2 = aie::max(line0, line1);

  aie::vector<uint8,32> tmp3 = aie::min(tmp2, line2);
  return aie::max(tmp1, tmp3);
}

void median1D_aie(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes)
{
  aie::vector<uint8, 32> line0, line1, line2, prev;
  aie::vector<uint8, 64> buf;

  // Left of nbytes, border extension by mirroring
  buf.insert(0, aie::load_v<32>(in_buffer)); in_buffer += VecFactor;
  buf.insert(1, aie::load_v<32>(in_buffer)); in_buffer += VecFactor;
  line1 = buf.extract<32>(0);
  // prev = line1;
  line0 = (aie::shuffle_up_replicate(buf, 1)).extract<32>(0);
  line2 = (aie::shuffle_down(buf, 1)).extract<32>(0);

  aie::vector<uint8,32> res = vsort3(line0, line1, line2);
  aie::store_v(out_buffer, res); out_buffer += VecFactor;

  // Middle of nbytes, no border extension needed
  int cnt=VecFactor;
  for(; cnt<nbytes-VecFactor; cnt+=VecFactor) {
    line0 = (aie::shuffle_up(buf, 1)).extract<32>(1);
    line1 = buf.extract<32>(1);

    buf.insert(0, buf.extract<32>(1));
    buf.insert(1, aie::load_v<32>(in_buffer)); in_buffer += VecFactor;
    line2 = (aie::shuffle_down(buf, 1)).extract<32>(0);

    res = vsort3(line0, line1, line2);
    aie::store_v(out_buffer, res); out_buffer += VecFactor;
  }
  // Right of nbytes, border extension by mirroring
  if(cnt < nbytes-1) {
    line0 = (aie::shuffle_up(buf, 1)).extract<32>(1);
    line1 = buf.extract<32>(1);
    line2 = (aie::shuffle_down_replicate(line1, 1)).extract<32>(0);
    line2[nbytes-cnt] = line2[nbytes-cnt-1];

    res = vsort3(line0, line1, line2);

    // If nbytes is a multiple of 32
    aie::store_v(out_buffer, res);

    // If out_buffer is NOT a multiple of 32
    // for(int j=0; j<nbytes-cnt; j++) {
    //   *out_buffer++ = res[j];
    // }
  }
}

extern "C" {

  void median(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes) {
      median1D_aie(in_buffer, out_buffer, nbytes);
  }

}
