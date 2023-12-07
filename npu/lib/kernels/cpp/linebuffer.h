// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

struct linebuffer_t{
    uint8_t* line0;
    uint8_t* line1;
    uint8_t* line2;
};

template<int TWidth>
linebuffer_t linebuffer(uint8_t * input, uint32_t num_lines) {
    static uint8_t linebuffer[3][TWidth];
    static uint32_t count = 0;

    memcpy(linebuffer[count%3], input, TWidth);

    linebuffer_t lb;
    
    if (count == 0) {
          lb.line0 = linebuffer[0];
          lb.line1 = linebuffer[0];
          lb.line2 = linebuffer[1];
    } else {
	if (count == num_lines) {
              lb.line0 = linebuffer[0];
    	      lb.line1 = linebuffer[1];
    	      lb.line2 = linebuffer[1];
	} else {
              lb.line0 = linebuffer[count%3];
    	      lb.line1 = linebuffer[(count+1)%3];
    	      lb.line2 = linebuffer[(count+2)%3];
	}
    }

    count++;
    return lb;
}
