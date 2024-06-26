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

    // count % 3 is the new loaded data, which is 'next row'
    // (count + 1) % 3 is the data loaded previous 2 calls, 'previous row'
    // (counter + 2) % 3 === (counter - 1) % 3 is the central row.

    linebuffer_t lb;

    if (count == 0) { // no valid output can be generated as the second row
                      // has not arrived
        lb.line0 = linebuffer[0];
        lb.line1 = linebuffer[0];
        lb.line2 = linebuffer[0]; // fake 1, this output is wrong
    } else if (count == 1) { // first output row, with mirrored boundary
        lb.line0 = linebuffer[1];
        lb.line1 = linebuffer[0];
        lb.line2 = linebuffer[1];
    } else{ // normal case, line 0 is previous row
            // line 1 is central row
            // line 2 is next row
        lb.line0 = linebuffer[(count+1)%3];
        lb.line1 = linebuffer[(count+2)%3];
        lb.line2 = linebuffer[count%3];
    }

    // reset count is necessary
    if(count >= num_lines){
	    count = 0;
    }
    else{
        count++;
    }
    return lb;
}
