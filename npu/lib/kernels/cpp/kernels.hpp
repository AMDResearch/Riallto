// Copyright 2023 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

void addWeighted(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, uint32_t nbytes, int16_t alpha, int16_t beta, uint8_t gamma);
void addweighted_scalar(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, int32_t nbytes, int16_t alpha, int16_t beta, uint8_t gamma);
void bitwiseand(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, int32_t nbytes);
void bitwiseand_scalar(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, uint32_t nbytes);
void bitwiseOr(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, int32_t nbytes);
void bitwiseor_scalar(uint8_t *in_buffer1, uint8_t *in_buffer2, uint8_t *out_buffer, int32_t nbytes);
void filter2d_720p(uint8_t *in_buffer, uint8_t *out_buffer, int16_t coeff_0_0, int16_t coeff_0_1, int16_t coeff_0_2, int16_t coeff_1_0, int16_t coeff_1_1, int16_t coeff_1_2, int16_t coeff_2_0, int16_t coeff_2_1, int16_t coeff_2_2);
void filter2d_1080p(uint8_t *in_buffer, uint8_t *out_buffer, int16_t coeff_0_0, int16_t coeff_0_1, int16_t coeff_0_2, int16_t coeff_1_0, int16_t coeff_1_1, int16_t coeff_1_2, int16_t coeff_2_0, int16_t coeff_2_1, int16_t coeff_2_2);
void gray2rgba(uint8_t *in_buffer, uint8_t *out_buffer, int32_t nbytes);
void gray2rgba_scalar(uint8_t *in_buffer, uint8_t *out_buffer, int32_t nbytes);
void in_range(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes, uint8_t range_low, uint8_t range_high);
void inverse(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes);
void median(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes);
void median_scalar(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes);
void plusone(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes);
void plusn(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes, uint8_t n);
void rgba_inverse(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes);
void rgba_rtp_thresh(uint8_t *in_buffer, uint8_t* out_buffer, uint32_t nbytes, uint8_t r_thresh, uint8_t g_thresh, uint8_t b_thresh);
void rgba2gray(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes);
void rgba2gray_scalar(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes);
void rgba2hue(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes);
void rgba2hue_scalar(uint8_t *in_buffer, uint8_t *out_buffer, uint32_t nbytes);
void threshold_grayscale(uint8_t *in_buffer, uint8_t *out_buffer, int32_t nbytes, uint8_t thresh_val, uint8_t max_val, uint8_t threshold_type);
void thresholdRgba(uint8_t *in_buffer, uint8_t *out_buffer, int32_t nbytes, uint8_t thresholdValue1, uint8_t thresholdValue2, uint8_t thresholdValue3, uint8_t thresholdValue4, uint8_t maxValue1, uint8_t maxValue2, uint8_t maxValue3, uint8_t maxValue4, uint8_t threshold_type);