#include <arm_neon.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

static inline int32_t horizontal_sum_s32(int32x4_t value) {
    int32x2_t pair = vadd_s32(vget_low_s32(value), vget_high_s32(value));
    pair = vpadd_s32(pair, pair);
    return vget_lane_s32(pair, 0);
}

void arm_neon_q6_vocab_sgemv_f32(const int8_t *matrix,
                                  const float *row_scales,
                                  const int16_t *vector,
                                  float activation_scale,
                                  float *output,
                                  int rows,
                                  int columns) {
    int row = 0;
    for (; row + 3 < rows; row += 4) {
        const int8_t *weights0 = matrix + (size_t)(row + 0) * (size_t)columns;
        const int8_t *weights1 = matrix + (size_t)(row + 1) * (size_t)columns;
        const int8_t *weights2 = matrix + (size_t)(row + 2) * (size_t)columns;
        const int8_t *weights3 = matrix + (size_t)(row + 3) * (size_t)columns;
        int32x4_t acc0_low = vdupq_n_s32(0);
        int32x4_t acc0_high = vdupq_n_s32(0);
        int32x4_t acc1_low = vdupq_n_s32(0);
        int32x4_t acc1_high = vdupq_n_s32(0);
        int32x4_t acc2_low = vdupq_n_s32(0);
        int32x4_t acc2_high = vdupq_n_s32(0);
        int32x4_t acc3_low = vdupq_n_s32(0);
        int32x4_t acc3_high = vdupq_n_s32(0);
        int column = 0;

        for (; column + 7 < columns; column += 8) {
            int16x8_t activation = vld1q_s16(vector + column);
            int16x8_t weight0 = vmovl_s8(vld1_s8(weights0 + column));
            int16x8_t weight1 = vmovl_s8(vld1_s8(weights1 + column));
            int16x8_t weight2 = vmovl_s8(vld1_s8(weights2 + column));
            int16x8_t weight3 = vmovl_s8(vld1_s8(weights3 + column));

            acc0_low = vmlal_s16(acc0_low, vget_low_s16(weight0),
                                 vget_low_s16(activation));
            acc0_high = vmlal_s16(acc0_high, vget_high_s16(weight0),
                                  vget_high_s16(activation));
            acc1_low = vmlal_s16(acc1_low, vget_low_s16(weight1),
                                 vget_low_s16(activation));
            acc1_high = vmlal_s16(acc1_high, vget_high_s16(weight1),
                                  vget_high_s16(activation));
            acc2_low = vmlal_s16(acc2_low, vget_low_s16(weight2),
                                 vget_low_s16(activation));
            acc2_high = vmlal_s16(acc2_high, vget_high_s16(weight2),
                                  vget_high_s16(activation));
            acc3_low = vmlal_s16(acc3_low, vget_low_s16(weight3),
                                 vget_low_s16(activation));
            acc3_high = vmlal_s16(acc3_high, vget_high_s16(weight3),
                                  vget_high_s16(activation));
        }

        int32_t sums[4] = {
            horizontal_sum_s32(acc0_low) + horizontal_sum_s32(acc0_high),
            horizontal_sum_s32(acc1_low) + horizontal_sum_s32(acc1_high),
            horizontal_sum_s32(acc2_low) + horizontal_sum_s32(acc2_high),
            horizontal_sum_s32(acc3_low) + horizontal_sum_s32(acc3_high),
        };
        for (; column < columns; ++column) {
            int32_t activation = vector[column];
            sums[0] += (int32_t)weights0[column] * activation;
            sums[1] += (int32_t)weights1[column] * activation;
            sums[2] += (int32_t)weights2[column] * activation;
            sums[3] += (int32_t)weights3[column] * activation;
        }

        output[row + 0] = (float)sums[0] * (row_scales[row + 0] * activation_scale);
        output[row + 1] = (float)sums[1] * (row_scales[row + 1] * activation_scale);
        output[row + 2] = (float)sums[2] * (row_scales[row + 2] * activation_scale);
        output[row + 3] = (float)sums[3] * (row_scales[row + 3] * activation_scale);
    }

    for (; row < rows; ++row) {
        const int8_t *weights = matrix + (size_t)row * (size_t)columns;
        int32_t sum = 0;
        for (int column = 0; column < columns; ++column) {
            sum += (int32_t)weights[column] * (int32_t)vector[column];
        }
        output[row] = (float)sum * (row_scales[row] * activation_scale);
    }
}
