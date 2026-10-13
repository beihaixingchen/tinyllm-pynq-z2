#include <arm_neon.h>
#include <stddef.h>

void arm_neon_sgemv_f32(const float *matrix, const float *vector,
                        float *output, int rows, int columns) {
    for (int row = 0; row < rows; ++row) {
        const float *weights = matrix + (size_t)row * (size_t)columns;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        int column = 0;

        for (; column + 15 < columns; column += 16) {
            float32x4_t x0 = vld1q_f32(vector + column);
            float32x4_t x1 = vld1q_f32(vector + column + 4);
            float32x4_t x2 = vld1q_f32(vector + column + 8);
            float32x4_t x3 = vld1q_f32(vector + column + 12);
            acc0 = vmlaq_f32(acc0, vld1q_f32(weights + column), x0);
            acc1 = vmlaq_f32(acc1, vld1q_f32(weights + column + 4), x1);
            acc2 = vmlaq_f32(acc2, vld1q_f32(weights + column + 8), x2);
            acc3 = vmlaq_f32(acc3, vld1q_f32(weights + column + 12), x3);
        }

        float32x4_t total = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        float32x2_t pair = vadd_f32(vget_low_f32(total), vget_high_f32(total));
        pair = vpadd_f32(pair, pair);
        float sum = vget_lane_f32(pair, 0);

        for (; column < columns; ++column) {
            sum += weights[column] * vector[column];
        }
        output[row] = sum;
    }
}

void arm_neon_sgemv_f32_pairrows(const float *matrix, const float *vector,
                                 float *output, int rows, int columns) {
    int row = 0;
    for (; row + 1 < rows; row += 2) {
        const float *weights0 = matrix + (size_t)row * (size_t)columns;
        const float *weights1 = weights0 + columns;
        float32x4_t acc00 = vdupq_n_f32(0.0f);
        float32x4_t acc01 = vdupq_n_f32(0.0f);
        float32x4_t acc02 = vdupq_n_f32(0.0f);
        float32x4_t acc03 = vdupq_n_f32(0.0f);
        float32x4_t acc10 = vdupq_n_f32(0.0f);
        float32x4_t acc11 = vdupq_n_f32(0.0f);
        float32x4_t acc12 = vdupq_n_f32(0.0f);
        float32x4_t acc13 = vdupq_n_f32(0.0f);
        int column = 0;

        for (; column + 15 < columns; column += 16) {
            float32x4_t x0 = vld1q_f32(vector + column);
            float32x4_t x1 = vld1q_f32(vector + column + 4);
            float32x4_t x2 = vld1q_f32(vector + column + 8);
            float32x4_t x3 = vld1q_f32(vector + column + 12);
            acc00 = vmlaq_f32(acc00, vld1q_f32(weights0 + column), x0);
            acc01 = vmlaq_f32(acc01, vld1q_f32(weights0 + column + 4), x1);
            acc02 = vmlaq_f32(acc02, vld1q_f32(weights0 + column + 8), x2);
            acc03 = vmlaq_f32(acc03, vld1q_f32(weights0 + column + 12), x3);
            acc10 = vmlaq_f32(acc10, vld1q_f32(weights1 + column), x0);
            acc11 = vmlaq_f32(acc11, vld1q_f32(weights1 + column + 4), x1);
            acc12 = vmlaq_f32(acc12, vld1q_f32(weights1 + column + 8), x2);
            acc13 = vmlaq_f32(acc13, vld1q_f32(weights1 + column + 12), x3);
        }

        float32x4_t total0 = vaddq_f32(vaddq_f32(acc00, acc01),
                                       vaddq_f32(acc02, acc03));
        float32x2_t pair0 = vadd_f32(vget_low_f32(total0), vget_high_f32(total0));
        pair0 = vpadd_f32(pair0, pair0);
        float sum0 = vget_lane_f32(pair0, 0);

        float32x4_t total1 = vaddq_f32(vaddq_f32(acc10, acc11),
                                       vaddq_f32(acc12, acc13));
        float32x2_t pair1 = vadd_f32(vget_low_f32(total1), vget_high_f32(total1));
        pair1 = vpadd_f32(pair1, pair1);
        float sum1 = vget_lane_f32(pair1, 0);

        for (; column < columns; ++column) {
            sum0 += weights0[column] * vector[column];
            sum1 += weights1[column] * vector[column];
        }
        output[row] = sum0;
        output[row + 1] = sum1;
    }

    if (row < rows) {
        arm_neon_sgemv_f32(matrix + (size_t)row * (size_t)columns,
                           vector, output + row, 1, columns);
    }
}

void arm_neon_sgemv_f32_fourrows(const float *matrix, const float *vector,
                                 float *output, int rows, int columns) {
    int row = 0;
    for (; row + 3 < rows; row += 4) {
        const float *weights0 = matrix + (size_t)row * (size_t)columns;
        const float *weights1 = weights0 + columns;
        const float *weights2 = weights1 + columns;
        const float *weights3 = weights2 + columns;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);
        int column = 0;

        for (; column + 3 < columns; column += 4) {
            float32x4_t x = vld1q_f32(vector + column);
            acc0 = vmlaq_f32(acc0, vld1q_f32(weights0 + column), x);
            acc1 = vmlaq_f32(acc1, vld1q_f32(weights1 + column), x);
            acc2 = vmlaq_f32(acc2, vld1q_f32(weights2 + column), x);
            acc3 = vmlaq_f32(acc3, vld1q_f32(weights3 + column), x);
        }

        float32x2_t pair0 = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
        pair0 = vpadd_f32(pair0, pair0);
        float32x2_t pair1 = vadd_f32(vget_low_f32(acc1), vget_high_f32(acc1));
        pair1 = vpadd_f32(pair1, pair1);
        float32x2_t pair2 = vadd_f32(vget_low_f32(acc2), vget_high_f32(acc2));
        pair2 = vpadd_f32(pair2, pair2);
        float32x2_t pair3 = vadd_f32(vget_low_f32(acc3), vget_high_f32(acc3));
        pair3 = vpadd_f32(pair3, pair3);
        float sum0 = vget_lane_f32(pair0, 0);
        float sum1 = vget_lane_f32(pair1, 0);
        float sum2 = vget_lane_f32(pair2, 0);
        float sum3 = vget_lane_f32(pair3, 0);

        for (; column < columns; ++column) {
            float value = vector[column];
            sum0 += weights0[column] * value;
            sum1 += weights1[column] * value;
            sum2 += weights2[column] * value;
            sum3 += weights3[column] * value;
        }
        output[row] = sum0;
        output[row + 1] = sum1;
        output[row + 2] = sum2;
        output[row + 3] = sum3;
    }

    int remaining = rows - row;
    if (remaining >= 2) {
        arm_neon_sgemv_f32_pairrows(
            matrix + (size_t)row * (size_t)columns, vector, output + row,
            2, columns
        );
        row += 2;
    }
    if (row < rows) {
        arm_neon_sgemv_f32(
            matrix + (size_t)row * (size_t)columns, vector, output + row,
            1, columns
        );
    }
}
