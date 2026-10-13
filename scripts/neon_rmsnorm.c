#include <arm_neon.h>
#include <math.h>

void arm_neon_rmsnorm_f32(const float *vector, const float *weight, float *output,
                          int length, float epsilon) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int index = 0;

    for (; index + 3 < length; index += 4) {
        float32x4_t value = vld1q_f32(vector + index);
        sum = vmlaq_f32(sum, value, value);
    }

    float total = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1)
        + vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
    for (; index < length; ++index) {
        total += vector[index] * vector[index];
    }

    float inverse_rms = 1.0f / sqrtf(total / (float)length + epsilon);
    float32x4_t inverse = vdupq_n_f32(inverse_rms);
    index = 0;
    for (; index + 3 < length; index += 4) {
        float32x4_t value = vld1q_f32(vector + index);
        float32x4_t scale = vld1q_f32(weight + index);
        vst1q_f32(output + index, vmulq_f32(vmulq_f32(value, inverse), scale));
    }
    for (; index < length; ++index) {
        output[index] = vector[index] * inverse_rms * weight[index];
    }
}
