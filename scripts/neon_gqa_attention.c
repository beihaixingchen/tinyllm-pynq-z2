#include <arm_neon.h>
#include <math.h>
#include <stddef.h>

static inline float horizontal_sum(float32x4_t value) {
    float32x2_t pair = vadd_f32(vget_low_f32(value), vget_high_f32(value));
    pair = vpadd_f32(pair, pair);
    return vget_lane_f32(pair, 0);
}

void arm_neon_gqa_attention_f32(const float *queries, const float *keys,
                                const float *values, float *output,
                                int sequence, int kv_heads,
                                int groups_per_kv, int head_dim) {
    if (sequence <= 0 || sequence > 512 || kv_heads <= 0 || kv_heads > 4 ||
        groups_per_kv <= 0 || kv_heads * groups_per_kv > 16 ||
        head_dim <= 0 || head_dim > 256) {
        return;
    }

    const int head_count = kv_heads * groups_per_kv;
    const float scale = 1.0f / sqrtf((float)head_dim);
    float scores[(size_t)head_count * (size_t)sequence];

    for (int head = 0; head < head_count; ++head) {
        const int kv_head = head / groups_per_kv;
        const float *query = queries + (size_t)head * (size_t)head_dim;
        float *head_scores = scores + (size_t)head * (size_t)sequence;

        for (int token = 0; token < sequence; ++token) {
            const float *key = keys +
                ((size_t)token * (size_t)kv_heads + (size_t)kv_head) * (size_t)head_dim;
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            int dim = 0;

            for (; dim + 15 < head_dim; dim += 16) {
                acc0 = vmlaq_f32(acc0, vld1q_f32(key + dim), vld1q_f32(query + dim));
                acc1 = vmlaq_f32(acc1, vld1q_f32(key + dim + 4), vld1q_f32(query + dim + 4));
                acc2 = vmlaq_f32(acc2, vld1q_f32(key + dim + 8), vld1q_f32(query + dim + 8));
                acc3 = vmlaq_f32(acc3, vld1q_f32(key + dim + 12), vld1q_f32(query + dim + 12));
            }

            float dot = horizontal_sum(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
            for (; dim < head_dim; ++dim) {
                dot += key[dim] * query[dim];
            }
            head_scores[token] = dot * scale;
        }

        float maximum = head_scores[0];
        for (int token = 1; token < sequence; ++token) {
            if (head_scores[token] > maximum) {
                maximum = head_scores[token];
            }
        }

        float total = 0.0f;
        for (int token = 0; token < sequence; ++token) {
            const float probability = expf(head_scores[token] - maximum);
            head_scores[token] = probability;
            total += probability;
        }
        const float inverse_total = 1.0f / total;
        for (int token = 0; token < sequence; ++token) {
            head_scores[token] *= inverse_total;
        }

        float *head_output = output + (size_t)head * (size_t)head_dim;
        int dim = 0;
        for (; dim + 3 < head_dim; dim += 4) {
            float32x4_t accumulator = vdupq_n_f32(0.0f);
            for (int token = 0; token < sequence; ++token) {
                const float *value = values +
                    ((size_t)token * (size_t)kv_heads + (size_t)kv_head) * (size_t)head_dim + (size_t)dim;
                accumulator = vmlaq_n_f32(
                    accumulator, vld1q_f32(value), head_scores[token]
                );
            }
            vst1q_f32(head_output + dim, accumulator);
        }

        for (; dim < head_dim; ++dim) {
            float attended = 0.0f;
            for (int token = 0; token < sequence; ++token) {
                const float *value = values +
                    ((size_t)token * (size_t)kv_heads + (size_t)kv_head) * (size_t)head_dim + (size_t)dim;
                attended += head_scores[token] * value[0];
            }
            head_output[dim] = attended;
        }
    }
}
