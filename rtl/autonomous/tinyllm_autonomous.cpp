#include <cmath>
#include <cstdint>
#ifndef __SYNTHESIS__
#include <assert.h>
#endif
#include "model_layout.h"

#define MAX_GENERATED_TOKENS 128
#define IO_OUTPUT_TOKEN_BASE MODEL_MAX_SEQ_LEN
#define IO_RESULT_COUNT_INDEX (IO_OUTPUT_TOKEN_BASE + MAX_GENERATED_TOKENS)
#define IO_STOPPED_TOKEN_INDEX (IO_RESULT_COUNT_INDEX + 1)
#define IO_LOGITS_BASE (IO_STOPPED_TOKEN_INDEX + 1)
#define IO_TRACE_BASE (IO_LOGITS_BASE + MAX_GENERATED_TOKENS * MODEL_VOCAB_SIZE)
#define TRACE_RECORD_WORDS 12
#define TRACE_STAGE_COUNT 80
#define IO_WORDS (IO_TRACE_BASE + TRACE_RECORD_WORDS * TRACE_STAGE_COUNT)
#define VOCAB_STREAM_HALF_ROWS (MODEL_VOCAB_SIZE / 2)
#define KV_WORD_BASE (MODEL_BINARY_BYTES / 4)
#define KV_LAYER_WORDS (MODEL_MAX_SEQ_LEN * MODEL_KV_HEADS * (MODEL_DIM / MODEL_HEADS))
#define KV_BYTES (MODEL_LAYERS * KV_LAYER_WORDS * 2 * 4)
#define MODEL_WORKSPACE_BYTES (MODEL_BINARY_BYTES + KV_BYTES)
#define ROPE_TABLE_BYTES (MODEL_MAX_SEQ_LEN * (MODEL_DIM / MODEL_HEADS) * 4)
#define VOCAB_STREAM_HALF_BYTES (VOCAB_STREAM_HALF_ROWS * TENSOR_TOK_EMBEDDINGS_STREAM_ROW_WORDS * 8)

static const int HEAD_DIM = MODEL_DIM / MODEL_HEADS;
static const int GROUPS_PER_KV = MODEL_HEADS / MODEL_KV_HEADS;
static const float INV_SQRT_HEAD_DIM = 0.1767766952966369f;
static const float ROPE_RATIO = 0.5623413251903491f;
static inline uint32_t model_word(const uint64_t *memory, unsigned int word32_offset) {
#pragma HLS INLINE
    const uint64_t packed = memory[word32_offset >> 1];
    return word32_offset & 1u ? static_cast<uint32_t>(packed >> 32)
                              : static_cast<uint32_t>(packed);
}

static inline float model_float(const uint64_t *memory, unsigned int byte_offset) {
#pragma HLS INLINE
    union FloatBits {
        uint32_t bits;
        float value;
    } conversion;
    conversion.bits = model_word(memory, byte_offset >> 2);
    return conversion.value;
}

static inline int32_t float_word(float value) {
    union FloatBits {
        uint32_t bits;
        float value;
    } conversion;
    conversion.value = value;
    return static_cast<int32_t>(conversion.bits);
}

static inline uint32_t ordered_float_key(float value) {
#pragma HLS INLINE
    const uint32_t bits = static_cast<uint32_t>(float_word(value));
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

#if defined(TINYLLM_PERFORMANCE_BUILD)
static inline void trace_vector(int32_t *, int, int, const float *) {}
static inline void trace_vocab_pair(int32_t *, int, const float *, const float *) {}
#else
static void trace_vector(int32_t *io, int stage, int length, const float *values) {
    const unsigned int base = IO_TRACE_BASE + static_cast<unsigned int>(stage * TRACE_RECORD_WORDS);
    int finite_count = 0;
    int first_bad = -1;
    float minimum = 3.402823466e+38f;
    float maximum = -3.402823466e+38f;
    for (int i = 0; i < length; ++i) {
#pragma HLS PIPELINE II=1
        const float value = values[i];
        const bool finite = value == value && value < 3.402823466e+38f
            && value > -3.402823466e+38f;
        if (finite) {
            ++finite_count;
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
        } else if (first_bad < 0) {
            first_bad = i;
        }
        if (i < 4) io[base + 5u + static_cast<unsigned int>(i)] = float_word(value);
        if (i == length - 1) io[base + 9u] = float_word(value);
        if (stage == 1 && i < MODEL_DIM) io[i] = float_word(value);
    }
    io[base + 0u] = stage;
    io[base + 1u] = length;
    io[base + 2u] = finite_count;
    io[base + 3u] = float_word(minimum);
    io[base + 4u] = float_word(maximum);
    io[base + 10u] = first_bad;
    io[base + 11u] = 0;
}

static void trace_vocab_pair(int32_t *io, int stage,
                             const float *lower, const float *upper) {
    const unsigned int base = IO_TRACE_BASE + static_cast<unsigned int>(stage * TRACE_RECORD_WORDS);
    int finite_count = 0;
    int first_bad = -1;
    float minimum = 3.402823466e+38f;
    float maximum = -3.402823466e+38f;
    for (int i = 0; i < MODEL_VOCAB_SIZE; ++i) {
#pragma HLS PIPELINE II=1
        const float value = i < VOCAB_STREAM_HALF_ROWS ? lower[i]
            : upper[i - VOCAB_STREAM_HALF_ROWS];
        const bool finite = value == value && value < 3.402823466e+38f
            && value > -3.402823466e+38f;
        if (finite) {
            ++finite_count;
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
        } else if (first_bad < 0) {
            first_bad = i;
        }
        if (i < 4) io[base + 5u + static_cast<unsigned int>(i)] = float_word(value);
        if (i == MODEL_VOCAB_SIZE - 1) io[base + 9u] = float_word(value);
    }
    io[base + 0u] = stage;
    io[base + 1u] = MODEL_VOCAB_SIZE;
    io[base + 2u] = finite_count;
    io[base + 3u] = float_word(minimum);
    io[base + 4u] = float_word(maximum);
    io[base + 10u] = first_bad;
    io[base + 11u] = 0;
}
#endif

struct TensorDesc {
    unsigned int data_offset;
    unsigned int scale_offset;
    unsigned int stream_offset;
};

#define LAYER_TENSORS(name, tensor) \
static const TensorDesc name[MODEL_LAYERS] = { \
    {TENSOR_LAYERS_0_##tensor##_DATA_OFFSET, TENSOR_LAYERS_0_##tensor##_SCALE_OFFSET, TENSOR_LAYERS_0_##tensor##_STREAM_OFFSET}, \
    {TENSOR_LAYERS_1_##tensor##_DATA_OFFSET, TENSOR_LAYERS_1_##tensor##_SCALE_OFFSET, TENSOR_LAYERS_1_##tensor##_STREAM_OFFSET}, \
    {TENSOR_LAYERS_2_##tensor##_DATA_OFFSET, TENSOR_LAYERS_2_##tensor##_SCALE_OFFSET, TENSOR_LAYERS_2_##tensor##_STREAM_OFFSET}, \
    {TENSOR_LAYERS_3_##tensor##_DATA_OFFSET, TENSOR_LAYERS_3_##tensor##_SCALE_OFFSET, TENSOR_LAYERS_3_##tensor##_STREAM_OFFSET}, \
    {TENSOR_LAYERS_4_##tensor##_DATA_OFFSET, TENSOR_LAYERS_4_##tensor##_SCALE_OFFSET, TENSOR_LAYERS_4_##tensor##_STREAM_OFFSET} \
}

LAYER_TENSORS(kWq, ATTENTION_WQ);
LAYER_TENSORS(kWk, ATTENTION_WK);
LAYER_TENSORS(kWv, ATTENTION_WV);
LAYER_TENSORS(kWo, ATTENTION_WO);
LAYER_TENSORS(kW1, FEED_FORWARD_W1);
LAYER_TENSORS(kW2, FEED_FORWARD_W2);
LAYER_TENSORS(kW3, FEED_FORWARD_W3);

static const unsigned int kAttentionNorm[MODEL_LAYERS] = {
    TENSOR_LAYERS_0_ATTENTION_NORM_DATA_OFFSET,
    TENSOR_LAYERS_1_ATTENTION_NORM_DATA_OFFSET,
    TENSOR_LAYERS_2_ATTENTION_NORM_DATA_OFFSET,
    TENSOR_LAYERS_3_ATTENTION_NORM_DATA_OFFSET,
    TENSOR_LAYERS_4_ATTENTION_NORM_DATA_OFFSET,
};
static const unsigned int kFfnNorm[MODEL_LAYERS] = {
    TENSOR_LAYERS_0_FFN_NORM_DATA_OFFSET,
    TENSOR_LAYERS_1_FFN_NORM_DATA_OFFSET,
    TENSOR_LAYERS_2_FFN_NORM_DATA_OFFSET,
    TENSOR_LAYERS_3_FFN_NORM_DATA_OFFSET,
    TENSOR_LAYERS_4_FFN_NORM_DATA_OFFSET,
};
#if !defined(TINYLLM_PERFORMANCE_BUILD)
static uint32_t g_q_debug[16];
#endif

static void int8_row(const uint64_t *memory, unsigned int data_offset,
                     unsigned int scale_offset, int row, int columns,
                     const int16_t *vector, float activation_scale,
                     float *result, bool capture_debug) {
#pragma HLS INLINE
    const unsigned int row_offset = data_offset + static_cast<unsigned int>(row * columns);
    const unsigned int row_scale_offset = scale_offset + static_cast<unsigned int>(row) * 4u;
    const uint32_t row_scale_word = model_word(memory, row_scale_offset >> 2);
    union ScaleBits {
        uint32_t bits;
        float value;
    } scale_conversion;
    scale_conversion.bits = row_scale_word;
    const float scale = scale_conversion.value;
    int32_t partial0 = 0, partial1 = 0, partial2 = 0, partial3 = 0;
    int32_t partial4 = 0, partial5 = 0, partial6 = 0, partial7 = 0;
    uint64_t first_packed[2];
#pragma HLS ARRAY_PARTITION variable=first_packed complete dim=1
    for (int group = 0; group < columns / 8; ++group) {
#pragma HLS PIPELINE II=1
        const uint64_t packed = memory[(row_offset >> 3) + static_cast<unsigned int>(group)];
        const int q0 = static_cast<int8_t>(packed & 0xffu);
        const int q1 = static_cast<int8_t>((packed >> 8) & 0xffu);
        const int q2 = static_cast<int8_t>((packed >> 16) & 0xffu);
        const int q3 = static_cast<int8_t>((packed >> 24) & 0xffu);
        const int q4 = static_cast<int8_t>((packed >> 32) & 0xffu);
        const int q5 = static_cast<int8_t>((packed >> 40) & 0xffu);
        const int q6 = static_cast<int8_t>((packed >> 48) & 0xffu);
        const int q7 = static_cast<int8_t>((packed >> 56) & 0xffu);
        if (group < 2) first_packed[group] = packed;
        partial0 += q0 * static_cast<int32_t>(vector[group * 8 + 0]);
        partial1 += q1 * static_cast<int32_t>(vector[group * 8 + 1]);
        partial2 += q2 * static_cast<int32_t>(vector[group * 8 + 2]);
        partial3 += q3 * static_cast<int32_t>(vector[group * 8 + 3]);
        partial4 += q4 * static_cast<int32_t>(vector[group * 8 + 4]);
        partial5 += q5 * static_cast<int32_t>(vector[group * 8 + 5]);
        partial6 += q6 * static_cast<int32_t>(vector[group * 8 + 6]);
        partial7 += q7 * static_cast<int32_t>(vector[group * 8 + 7]);
    }
    const int32_t sum = (((partial0 + partial1) + (partial2 + partial3))
                         + ((partial4 + partial5) + (partial6 + partial7)));
    result[row] = static_cast<float>(sum) * activation_scale * scale;
#if !defined(TINYLLM_PERFORMANCE_BUILD)
    if (capture_debug && row == 0) {
        g_q_debug[0] = row_offset >> 2;
        g_q_debug[1] = row_scale_offset >> 2;
        g_q_debug[2] = static_cast<uint32_t>(first_packed[0]);
        g_q_debug[3] = static_cast<uint32_t>(first_packed[0] >> 32);
        for (int i = 0; i < 4; ++i) {
            const int32_t partial = i == 0 ? partial0
                : (i == 1 ? partial1 : (i == 2 ? partial2 : partial3));
            g_q_debug[4 + i] = static_cast<uint32_t>(float_word(
                static_cast<float>(vector[i]) * activation_scale));
            g_q_debug[10 + i] = static_cast<uint32_t>(float_word(
                static_cast<float>(partial) * activation_scale * scale));
        }
        g_q_debug[8] = row_scale_word;
        g_q_debug[9] = static_cast<uint32_t>(float_word(scale));
        g_q_debug[14] = static_cast<uint32_t>(float_word(result[row]));
        g_q_debug[15] = 0x51305230u;
    }
#endif
}

static void int8_matvec(const uint64_t *memory, unsigned int data_offset,
                        unsigned int scale_offset, int rows, int columns,
                        const int16_t *vector, float activation_scale,
                        float *result, bool capture_debug) {
#pragma HLS INLINE
    for (int row = 0; row < rows; ++row) {
        int8_row(memory, data_offset, scale_offset, row, columns, vector,
                 activation_scale, result, capture_debug);
    }
}

static void int8_matvec_stream(const uint64_t *stream, int rows, int columns,
                               const int16_t *vector, float activation_scale,
                               float *result) {
#pragma HLS INLINE off
    const int groups = columns / 8;
    const int words_per_row = groups + 1;
    int row = 0;
    int phase = 0;
    float scale = 1.0f;
    int32_t partial0 = 0, partial1 = 0, partial2 = 0, partial3 = 0;
    int32_t partial4 = 0, partial5 = 0, partial6 = 0, partial7 = 0;
    for (int word = 0; word < rows * words_per_row; ++word) {
#pragma HLS PIPELINE II=1
        const uint64_t packed = stream[word];
        if (phase == 0) {
            union ScaleBits {
                uint32_t bits;
                float value;
            } scale_conversion;
            scale_conversion.bits = static_cast<uint32_t>(packed);
            scale = scale_conversion.value;
            partial0 = partial1 = partial2 = partial3 = 0;
            partial4 = partial5 = partial6 = partial7 = 0;
        } else {
            const int group = phase - 1;
            const int q0 = static_cast<int8_t>(packed & 0xffu);
            const int q1 = static_cast<int8_t>((packed >> 8) & 0xffu);
            const int q2 = static_cast<int8_t>((packed >> 16) & 0xffu);
            const int q3 = static_cast<int8_t>((packed >> 24) & 0xffu);
            const int q4 = static_cast<int8_t>((packed >> 32) & 0xffu);
            const int q5 = static_cast<int8_t>((packed >> 40) & 0xffu);
            const int q6 = static_cast<int8_t>((packed >> 48) & 0xffu);
            const int q7 = static_cast<int8_t>((packed >> 56) & 0xffu);
            partial0 += q0 * static_cast<int32_t>(vector[group * 8 + 0]);
            partial1 += q1 * static_cast<int32_t>(vector[group * 8 + 1]);
            partial2 += q2 * static_cast<int32_t>(vector[group * 8 + 2]);
            partial3 += q3 * static_cast<int32_t>(vector[group * 8 + 3]);
            partial4 += q4 * static_cast<int32_t>(vector[group * 8 + 4]);
            partial5 += q5 * static_cast<int32_t>(vector[group * 8 + 5]);
            partial6 += q6 * static_cast<int32_t>(vector[group * 8 + 6]);
            partial7 += q7 * static_cast<int32_t>(vector[group * 8 + 7]);
            if (phase == groups) {
                const int32_t sum = (((partial0 + partial1) + (partial2 + partial3))
                    + ((partial4 + partial5) + (partial6 + partial7)));
                result[row] = static_cast<float>(sum) * activation_scale * scale;
                ++row;
            }
        }
        phase = phase == groups ? 0 : phase + 1;
    }
}

static void int8_matvec_stream_pair(const uint64_t *stream0, const uint64_t *stream1,
                                    int rows, int columns, const int16_t *vector0,
                                    const int16_t *vector1, float activation_scale,
                                    float *result0, float *result1) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    int8_matvec_stream(stream0, rows, columns, vector0, activation_scale, result0);
    int8_matvec_stream(stream1, rows, columns, vector1, activation_scale, result1);
}

#if defined(TINYLLM_PARALLEL_KV)
static void int8_matvec_stream_pair_kv(const uint64_t *key_stream,
                                       const uint64_t *value_stream,
                                       int rows, int columns,
                                       const int16_t *key_vector,
                                       const int16_t *value_vector,
                                       float activation_scale,
                                       float *key_result, float *value_result) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    int8_matvec_stream(key_stream, rows, columns, key_vector,
                       activation_scale, key_result);
    int8_matvec_stream(value_stream, rows, columns, value_vector,
                       activation_scale, value_result);
}
#endif

static void vocab_matvec_stream_half(const uint64_t *stream, const int16_t *vector,
                                     float activation_scale, float *result) {
#pragma HLS INLINE off
    const int groups = MODEL_DIM / 8;
    const int words_per_row = TENSOR_TOK_EMBEDDINGS_STREAM_ROW_WORDS;
    int row = 0;
    int phase = 0;
    float scale = 1.0f;
    int32_t partial0 = 0, partial1 = 0, partial2 = 0, partial3 = 0;
    int32_t partial4 = 0, partial5 = 0, partial6 = 0, partial7 = 0;
    for (int word = 0; word < VOCAB_STREAM_HALF_ROWS * words_per_row; ++word) {
#pragma HLS PIPELINE II=1
        const uint64_t packed = stream[word];
        if (phase == 0) {
            union ScaleBits {
                uint32_t bits;
                float value;
            } scale_conversion;
            scale_conversion.bits = static_cast<uint32_t>(packed);
            scale = scale_conversion.value;
            partial0 = partial1 = partial2 = partial3 = 0;
            partial4 = partial5 = partial6 = partial7 = 0;
        } else {
            const int group = phase - 1;
            const int q0 = static_cast<int8_t>(packed & 0xffu);
            const int q1 = static_cast<int8_t>((packed >> 8) & 0xffu);
            const int q2 = static_cast<int8_t>((packed >> 16) & 0xffu);
            const int q3 = static_cast<int8_t>((packed >> 24) & 0xffu);
            const int q4 = static_cast<int8_t>((packed >> 32) & 0xffu);
            const int q5 = static_cast<int8_t>((packed >> 40) & 0xffu);
            const int q6 = static_cast<int8_t>((packed >> 48) & 0xffu);
            const int q7 = static_cast<int8_t>((packed >> 56) & 0xffu);
            partial0 += q0 * static_cast<int32_t>(vector[group * 8 + 0]);
            partial1 += q1 * static_cast<int32_t>(vector[group * 8 + 1]);
            partial2 += q2 * static_cast<int32_t>(vector[group * 8 + 2]);
            partial3 += q3 * static_cast<int32_t>(vector[group * 8 + 3]);
            partial4 += q4 * static_cast<int32_t>(vector[group * 8 + 4]);
            partial5 += q5 * static_cast<int32_t>(vector[group * 8 + 5]);
            partial6 += q6 * static_cast<int32_t>(vector[group * 8 + 6]);
            partial7 += q7 * static_cast<int32_t>(vector[group * 8 + 7]);
            if (phase == groups) {
                const int32_t sum = (((partial0 + partial1) + (partial2 + partial3))
                    + ((partial4 + partial5) + (partial6 + partial7)));
                result[row] = static_cast<float>(sum) * activation_scale * scale;
                ++row;
            }
        }
        phase = phase == groups ? 0 : phase + 1;
    }
}

static void vocab_matvec_stream_dual(const uint64_t *memory, const uint64_t *upper_stream,
                                     const int16_t *vector, float activation_scale,
                                     float *lower, float *upper) {
#pragma HLS INLINE off
    const uint64_t *lower_stream = memory + (TENSOR_TOK_EMBEDDINGS_STREAM_OFFSET >> 3);
    const int groups = MODEL_DIM / 8;
    const int words_per_row = TENSOR_TOK_EMBEDDINGS_STREAM_ROW_WORDS;
    int row = 0;
    int phase = 0;
    float lower_scale = 1.0f;
    float upper_scale = 1.0f;
    int32_t lower_partial0 = 0, lower_partial1 = 0;
    int32_t lower_partial2 = 0, lower_partial3 = 0;
    int32_t lower_partial4 = 0, lower_partial5 = 0;
    int32_t lower_partial6 = 0, lower_partial7 = 0;
    int32_t upper_partial0 = 0, upper_partial1 = 0;
    int32_t upper_partial2 = 0, upper_partial3 = 0;
    int32_t upper_partial4 = 0, upper_partial5 = 0;
    int32_t upper_partial6 = 0, upper_partial7 = 0;
    for (int word = 0; word < VOCAB_STREAM_HALF_ROWS * words_per_row; ++word) {
#pragma HLS PIPELINE II=1
        const uint64_t lower_packed = lower_stream[word];
        const uint64_t upper_packed = upper_stream[word];
        if (phase == 0) {
            union ScaleBits {
                uint32_t bits;
                float value;
            } lower_scale_conversion, upper_scale_conversion;
            lower_scale_conversion.bits = static_cast<uint32_t>(lower_packed);
            upper_scale_conversion.bits = static_cast<uint32_t>(upper_packed);
            lower_scale = lower_scale_conversion.value;
            upper_scale = upper_scale_conversion.value;
            lower_partial0 = lower_partial1 = lower_partial2 = lower_partial3 = 0;
            lower_partial4 = lower_partial5 = lower_partial6 = lower_partial7 = 0;
            upper_partial0 = upper_partial1 = upper_partial2 = upper_partial3 = 0;
            upper_partial4 = upper_partial5 = upper_partial6 = upper_partial7 = 0;
        } else {
            const int group = phase - 1;
            const int lower_q0 = static_cast<int8_t>(lower_packed & 0xffu);
            const int lower_q1 = static_cast<int8_t>((lower_packed >> 8) & 0xffu);
            const int lower_q2 = static_cast<int8_t>((lower_packed >> 16) & 0xffu);
            const int lower_q3 = static_cast<int8_t>((lower_packed >> 24) & 0xffu);
            const int lower_q4 = static_cast<int8_t>((lower_packed >> 32) & 0xffu);
            const int lower_q5 = static_cast<int8_t>((lower_packed >> 40) & 0xffu);
            const int lower_q6 = static_cast<int8_t>((lower_packed >> 48) & 0xffu);
            const int lower_q7 = static_cast<int8_t>((lower_packed >> 56) & 0xffu);
            const int upper_q0 = static_cast<int8_t>(upper_packed & 0xffu);
            const int upper_q1 = static_cast<int8_t>((upper_packed >> 8) & 0xffu);
            const int upper_q2 = static_cast<int8_t>((upper_packed >> 16) & 0xffu);
            const int upper_q3 = static_cast<int8_t>((upper_packed >> 24) & 0xffu);
            const int upper_q4 = static_cast<int8_t>((upper_packed >> 32) & 0xffu);
            const int upper_q5 = static_cast<int8_t>((upper_packed >> 40) & 0xffu);
            const int upper_q6 = static_cast<int8_t>((upper_packed >> 48) & 0xffu);
            const int upper_q7 = static_cast<int8_t>((upper_packed >> 56) & 0xffu);
            lower_partial0 += lower_q0 * static_cast<int32_t>(vector[group * 8 + 0]);
            lower_partial1 += lower_q1 * static_cast<int32_t>(vector[group * 8 + 1]);
            lower_partial2 += lower_q2 * static_cast<int32_t>(vector[group * 8 + 2]);
            lower_partial3 += lower_q3 * static_cast<int32_t>(vector[group * 8 + 3]);
            lower_partial4 += lower_q4 * static_cast<int32_t>(vector[group * 8 + 4]);
            lower_partial5 += lower_q5 * static_cast<int32_t>(vector[group * 8 + 5]);
            lower_partial6 += lower_q6 * static_cast<int32_t>(vector[group * 8 + 6]);
            lower_partial7 += lower_q7 * static_cast<int32_t>(vector[group * 8 + 7]);
            upper_partial0 += upper_q0 * static_cast<int32_t>(vector[group * 8 + 0]);
            upper_partial1 += upper_q1 * static_cast<int32_t>(vector[group * 8 + 1]);
            upper_partial2 += upper_q2 * static_cast<int32_t>(vector[group * 8 + 2]);
            upper_partial3 += upper_q3 * static_cast<int32_t>(vector[group * 8 + 3]);
            upper_partial4 += upper_q4 * static_cast<int32_t>(vector[group * 8 + 4]);
            upper_partial5 += upper_q5 * static_cast<int32_t>(vector[group * 8 + 5]);
            upper_partial6 += upper_q6 * static_cast<int32_t>(vector[group * 8 + 6]);
            upper_partial7 += upper_q7 * static_cast<int32_t>(vector[group * 8 + 7]);
            if (phase == groups) {
                const int32_t lower_sum =
                    (((lower_partial0 + lower_partial1) + (lower_partial2 + lower_partial3))
                    + ((lower_partial4 + lower_partial5) + (lower_partial6 + lower_partial7)));
                const int32_t upper_sum =
                    (((upper_partial0 + upper_partial1) + (upper_partial2 + upper_partial3))
                    + ((upper_partial4 + upper_partial5) + (upper_partial6 + upper_partial7)));
                lower[row] = static_cast<float>(lower_sum) * activation_scale * lower_scale;
                upper[row] = static_cast<float>(upper_sum) * activation_scale * upper_scale;
                ++row;
            }
        }
        phase = phase == groups ? 0 : phase + 1;
    }
}

static int64_t q24_round(float value) {
#pragma HLS INLINE
    const float scaled = value * 16777216.0f;
    return static_cast<int64_t>(scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
}

static inline bool magnitude_bits_greater(uint32_t a, uint32_t b) {
#pragma HLS INLINE
    return a > b;
}

static void quantize_activation(const float *input, int16_t *output,
                                int length, float *activation_scale) {
#pragma HLS INLINE
    uint32_t maximum_lanes[4];
#pragma HLS ARRAY_PARTITION variable=maximum_lanes complete dim=1
    for (int lane = 0; lane < 4; ++lane) {
        maximum_lanes[lane] = 0;
    }
    for (int i = 0; i < length; ++i) {
#pragma HLS PIPELINE II=1
        const uint32_t magnitude_bits =
            static_cast<uint32_t>(float_word(input[i])) & 0x7fffffffu;
        switch (i & 3) {
            case 0: if (magnitude_bits_greater(magnitude_bits, maximum_lanes[0])) maximum_lanes[0] = magnitude_bits; break;
            case 1: if (magnitude_bits_greater(magnitude_bits, maximum_lanes[1])) maximum_lanes[1] = magnitude_bits; break;
            case 2: if (magnitude_bits_greater(magnitude_bits, maximum_lanes[2])) maximum_lanes[2] = magnitude_bits; break;
            default: if (magnitude_bits_greater(magnitude_bits, maximum_lanes[3])) maximum_lanes[3] = magnitude_bits; break;
        }
    }
    uint32_t maximum_bits = maximum_lanes[0];
    for (int lane = 1; lane < 4; ++lane) {
        if (magnitude_bits_greater(maximum_lanes[lane], maximum_bits)) {
            maximum_bits = maximum_lanes[lane];
        }
    }
    if (maximum_bits == 0) {
        *activation_scale = 1.0f;
        for (int i = 0; i < length; ++i) {
#pragma HLS PIPELINE II=1
            output[i] = 0;
        }
        return;
    }
    union FloatBits { uint32_t bits; float value; } maximum_value;
    maximum_value.bits = maximum_bits;
    const float maximum = maximum_value.value;
    *activation_scale = maximum / 32767.0f;
    const float quant_multiplier = 32767.0f / maximum;
    for (int i = 0; i < length; ++i) {
#pragma HLS PIPELINE II=1
        const float scaled = input[i] * quant_multiplier;
        int32_t rounded = static_cast<int32_t>(
            scaled >= 0.0f ? scaled + 0.5f : scaled - 0.5f);
        if (rounded > 32767) rounded = 32767;
        if (rounded < -32767) rounded = -32767;
        output[i] = static_cast<int16_t>(rounded);
    }
}

static void embedding(const uint64_t *memory, int token, float *result) {
#pragma HLS INLINE
    const unsigned int row_offset = TENSOR_TOK_EMBEDDINGS_DATA_OFFSET
        + static_cast<unsigned int>(token * MODEL_DIM);
    const float scale = model_float(memory, TENSOR_TOK_EMBEDDINGS_SCALE_OFFSET
        + static_cast<unsigned int>(token) * 4u);
    for (int group = 0; group < MODEL_DIM / 8; ++group) {
#pragma HLS PIPELINE II=1
        const uint64_t packed = memory[(row_offset >> 3) + static_cast<unsigned int>(group)];
        result[group * 8 + 0] = static_cast<float>(static_cast<int8_t>(packed & 0xffu)) * scale;
        result[group * 8 + 1] = static_cast<float>(static_cast<int8_t>((packed >> 8) & 0xffu)) * scale;
        result[group * 8 + 2] = static_cast<float>(static_cast<int8_t>((packed >> 16) & 0xffu)) * scale;
        result[group * 8 + 3] = static_cast<float>(static_cast<int8_t>((packed >> 24) & 0xffu)) * scale;
        result[group * 8 + 4] = static_cast<float>(static_cast<int8_t>((packed >> 32) & 0xffu)) * scale;
        result[group * 8 + 5] = static_cast<float>(static_cast<int8_t>((packed >> 40) & 0xffu)) * scale;
        result[group * 8 + 6] = static_cast<float>(static_cast<int8_t>((packed >> 48) & 0xffu)) * scale;
        result[group * 8 + 7] = static_cast<float>(static_cast<int8_t>((packed >> 56) & 0xffu)) * scale;
    }
}

static void rms_norm(const uint64_t *memory, unsigned int norm_offset,
                     const float *input, float *output) {
#pragma HLS INLINE
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (int i = 0; i < MODEL_DIM; ++i) {
#pragma HLS PIPELINE II=1
        const float value = input[i];
        switch (i & 3) {
            case 0: sum0 += value * value; break;
            case 1: sum1 += value * value; break;
            case 2: sum2 += value * value; break;
            default: sum3 += value * value; break;
        }
    }
    const float inverse_rms = 1.0f / sqrtf(((sum0 + sum1) + (sum2 + sum3)) / MODEL_DIM + MODEL_NORM_EPS);
    for (int i = 0; i < MODEL_DIM; ++i) {
#pragma HLS PIPELINE II=1
        output[i] = input[i] * inverse_rms * model_float(memory, norm_offset + static_cast<unsigned int>(i) * 4u);
    }
}

static inline unsigned int kv_index(int layer, int position, int head, int dimension) {
    return KV_WORD_BASE + static_cast<unsigned int>(layer * KV_LAYER_WORDS
        + (position * MODEL_KV_HEADS + head) * HEAD_DIM + dimension);
}

static inline unsigned int kv_mirror_index(int layer, int position, int head, int dimension) {
#pragma HLS INLINE
    return static_cast<unsigned int>(layer * KV_LAYER_WORDS
        + (position * MODEL_KV_HEADS + head) * HEAD_DIM + dimension);
}

static void attention_score_half(const uint64_t *storage, const float *query,
                                 int layer, int position, int parity, bool mirror_storage,
                                 float scores[MODEL_HEADS][MODEL_MAX_SEQ_LEN / 2]) {
#pragma HLS INLINE off
    for (int head = 0; head < MODEL_HEADS; ++head) {
        const int kv_head = head / GROUPS_PER_KV;
        for (int t = parity; t <= position; t += 2) {
            int64_t score_accumulator = 0;
            for (int d = 0; d < HEAD_DIM; d += 2) {
#pragma HLS PIPELINE II=1
                union FloatPair { uint64_t bits; float value[2]; } key_value;
                const unsigned int index = mirror_storage
                    ? kv_mirror_index(layer, t, kv_head, d)
                    : kv_index(layer, t, kv_head, d);
                key_value.bits = storage[index >> 1];
                const float product0 = query[head * HEAD_DIM + d] * key_value.value[0];
                const float product1 = query[head * HEAD_DIM + d + 1] * key_value.value[1];
                score_accumulator += q24_round(product0) + q24_round(product1);
            }
            scores[head][t >> 1] = static_cast<float>(score_accumulator)
                * (INV_SQRT_HEAD_DIM / 16777216.0f);
        }
    }
}

static void attention_score_dual(const uint64_t *memory, const uint64_t *kv_mirror,
                                 const float *query, int layer, int position,
                                 float scores_even[MODEL_HEADS][MODEL_MAX_SEQ_LEN / 2],
                                 float scores_odd[MODEL_HEADS][MODEL_MAX_SEQ_LEN / 2]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    float query_even[MODEL_DIM];
    float query_odd[MODEL_DIM];
    for (int i = 0; i < MODEL_DIM; ++i) {
#pragma HLS PIPELINE II=1
        query_even[i] = query[i];
        query_odd[i] = query[i];
    }
    attention_score_half(memory, query_even, layer, position, 0, false, scores_even);
    attention_score_half(kv_mirror, query_odd, layer, position, 1, true, scores_odd);
}

static void attention_value_half(const uint64_t *storage,
                                 const float weights[MODEL_HEADS][MODEL_MAX_SEQ_LEN / 2],
                                 int layer, int position, int parity, bool mirror_storage,
                                 int64_t output[MODEL_DIM]) {
#pragma HLS INLINE off
    for (int head = 0; head < MODEL_HEADS; ++head) {
        const int kv_head = head / GROUPS_PER_KV;
        int32_t accumulator_sum[HEAD_DIM];
        int32_t accumulator_carry[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE II=1
            accumulator_sum[d] = 0;
            accumulator_carry[d] = 0;
        }
        for (int t = parity; t <= position; t += 2) {
            const float score = weights[head][t >> 1];
            const unsigned int value_row = (mirror_storage
                ? kv_mirror_index(layer, t, kv_head, 0)
                : kv_index(layer, t, kv_head, 0))
                + MODEL_LAYERS * KV_LAYER_WORDS;
            for (int d = 0; d < HEAD_DIM; d += 2) {
#pragma HLS PIPELINE II=1
                union FloatPair { uint64_t bits; float value[2]; } val_value;
                val_value.bits = storage[(value_row + static_cast<unsigned int>(d)) >> 1];
                const int64_t contribution0 = q24_round(score * val_value.value[0]);
                const int64_t contribution1 = q24_round(score * val_value.value[1]);
#ifndef __SYNTHESIS__
                const int32_t accumulated0 = static_cast<int32_t>(
                    static_cast<uint32_t>(accumulator_sum[d])
                    + static_cast<uint32_t>(accumulator_carry[d]));
                const int32_t accumulated1 = static_cast<int32_t>(
                    static_cast<uint32_t>(accumulator_sum[d + 1])
                    + static_cast<uint32_t>(accumulator_carry[d + 1]));
                const int64_t next0 = accumulated0 + contribution0;
                const int64_t next1 = accumulated1 + contribution1;
                assert(contribution0 >= -2147483648LL && contribution0 <= 2147483647LL);
                assert(contribution1 >= -2147483648LL && contribution1 <= 2147483647LL);
                assert(next0 >= -2147483648LL && next0 <= 2147483647LL);
                assert(next1 >= -2147483648LL && next1 <= 2147483647LL);
#endif
                const uint32_t a0 = static_cast<uint32_t>(accumulator_sum[d]);
                const uint32_t b0 = static_cast<uint32_t>(accumulator_carry[d]);
                const uint32_t x0 = static_cast<uint32_t>(contribution0);
                const uint32_t a1 = static_cast<uint32_t>(accumulator_sum[d + 1]);
                const uint32_t b1 = static_cast<uint32_t>(accumulator_carry[d + 1]);
                const uint32_t x1 = static_cast<uint32_t>(contribution1);
                accumulator_sum[d] = static_cast<int32_t>(a0 ^ b0 ^ x0);
                accumulator_carry[d] = static_cast<int32_t>(
                    ((a0 & b0) | (a0 & x0) | (b0 & x0)) << 1);
                accumulator_sum[d + 1] = static_cast<int32_t>(a1 ^ b1 ^ x1);
                accumulator_carry[d + 1] = static_cast<int32_t>(
                    ((a1 & b1) | (a1 & x1) | (b1 & x1)) << 1);
            }
        }
        for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE II=1
            const uint32_t total = static_cast<uint32_t>(accumulator_sum[d])
                + static_cast<uint32_t>(accumulator_carry[d]);
            output[head * HEAD_DIM + d] = static_cast<int64_t>(static_cast<int32_t>(total));
        }
    }
}

static void attention_value_dual(const uint64_t *memory, const uint64_t *kv_mirror,
                                 const float weights_even[MODEL_HEADS][MODEL_MAX_SEQ_LEN / 2],
                                 const float weights_odd[MODEL_HEADS][MODEL_MAX_SEQ_LEN / 2],
                                 int layer, int position, int64_t even[MODEL_DIM],
                                 int64_t odd[MODEL_DIM]) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    attention_value_half(memory, weights_even, layer, position, 0, false, even);
    attention_value_half(kv_mirror, weights_odd, layer, position, 1, true, odd);
}

static void attention_dual(const uint64_t *memory, const uint64_t *kv_mirror,
                           int layer, int position, const float *query, float *attended) {
#pragma HLS INLINE off
    float scores_even[MODEL_HEADS][MODEL_MAX_SEQ_LEN / 2];
    float scores_odd[MODEL_HEADS][MODEL_MAX_SEQ_LEN / 2];
    attention_score_dual(memory, kv_mirror, query, layer, position, scores_even, scores_odd);

    float weights_even[MODEL_HEADS][MODEL_MAX_SEQ_LEN / 2];
    float weights_odd[MODEL_HEADS][MODEL_MAX_SEQ_LEN / 2];
    for (int head = 0; head < MODEL_HEADS; ++head) {
        float scores[MODEL_MAX_SEQ_LEN];
        for (int t = 0; t <= position; ++t) {
#pragma HLS PIPELINE II=1
            scores[t] = (t & 1) ? scores_odd[head][t >> 1] : scores_even[head][t >> 1];
        }
        float maximum = -3.402823466e+38f;
        uint32_t maximum_key = 0x00800000u;
        for (int t = 0; t <= position; ++t) {
#pragma HLS PIPELINE II=1
            const uint32_t score_key = ordered_float_key(scores[t]);
            if (score_key > maximum_key) {
                maximum_key = score_key;
                maximum = scores[t];
            }
        }
        int64_t denominator_accumulator = 0;
        for (int t = 0; t <= position; ++t) {
#pragma HLS PIPELINE II=1
            scores[t] = expf(scores[t] - maximum);
            denominator_accumulator += q24_round(scores[t]);
        }
        const float denominator = static_cast<float>(denominator_accumulator)
            / 16777216.0f;
        for (int t = 0; t <= position; ++t) {
#pragma HLS PIPELINE II=1
            const float weight = scores[t] / denominator;
            if (t & 1) weights_odd[head][t >> 1] = weight;
            else weights_even[head][t >> 1] = weight;
        }
    }

    int64_t attended_even[MODEL_DIM];
    int64_t attended_odd[MODEL_DIM];
    attention_value_dual(memory, kv_mirror, weights_even, weights_odd,
                         layer, position, attended_even, attended_odd);
    for (int i = 0; i < MODEL_DIM; ++i) {
#pragma HLS PIPELINE II=1
        attended[i] = static_cast<float>(attended_even[i] + attended_odd[i])
            / 16777216.0f;
    }
}

static void model_step(uint64_t *memory, int32_t *trace_io, const uint64_t *vocab_stream,
#if defined(TINYLLM_LEGACY_KV_MIRROR)
                       uint64_t *kv_mirror,
#else
                       const uint64_t *kv_mirror,
#endif
#if defined(TINYLLM_PARALLEL_KV)
                       const uint64_t *kv_memory,
#endif
                       const uint64_t *w3_memory, int token, int position,
                       float *logits_lower, float *logits_upper, bool trace_enabled) {
    float x[MODEL_DIM];
    float normalized[MODEL_DIM];
    float q[MODEL_DIM];
    float k[MODEL_KV_HEADS * HEAD_DIM];
    float v[MODEL_KV_HEADS * HEAD_DIM];
    float attended[MODEL_DIM];
    float gate[352];
    float up[352];
    float ffn_input[352];
    float ffn_output[MODEL_DIM];
    int16_t normalized_q[MODEL_DIM];
#if defined(TINYLLM_PARALLEL_FFN)
    int16_t normalized_q_w3[MODEL_DIM];
#pragma HLS ARRAY_PARTITION variable=normalized_q_w3 cyclic factor=4 dim=1
#endif
#if defined(TINYLLM_PARALLEL_KV)
    int16_t normalized_q_k[MODEL_DIM];
    int16_t normalized_q_v[MODEL_DIM];
#pragma HLS ARRAY_PARTITION variable=normalized_q_k cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=normalized_q_v cyclic factor=4 dim=1
#endif
    int16_t attended_q[MODEL_DIM];
    int16_t ffn_input_q[352];
    float normalized_scale = 1.0f;
    float attended_scale = 1.0f;
    float ffn_input_scale = 1.0f;
#pragma HLS ARRAY_PARTITION variable=x cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=normalized cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=q cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=k cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=v cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=attended cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=gate cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=up cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=ffn_input cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=ffn_output cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=normalized_q cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=attended_q cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=ffn_input_q cyclic factor=4 dim=1

    for (int i = 0; i < MODEL_DIM; ++i) {
#pragma HLS PIPELINE II=1
        x[i] = 0.0f;
    }
    embedding(memory, token, x);
    int trace_stage = 0;
    if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, x);

    float cosines[HEAD_DIM / 2];
    float sines[HEAD_DIM / 2];
#if defined(TINYLLM_PRECOMPUTED_ROPE)
    const uint64_t *rope_table = memory + MODEL_WORKSPACE_BYTES / 8;
    for (int pair = 0; pair < HEAD_DIM / 2; ++pair) {
#pragma HLS PIPELINE II=1
        union FloatPair { uint64_t bits; float value[2]; } values;
        values.bits = rope_table[position * (HEAD_DIM / 2) + pair];
        cosines[pair] = values.value[0];
        sines[pair] = values.value[1];
    }
#else
    float frequency = 1.0f;
    for (int pair = 0; pair < HEAD_DIM / 2; ++pair) {
#pragma HLS PIPELINE II=1
        const float angle = static_cast<float>(position) * frequency;
        cosines[pair] = cosf(angle);
        sines[pair] = sinf(angle);
        frequency *= ROPE_RATIO;
    }
#endif

    for (int layer = 0; layer < MODEL_LAYERS; ++layer) {
        rms_norm(memory, kAttentionNorm[layer], x, normalized);
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, normalized);
        quantize_activation(normalized, normalized_q, MODEL_DIM, &normalized_scale);
        if (trace_enabled && layer == 0) {
            int8_matvec(memory, kWq[layer].data_offset, kWq[layer].scale_offset,
                MODEL_DIM, MODEL_DIM, normalized_q, normalized_scale, q, true);
        } else {
            int8_matvec_stream(memory + (kWq[layer].stream_offset >> 3),
                MODEL_DIM, MODEL_DIM, normalized_q, normalized_scale, q);
        }
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, q);
#if defined(TINYLLM_PARALLEL_KV)
        for (int i = 0; i < MODEL_DIM; ++i) {
#pragma HLS PIPELINE II=1
            normalized_q_k[i] = normalized_q[i];
            normalized_q_v[i] = normalized_q[i];
        }
        int8_matvec_stream_pair_kv(
            kv_memory + (kWk[layer].stream_offset >> 3),
            w3_memory + (kWv[layer].stream_offset >> 3),
            MODEL_KV_HEADS * HEAD_DIM, MODEL_DIM, normalized_q_k,
            normalized_q_v, normalized_scale, k, v);
#else
        int8_matvec_stream(memory + (kWk[layer].stream_offset >> 3),
            MODEL_KV_HEADS * HEAD_DIM, MODEL_DIM, normalized_q, normalized_scale, k);
        int8_matvec_stream(memory + (kWv[layer].stream_offset >> 3),
            MODEL_KV_HEADS * HEAD_DIM, MODEL_DIM, normalized_q, normalized_scale, v);
#endif
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_KV_HEADS * HEAD_DIM, k);
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_KV_HEADS * HEAD_DIM, v);

        for (int head = 0; head < MODEL_HEADS; ++head) {
            for (int pair = 0; pair < HEAD_DIM / 2; ++pair) {
                const int index = head * HEAD_DIM + pair * 2;
                const float real = q[index];
                const float imag = q[index + 1];
                q[index] = real * cosines[pair] - imag * sines[pair];
                q[index + 1] = real * sines[pair] + imag * cosines[pair];
            }
        }
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, q);
        for (int head = 0; head < MODEL_KV_HEADS; ++head) {
            for (int pair = 0; pair < HEAD_DIM / 2; ++pair) {
                const int index = head * HEAD_DIM + pair * 2;
                const float real = k[index];
                const float imag = k[index + 1];
                k[index] = real * cosines[pair] - imag * sines[pair];
                k[index + 1] = real * sines[pair] + imag * cosines[pair];
            }
        }
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_KV_HEADS * HEAD_DIM, k);
        for (int head = 0; head < MODEL_KV_HEADS; ++head) {
            for (int d = 0; d < HEAD_DIM; d += 2) {
                const unsigned int index = kv_index(layer, position, head, d);
                const uint32_t key0 = static_cast<uint32_t>(float_word(k[head * HEAD_DIM + d]));
                const uint32_t key1 = static_cast<uint32_t>(float_word(k[head * HEAD_DIM + d + 1]));
                const uint32_t value0 = static_cast<uint32_t>(float_word(v[head * HEAD_DIM + d]));
                const uint32_t value1 = static_cast<uint32_t>(float_word(v[head * HEAD_DIM + d + 1]));
                memory[index >> 1] = static_cast<uint64_t>(key0)
                    | (static_cast<uint64_t>(key1) << 32);
                memory[(index + MODEL_LAYERS * KV_LAYER_WORDS) >> 1] =
                    static_cast<uint64_t>(value0) | (static_cast<uint64_t>(value1) << 32);
#if defined(TINYLLM_LEGACY_KV_MIRROR)
                const unsigned int mirror_index = kv_mirror_index(layer, position, head, d);
                kv_mirror[mirror_index >> 1] = static_cast<uint64_t>(key0)
                    | (static_cast<uint64_t>(key1) << 32);
                kv_mirror[(mirror_index + MODEL_LAYERS * KV_LAYER_WORDS) >> 1] =
                    static_cast<uint64_t>(value0) | (static_cast<uint64_t>(value1) << 32);
#endif
            }
        }

        attention_dual(memory, kv_mirror, layer, position, q, attended);
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, attended);

        quantize_activation(attended, attended_q, MODEL_DIM, &attended_scale);
        int8_matvec_stream(memory + (kWo[layer].stream_offset >> 3),
            MODEL_DIM, MODEL_DIM, attended_q, attended_scale, ffn_output);
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, ffn_output);
        for (int i = 0; i < MODEL_DIM; ++i) x[i] += ffn_output[i];
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, x);

        rms_norm(memory, kFfnNorm[layer], x, normalized);
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, normalized);
        quantize_activation(normalized, normalized_q, MODEL_DIM, &normalized_scale);
#if defined(TINYLLM_PARALLEL_FFN)
        if (!trace_enabled) {
            for (int i = 0; i < MODEL_DIM; ++i) {
#pragma HLS PIPELINE II=1
                normalized_q_w3[i] = normalized_q[i];
            }
            int8_matvec_stream_pair(memory + (kW1[layer].stream_offset >> 3),
                w3_memory + (kW3[layer].stream_offset >> 3), 352, MODEL_DIM,
                normalized_q, normalized_q_w3, normalized_scale, gate, up);
        } else {
            int8_matvec_stream(memory + (kW1[layer].stream_offset >> 3),
                352, MODEL_DIM, normalized_q, normalized_scale, gate);
            int8_matvec_stream(memory + (kW3[layer].stream_offset >> 3),
                352, MODEL_DIM, normalized_q, normalized_scale, up);
        }
#else
        int8_matvec_stream(memory + (kW1[layer].stream_offset >> 3),
            352, MODEL_DIM, normalized_q, normalized_scale, gate);
        int8_matvec_stream(memory + (kW3[layer].stream_offset >> 3),
            352, MODEL_DIM, normalized_q, normalized_scale, up);
#endif
        if (trace_enabled) trace_vector(trace_io, trace_stage++, 352, gate);
        if (trace_enabled) trace_vector(trace_io, trace_stage++, 352, up);
        for (int i = 0; i < 352; ++i) {
#pragma HLS PIPELINE II=1
            ffn_input[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
        }
        if (trace_enabled) trace_vector(trace_io, trace_stage++, 352, ffn_input);
        quantize_activation(ffn_input, ffn_input_q, 352, &ffn_input_scale);
        int8_matvec_stream(memory + (kW2[layer].stream_offset >> 3),
            MODEL_DIM, 352, ffn_input_q, ffn_input_scale, ffn_output);
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, ffn_output);
        for (int i = 0; i < MODEL_DIM; ++i) x[i] += ffn_output[i];
        if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, x);
    }

    rms_norm(memory, TENSOR_NORM_DATA_OFFSET, x, normalized);
    if (trace_enabled) trace_vector(trace_io, trace_stage++, MODEL_DIM, normalized);
    quantize_activation(normalized, normalized_q, MODEL_DIM, &normalized_scale);
    vocab_matvec_stream_dual(memory, vocab_stream, normalized_q, normalized_scale,
                             logits_lower, logits_upper);
    if (trace_enabled) trace_vocab_pair(trace_io, trace_stage++, logits_lower, logits_upper);
}

void tinyllm_autonomous(uint64_t *workspace, int32_t *io,
                        int prompt_length, int max_new_tokens, int min_new_tokens,
                        int stop_token0, int stop_token1, int dump_logits,
#if defined(TINYLLM_LEGACY_KV_MIRROR)
#if defined(TINYLLM_PARALLEL_FFN)
                        uint64_t *auxiliary, const uint64_t *w3_workspace) {
#else
                        uint64_t *auxiliary) {
#endif
#else
                        const uint64_t *auxiliary,
                        const uint64_t *kv_cache_alias) {
#endif
#pragma HLS INTERFACE m_axi port=workspace offset=slave bundle=gmem0 depth=600000 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=64 num_write_outstanding=4
#pragma HLS INTERFACE m_axi port=io offset=slave bundle=gmem1 depth=595522 max_read_burst_length=16 max_write_burst_length=16
#if defined(TINYLLM_LEGACY_KV_MIRROR)
#if defined(TINYLLM_PARALLEL_KV)
#pragma HLS INTERFACE m_axi port=auxiliary offset=slave bundle=gmem2 depth=600000 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=8 num_write_outstanding=1
#else
#pragma HLS INTERFACE m_axi port=auxiliary offset=slave bundle=gmem2 depth=198656 max_read_burst_length=64 num_read_outstanding=16 max_write_burst_length=8 num_write_outstanding=1
#endif
#if defined(TINYLLM_PARALLEL_FFN)
#pragma HLS INTERFACE m_axi port=w3_workspace offset=slave bundle=gmem3 depth=600000 max_read_burst_length=64 num_read_outstanding=16
#endif
#else
#pragma HLS INTERFACE m_axi port=auxiliary offset=slave bundle=gmem2 depth=34816 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=kv_cache_alias offset=slave bundle=gmem3 depth=163840 max_read_burst_length=64
#endif
#pragma HLS INTERFACE s_axilite port=workspace bundle=control
#pragma HLS INTERFACE s_axilite port=io bundle=control
#pragma HLS INTERFACE s_axilite port=prompt_length bundle=control
#pragma HLS INTERFACE s_axilite port=max_new_tokens bundle=control
#pragma HLS INTERFACE s_axilite port=min_new_tokens bundle=control
#pragma HLS INTERFACE s_axilite port=stop_token0 bundle=control
#pragma HLS INTERFACE s_axilite port=stop_token1 bundle=control
#pragma HLS INTERFACE s_axilite port=dump_logits bundle=control
#pragma HLS INTERFACE s_axilite port=auxiliary bundle=control
#if defined(TINYLLM_PARALLEL_FFN)
#pragma HLS INTERFACE s_axilite port=w3_workspace bundle=control
#endif
#if !defined(TINYLLM_LEGACY_KV_MIRROR)
#pragma HLS INTERFACE s_axilite port=kv_cache_alias bundle=control
#endif
#pragma HLS INTERFACE s_axilite port=return bundle=control

#if defined(TINYLLM_LEGACY_KV_MIRROR)
#if defined(TINYLLM_PARALLEL_KV)
    const uint64_t *kv_memory = auxiliary;
    const uint64_t *vocab_stream = auxiliary
        + ((TENSOR_TOK_EMBEDDINGS_STREAM_OFFSET
            + VOCAB_STREAM_HALF_ROWS * TENSOR_TOK_EMBEDDINGS_STREAM_ROW_WORDS * 8u) >> 3);
    uint64_t *kv_mirror = auxiliary + MODEL_BINARY_BYTES / 8;
#else
    const uint64_t *vocab_stream = auxiliary;
    uint64_t *kv_mirror = auxiliary
        + (VOCAB_STREAM_HALF_ROWS * TENSOR_TOK_EMBEDDINGS_STREAM_ROW_WORDS);
#endif
#if defined(TINYLLM_PARALLEL_FFN)
    const uint64_t *w3_memory = w3_workspace;
#else
    const uint64_t *w3_memory = workspace;
#endif
#else
    const uint64_t *vocab_stream = auxiliary;
    const uint64_t *kv_mirror = kv_cache_alias;
    const uint64_t *w3_memory = workspace;
#endif

#if !defined(TINYLLM_PERFORMANCE_BUILD)
    if (prompt_length == 0 && max_new_tokens == -3) {
        const uint32_t debug_words[4] = {
            model_word(workspace, kWq[0].data_offset >> 2),
            model_word(workspace, (kWq[0].data_offset >> 2) + 1u),
            model_word(workspace, kWq[0].scale_offset >> 2),
            model_word(workspace, (kWq[0].scale_offset >> 2) + 1u),
        };
        for (int i = 0; i < 16; ++i) g_q_debug[i] = 0;
        float logits_lower[VOCAB_STREAM_HALF_ROWS];
        float logits_upper[VOCAB_STREAM_HALF_ROWS];
        model_step(workspace, io, vocab_stream, kv_mirror,
#if defined(TINYLLM_PARALLEL_KV)
                   kv_memory,
#endif
                   w3_memory, io[0], io[1],
                   logits_lower, logits_upper, true);
        const unsigned int meta = IO_TRACE_BASE + 78u * TRACE_RECORD_WORDS;
        io[meta + 0u] = kWq[0].data_offset;
        io[meta + 1u] = kWq[0].scale_offset;
        for (int i = 0; i < 4; ++i) io[meta + 2u + static_cast<unsigned int>(i)] = static_cast<int32_t>(debug_words[i]);
        for (int i = 0; i < 16; ++i) io[meta + 6u + static_cast<unsigned int>(i)] = static_cast<int32_t>(g_q_debug[i]);
        int selected = 0;
        float maximum = -3.402823466e+38f;
        for (int token = 0; token < MODEL_VOCAB_SIZE; ++token) {
#pragma HLS PIPELINE II=1
            const float value = token < VOCAB_STREAM_HALF_ROWS
                ? logits_lower[token] : logits_upper[token - VOCAB_STREAM_HALF_ROWS];
            if (value > maximum) {
                maximum = value;
                selected = token;
            }
            io[IO_LOGITS_BASE + token] = float_word(value);
        }
        io[IO_OUTPUT_TOKEN_BASE] = selected;
        io[IO_RESULT_COUNT_INDEX] = -3;
        io[IO_STOPPED_TOKEN_INDEX] = -3;
        return;
    }
#endif

    if (prompt_length == 0 && max_new_tokens == 0) {
        const unsigned int probe_offsets[18] = {
            TENSOR_TOK_EMBEDDINGS_SCALE_OFFSET,
            TENSOR_TOK_EMBEDDINGS_SCALE_OFFSET + 4u,
            TENSOR_TOK_EMBEDDINGS_SCALE_OFFSET + 16380u,
            TENSOR_TOK_EMBEDDINGS_DATA_OFFSET,
            TENSOR_TOK_EMBEDDINGS_DATA_OFFSET + 4u,
            TENSOR_TOK_EMBEDDINGS_DATA_OFFSET + 12u,
            TENSOR_LAYERS_0_ATTENTION_WQ_SCALE_OFFSET,
            TENSOR_LAYERS_0_ATTENTION_WQ_SCALE_OFFSET + 4u,
            TENSOR_LAYERS_0_ATTENTION_WQ_SCALE_OFFSET + 508u,
            TENSOR_LAYERS_0_ATTENTION_WQ_DATA_OFFSET,
            TENSOR_LAYERS_0_ATTENTION_WQ_DATA_OFFSET + 4u,
            TENSOR_LAYERS_0_ATTENTION_WQ_DATA_OFFSET + 12u,
            TENSOR_LAYERS_0_ATTENTION_NORM_DATA_OFFSET,
            TENSOR_LAYERS_0_ATTENTION_NORM_DATA_OFFSET + 4u,
            TENSOR_LAYERS_0_FFN_NORM_DATA_OFFSET,
            TENSOR_LAYERS_0_FFN_NORM_DATA_OFFSET + 4u,
            TENSOR_NORM_DATA_OFFSET,
            MODEL_BINARY_BYTES - 4u,
        };
        for (int i = 0; i < 18; ++i) {
#pragma HLS PIPELINE II=1
            io[IO_LOGITS_BASE + i] = static_cast<int32_t>(model_word(workspace, probe_offsets[i] >> 2));
        }
        io[IO_RESULT_COUNT_INDEX] = -2;
        return;
    }

    if (prompt_length < 1 || prompt_length > MODEL_MAX_SEQ_LEN
        || max_new_tokens < 1 || max_new_tokens > MAX_GENERATED_TOKENS
        || min_new_tokens < 0 || min_new_tokens > max_new_tokens
        || prompt_length + max_new_tokens > MODEL_MAX_SEQ_LEN) {
        io[IO_RESULT_COUNT_INDEX] = -1;
        return;
    }

    float logits_lower[VOCAB_STREAM_HALF_ROWS];
    float logits_upper[VOCAB_STREAM_HALF_ROWS];
    for (int position = 0; position < prompt_length; ++position) {
        model_step(workspace, io, vocab_stream, kv_mirror,
#if defined(TINYLLM_PARALLEL_KV)
                   kv_memory,
#endif
                   w3_memory, io[position], position,
                   logits_lower, logits_upper, false);
    }

    uint8_t generated = 0;
    int stopped_on = -1;
    for (uint8_t step = 0; step < max_new_tokens; ++step) {
        float maximum = -3.402823466e+38f;
        int selected = 0;
        for (int token = 0; token < MODEL_VOCAB_SIZE; ++token) {
            const bool is_stop = token == stop_token0 || token == stop_token1;
            if (step < min_new_tokens && is_stop) continue;
            const float value = token < VOCAB_STREAM_HALF_ROWS
                ? logits_lower[token] : logits_upper[token - VOCAB_STREAM_HALF_ROWS];
            if (value > maximum) {
                maximum = value;
                selected = token;
            }
        }
        io[IO_OUTPUT_TOKEN_BASE + step] = selected;
        if (dump_logits) {
            union FloatBits { uint32_t bits; float value; } value;
            for (int token = 0; token < MODEL_VOCAB_SIZE; ++token) {
#pragma HLS PIPELINE II=1
                value.value = token < VOCAB_STREAM_HALF_ROWS
                    ? logits_lower[token] : logits_upper[token - VOCAB_STREAM_HALF_ROWS];
                io[IO_LOGITS_BASE + step * MODEL_VOCAB_SIZE + token] = static_cast<int32_t>(value.bits);
            }
        }
        ++generated;
        if (step >= min_new_tokens && (selected == stop_token0 || selected == stop_token1)) {
            stopped_on = selected;
            break;
        }
        if (step + 1 < max_new_tokens) {
            model_step(workspace, io, vocab_stream, kv_mirror,
#if defined(TINYLLM_PARALLEL_KV)
                       kv_memory,
#endif
                       w3_memory, selected, prompt_length + step,
                       logits_lower, logits_upper, false);
        }
    }
    io[IO_RESULT_COUNT_INDEX] = generated;
    io[IO_STOPPED_TOKEN_INDEX] = stopped_on;
}
