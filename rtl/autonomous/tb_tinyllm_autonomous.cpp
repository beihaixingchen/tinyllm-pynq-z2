#include <cstdint>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
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
#define VOCAB_STREAM_HALF_BYTES (VOCAB_STREAM_HALF_ROWS * TENSOR_TOK_EMBEDDINGS_STREAM_ROW_WORDS * 8)
#define KV_WORDS (MODEL_LAYERS * MODEL_MAX_SEQ_LEN * MODEL_KV_HEADS * (MODEL_DIM / MODEL_HEADS) * 2)
#define MODEL_WORKSPACE_BYTES (MODEL_BINARY_BYTES + KV_WORDS * 4)
#define ROPE_TABLE_BYTES (MODEL_MAX_SEQ_LEN * (MODEL_DIM / MODEL_HEADS) * 4)

void tinyllm_autonomous(uint64_t *workspace, int32_t *io,
                        int prompt_length, int max_new_tokens, int min_new_tokens,
                        int stop_token0, int stop_token1, int dump_logits,
#if defined(TINYLLM_LEGACY_KV_MIRROR)
#if defined(TINYLLM_PARALLEL_FFN)
                        uint64_t *auxiliary, const uint64_t *w3_workspace);
#else
                        uint64_t *auxiliary);
#endif
#else
                        const uint64_t *auxiliary,
                        const uint64_t *kv_cache_alias);
#endif

int main(int argc, char **argv) {
#if defined(TINYLLM_PRECOMPUTED_ROPE)
    if (argc != 5) {
        std::cerr << "usage: tb autonomous_model.bin hls_prompt.bin reference_logits.bin rope_table.bin" << std::endl;
        return 2;
    }
#else
    if (argc != 4) {
        std::cerr << "usage: tb autonomous_model.bin hls_prompt.bin reference_logits.bin" << std::endl;
        return 2;
    }
#endif
    size_t workspace_bytes = MODEL_WORKSPACE_BYTES;
#if defined(TINYLLM_PRECOMPUTED_ROPE)
    workspace_bytes += ROPE_TABLE_BYTES;
#endif
    std::vector<uint64_t> workspace(workspace_bytes / 8, 0);
#if defined(TINYLLM_PARALLEL_KV)
    std::vector<uint64_t> auxiliary((MODEL_BINARY_BYTES + KV_WORDS * 4) / 8, 0);
#elif defined(TINYLLM_LEGACY_KV_MIRROR)
    std::vector<uint64_t> auxiliary((VOCAB_STREAM_HALF_BYTES + KV_WORDS * 4) / 8, 0);
#else
    std::vector<uint64_t> vocab_stream(VOCAB_STREAM_HALF_BYTES / 8, 0);
#endif
    std::ifstream model(argv[1], std::ios::binary);
    model.read(reinterpret_cast<char *>(workspace.data()), MODEL_BINARY_BYTES);
    if (!model || model.gcount() != MODEL_BINARY_BYTES) {
        std::cerr << "could not read complete autonomous model binary" << std::endl;
        return 3;
    }
#if defined(TINYLLM_PRECOMPUTED_ROPE)
    std::ifstream rope_table(argv[4], std::ios::binary);
    rope_table.read(reinterpret_cast<char *>(workspace.data()) + MODEL_WORKSPACE_BYTES,
                    ROPE_TABLE_BYTES);
    if (!rope_table || rope_table.gcount() != ROPE_TABLE_BYTES) {
        std::cerr << "could not read complete RoPE table" << std::endl;
        return 3;
    }
#endif
    std::vector<int32_t> io(IO_WORDS, 0);
    const unsigned int upper_stream_offset = TENSOR_TOK_EMBEDDINGS_STREAM_OFFSET
        + VOCAB_STREAM_HALF_ROWS * TENSOR_TOK_EMBEDDINGS_STREAM_ROW_WORDS * 8u;
#if defined(TINYLLM_PARALLEL_KV)
    std::memcpy(auxiliary.data(), workspace.data(), MODEL_BINARY_BYTES);
#elif defined(TINYLLM_LEGACY_KV_MIRROR)
    std::memcpy(auxiliary.data(),
                reinterpret_cast<const uint8_t *>(workspace.data()) + upper_stream_offset,
                VOCAB_STREAM_HALF_BYTES);
#else
    std::memcpy(vocab_stream.data(),
                reinterpret_cast<const uint8_t *>(workspace.data()) + upper_stream_offset,
                VOCAB_STREAM_HALF_BYTES);
#endif
    std::ifstream prompt(argv[2], std::ios::binary);
    int32_t header[5] = {0, 0, 0, 0, 0};
    prompt.read(reinterpret_cast<char *>(header), sizeof(header));
    const int prompt_length = header[0];
    const int max_new = header[1];
    const int min_new = header[2];
    const int stop0 = header[3];
    const int stop1 = header[4];
    if (!prompt || prompt_length < 1 || prompt_length > MODEL_MAX_SEQ_LEN
        || max_new < 1 || max_new > MAX_GENERATED_TOKENS
        || min_new < 0 || min_new > max_new) {
        std::cerr << "invalid HLS prompt header" << std::endl;
        return 4;
    }
    prompt.read(reinterpret_cast<char *>(io.data()), prompt_length * sizeof(int32_t));
    if (!prompt || prompt.gcount() != prompt_length * static_cast<std::streamsize>(sizeof(int32_t))) {
        std::cerr << "could not read prompt token IDs" << std::endl;
        return 4;
    }
    std::vector<float> reference(max_new * MODEL_VOCAB_SIZE);
    std::ifstream golden(argv[3], std::ios::binary);
    golden.seekg(0, std::ios::end);
    const std::streamsize golden_bytes = golden.tellg();
    golden.seekg(0, std::ios::beg);
    const int reference_steps = static_cast<int>(golden_bytes / (MODEL_VOCAB_SIZE * sizeof(float)));
    if (reference_steps < 1 || reference_steps > max_new
        || golden_bytes % (MODEL_VOCAB_SIZE * sizeof(float)) != 0) {
        std::cerr << "invalid golden-logit file size" << std::endl;
        return 5;
    }
    reference.resize(reference_steps * MODEL_VOCAB_SIZE);
    golden.read(reinterpret_cast<char *>(reference.data()), reference.size() * sizeof(float));
    if (!golden || golden.gcount() != reference.size() * static_cast<std::streamsize>(sizeof(float))) {
        std::cerr << "could not read complete golden logits" << std::endl;
        return 5;
    }

#if defined(TINYLLM_LEGACY_KV_MIRROR)
#if defined(TINYLLM_PARALLEL_FFN)
    tinyllm_autonomous(workspace.data(), io.data(), prompt_length, max_new, min_new,
                       stop0, stop1, 1, auxiliary.data(), workspace.data());
#else
    tinyllm_autonomous(workspace.data(), io.data(), prompt_length, max_new, min_new,
                       stop0, stop1, 1, auxiliary.data());
#endif
#else
    tinyllm_autonomous(workspace.data(), io.data(), prompt_length, max_new, min_new,
                       stop0, stop1, 1, vocab_stream.data(),
                       workspace.data() + MODEL_BINARY_BYTES / sizeof(uint64_t));
#endif
    if (io[IO_RESULT_COUNT_INDEX] < 0) {
        std::cerr << "invalid invocation rejected by accelerator" << std::endl;
        return 6;
    }
    int mismatch = 0;
    int64_t exact_argmax = 0;
    double squared_error = 0.0;
    double reference_squared = 0.0;
    double max_abs_error = 0.0;
    const int output_count = io[IO_RESULT_COUNT_INDEX];
    if (output_count != reference_steps) {
        std::cerr << "sample count differs from golden: HLS=" << output_count
                  << " golden=" << reference_steps << std::endl;
        return 7;
    }
    for (int step = 0; step < reference_steps; ++step) {
        int expected_argmax = 0;
        int actual_argmax = 0;
        float expected_max = reference[step * MODEL_VOCAB_SIZE];
        float actual_max = reinterpret_cast<float *>(io.data() + IO_LOGITS_BASE)[step * MODEL_VOCAB_SIZE];
        int expected_sample = (step < min_new && (0 == stop0 || 0 == stop1)) ? -1 : 0;
        float sample_max = expected_sample < 0 ? -3.402823466e+38f : expected_max;
        for (int token = 1; token < MODEL_VOCAB_SIZE; ++token) {
            const float expected = reference[step * MODEL_VOCAB_SIZE + token];
            const float actual = reinterpret_cast<float *>(io.data() + IO_LOGITS_BASE)[step * MODEL_VOCAB_SIZE + token];
            if (expected > expected_max) { expected_max = expected; expected_argmax = token; }
            if (actual > actual_max) { actual_max = actual; actual_argmax = token; }
            if (!(step < min_new && (token == stop0 || token == stop1)) && expected > sample_max) {
                sample_max = expected;
                expected_sample = token;
            }
            const double delta = static_cast<double>(actual) - expected;
            squared_error += delta * delta;
            reference_squared += static_cast<double>(expected) * expected;
            const double abs_error = delta < 0.0 ? -delta : delta;
            if (abs_error > max_abs_error) max_abs_error = abs_error;
        }
        if (expected_argmax == actual_argmax) ++exact_argmax;
        else ++mismatch;
        std::cout << "step " << step << ": token=" << io[IO_OUTPUT_TOKEN_BASE + step]
                  << " expected_token=" << expected_sample
                  << " ref_argmax=" << expected_argmax << " hls_argmax=" << actual_argmax << std::endl;
        if (io[IO_OUTPUT_TOKEN_BASE + step] != expected_sample) ++mismatch;
    }
    const double relative_rmse = std::sqrt(squared_error / reference_squared);
    std::cout << "generated=" << io[IO_RESULT_COUNT_INDEX]
              << " stopped_on=" << io[IO_STOPPED_TOKEN_INDEX]
              << " argmax_match=" << exact_argmax << "/" << reference_steps
              << " mismatches=" << mismatch
              << " relative_rmse=" << relative_rmse
              << " max_abs_error=" << max_abs_error << std::endl;
    return mismatch == 0 ? 0 : 1;
}
