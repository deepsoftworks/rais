#import <Metal/Metal.h>

#include <catch2/catch_test_macros.hpp>

#include <rais/metal_executor.hpp>

#include <catch2/catch_approx.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <thread>
#include <vector>

// Path to the compiled metallib set by CMake as a compile definition
#ifndef RAIS_METALLIB_PATH
#error "RAIS_METALLIB_PATH must be defined"
#endif

static id<MTLDevice> get_device() {
    return MTLCreateSystemDefaultDevice();
}

static std::filesystem::path metallib_path() {
    return RAIS_METALLIB_PATH;
}

TEST_CASE("Trivial kernel (elementwise add) produces correct output", "[metal]") {
    id<MTLDevice> device = get_device();
    REQUIRE(device != nil);

    rais::MetalExecutor exec((__bridge void*)device, metallib_path());

    constexpr uint32_t N = 1024;
    constexpr size_t buf_size = N * sizeof(float);

    id<MTLBuffer> buf_a = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_b = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_out = [device newBufferWithLength:buf_size
                                                options:MTLResourceStorageModeShared];

    auto* a = static_cast<float*>(buf_a.contents);
    auto* b = static_cast<float*>(buf_b.contents);
    for (uint32_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    void* pso = exec.pipeline("elementwise_add_f32");
    REQUIRE(pso != nullptr);

    std::atomic<bool> completed{false};

    bool submitted = exec.submit(
        [&](void* /* cmd_buf */, void* enc) {
            auto encoder = (__bridge id<MTLComputeCommandEncoder>)enc;
            [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
            [encoder setBuffer:buf_a offset:0 atIndex:0];
            [encoder setBuffer:buf_b offset:0 atIndex:1];
            [encoder setBuffer:buf_out offset:0 atIndex:2];

            MTLSize grid = MTLSizeMake(N, 1, 1);
            NSUInteger thread_width =
                ((__bridge id<MTLComputePipelineState>)pso).threadExecutionWidth;
            MTLSize tg = MTLSizeMake(thread_width, 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        },
        [&completed]() {
            completed.store(true, std::memory_order_release);
        });

    REQUIRE(submitted);
    exec.flush();
    REQUIRE(completed.load(std::memory_order_acquire));

    auto* out = static_cast<float*>(buf_out.contents);
    for (uint32_t i = 0; i < N; ++i) {
        REQUIRE(out[i] == static_cast<float>(i + i * 2));
    }
}

TEST_CASE("100 concurrent GPU tasks all complete", "[metal]") {
    id<MTLDevice> device = get_device();
    REQUIRE(device != nil);

    rais::MetalExecutor exec((__bridge void*)device, metallib_path());

    constexpr int NUM_TASKS = 100;
    constexpr uint32_t N = 256;
    constexpr size_t buf_size = N * sizeof(float);

    std::atomic<int> complete_count{0};
    void* pso = exec.pipeline("elementwise_add_f32");

    // Shared input buffers
    id<MTLBuffer> buf_a = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_b = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];

    auto* a = static_cast<float*>(buf_a.contents);
    auto* b = static_cast<float*>(buf_b.contents);
    for (uint32_t i = 0; i < N; ++i) { a[i] = 1.0f; b[i] = 2.0f; }

    // Each task gets its own output buffer
    std::vector<id<MTLBuffer>> outputs(NUM_TASKS);
    for (int t = 0; t < NUM_TASKS; ++t) {
        outputs[t] = [device newBufferWithLength:buf_size
                                         options:MTLResourceStorageModeShared];
    }

    for (int t = 0; t < NUM_TASKS; ++t) {
        // Retry with backoff if backpressure
        while (!exec.submit(
            [&, t](void* /* cmd_buf */, void* enc) {
                auto encoder = (__bridge id<MTLComputeCommandEncoder>)enc;
                [encoder setComputePipelineState:
                    (__bridge id<MTLComputePipelineState>)pso];
                [encoder setBuffer:buf_a offset:0 atIndex:0];
                [encoder setBuffer:buf_b offset:0 atIndex:1];
                [encoder setBuffer:outputs[t] offset:0 atIndex:2];

                MTLSize grid = MTLSizeMake(N, 1, 1);
                NSUInteger tw =
                    ((__bridge id<MTLComputePipelineState>)pso).threadExecutionWidth;
                MTLSize tg = MTLSizeMake(tw, 1, 1);
                [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
            },
            [&complete_count]() {
                complete_count.fetch_add(1, std::memory_order_relaxed);
            }))
        {
            std::this_thread::yield();
        }
    }

    exec.flush();
    REQUIRE(complete_count.load() == NUM_TASKS);

    // Verify all outputs
    for (int t = 0; t < NUM_TASKS; ++t) {
        auto* out = static_cast<float*>(outputs[t].contents);
        for (uint32_t i = 0; i < N; ++i) {
            REQUIRE(out[i] == 3.0f);
        }
    }
}

TEST_CASE("Backpressure — submit returns false when ring full", "[metal]") {
    id<MTLDevice> device = get_device();
    REQUIRE(device != nil);

    rais::MetalExecutor exec((__bridge void*)device, metallib_path());
    void* pso = exec.pipeline("elementwise_add_f32");

    constexpr uint32_t N = 64;
    constexpr size_t buf_size = N * sizeof(float);

    id<MTLBuffer> buf_a = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_b = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_out = [device newBufferWithLength:buf_size
                                                options:MTLResourceStorageModeShared];

    int submitted = 0;
    int rejected = 0;
    std::atomic<int> completed{0};

    // Try to submit 200 tasks rapidly — some should be rejected
    for (int i = 0; i < 200; ++i) {
        bool ok = exec.submit(
            [&](void* /* cmd_buf */, void* enc) {
                auto encoder = (__bridge id<MTLComputeCommandEncoder>)enc;
                [encoder setComputePipelineState:
                    (__bridge id<MTLComputePipelineState>)pso];
                [encoder setBuffer:buf_a offset:0 atIndex:0];
                [encoder setBuffer:buf_b offset:0 atIndex:1];
                [encoder setBuffer:buf_out offset:0 atIndex:2];

                MTLSize grid = MTLSizeMake(N, 1, 1);
                NSUInteger tw =
                    ((__bridge id<MTLComputePipelineState>)pso).threadExecutionWidth;
                MTLSize tg = MTLSizeMake(tw, 1, 1);
                [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
            },
            [&completed]() {
                completed.fetch_add(1, std::memory_order_relaxed);
            });

        if (ok) ++submitted;
        else ++rejected;
    }

    // We expect some rejections (ring buffer capacity is 8)
    REQUIRE(rejected > 0);
    REQUIRE(submitted > 0);

    exec.flush();
    REQUIRE(completed.load() == submitted);
}

// ---------------------------------------------------------------------------
// Kernel tests for rms_norm, silu, and attention_scores
// ---------------------------------------------------------------------------

TEST_CASE("rms_norm_f32 — single row matches CPU reference", "[metal]") {
    id<MTLDevice> device = get_device();
    REQUIRE(device != nil);

    rais::MetalExecutor exec((__bridge void*)device, metallib_path());
    void* pso = exec.pipeline("rms_norm_f32");
    REQUIRE(pso != nullptr);

    constexpr uint32_t ROWS = 4;
    constexpr uint32_t COLS = 128;
    constexpr float EPS = 1e-5f;
    constexpr size_t data_size = ROWS * COLS * sizeof(float);
    constexpr size_t weight_size = COLS * sizeof(float);

    id<MTLBuffer> buf_input  = [device newBufferWithLength:data_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_weight = [device newBufferWithLength:weight_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_output = [device newBufferWithLength:data_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_cols   = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_eps    = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

    auto* input  = static_cast<float*>(buf_input.contents);
    auto* weight = static_cast<float*>(buf_weight.contents);
    auto* cols_p = static_cast<uint32_t*>(buf_cols.contents);
    auto* eps_p  = static_cast<float*>(buf_eps.contents);

    *cols_p = COLS;
    *eps_p = EPS;

    for (uint32_t r = 0; r < ROWS; ++r) {
        for (uint32_t c = 0; c < COLS; ++c) {
            input[r * COLS + c] = static_cast<float>(r * COLS + c) * 0.01f;
        }
    }
    for (uint32_t c = 0; c < COLS; ++c) {
        weight[c] = 1.0f; // identity weight for easy verification
    }

    bool ok = exec.submit([&](void*, void* enc) {
        auto encoder = (__bridge id<MTLComputeCommandEncoder>)enc;
        [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
        [encoder setBuffer:buf_input  offset:0 atIndex:0];
        [encoder setBuffer:buf_weight offset:0 atIndex:1];
        [encoder setBuffer:buf_output offset:0 atIndex:2];
        [encoder setBuffer:buf_cols   offset:0 atIndex:3];
        [encoder setBuffer:buf_eps    offset:0 atIndex:4];

        // One threadgroup per row, threadgroup size = min(COLS, 1024)
        uint32_t tg_size = COLS < 1024 ? COLS : 1024;
        MTLSize grid = MTLSizeMake(tg_size * ROWS, 1, 1);
        MTLSize tg = MTLSizeMake(tg_size, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    });
    REQUIRE(ok);
    exec.flush();

    auto* output = static_cast<float*>(buf_output.contents);

    // CPU reference
    for (uint32_t r = 0; r < ROWS; ++r) {
        float sum_sq = 0.0f;
        for (uint32_t c = 0; c < COLS; ++c) {
            float v = input[r * COLS + c];
            sum_sq += v * v;
        }
        float rms = 1.0f / std::sqrt(sum_sq / COLS + EPS);

        for (uint32_t c = 0; c < COLS; ++c) {
            float expected = input[r * COLS + c] * rms * weight[c];
            REQUIRE(output[r * COLS + c] == Catch::Approx(expected).epsilon(1e-4));
        }
    }
}

TEST_CASE("silu_f32 — matches CPU reference", "[metal]") {
    id<MTLDevice> device = get_device();
    REQUIRE(device != nil);

    rais::MetalExecutor exec((__bridge void*)device, metallib_path());
    void* pso = exec.pipeline("silu_f32");
    REQUIRE(pso != nullptr);

    // N must be multiple of 4 (kernel uses float4)
    constexpr uint32_t N = 1024;
    constexpr size_t buf_size = N * sizeof(float);

    id<MTLBuffer> buf_input  = [device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_output = [device newBufferWithLength:buf_size options:MTLResourceStorageModeShared];

    auto* input = static_cast<float*>(buf_input.contents);
    for (uint32_t i = 0; i < N; ++i) {
        input[i] = (static_cast<float>(i) - static_cast<float>(N / 2)) * 0.01f;
    }

    bool ok = exec.submit([&](void*, void* enc) {
        auto encoder = (__bridge id<MTLComputeCommandEncoder>)enc;
        [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
        [encoder setBuffer:buf_input  offset:0 atIndex:0];
        [encoder setBuffer:buf_output offset:0 atIndex:1];

        // Dispatch N/4 threads (each processes a float4)
        uint32_t num_vec4 = N / 4;
        NSUInteger tw = ((__bridge id<MTLComputePipelineState>)pso).threadExecutionWidth;
        MTLSize grid = MTLSizeMake(num_vec4, 1, 1);
        MTLSize tg = MTLSizeMake(tw, 1, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    });
    REQUIRE(ok);
    exec.flush();

    auto* output = static_cast<float*>(buf_output.contents);
    for (uint32_t i = 0; i < N; ++i) {
        float x = input[i];
        float expected = x / (1.0f + std::exp(-x));
        REQUIRE(output[i] == Catch::Approx(expected).epsilon(1e-4));
    }
}

TEST_CASE("attention_scores_f32 — small matrix matches CPU reference", "[metal]") {
    id<MTLDevice> device = get_device();
    REQUIRE(device != nil);

    rais::MetalExecutor exec((__bridge void*)device, metallib_path());
    void* pso = exec.pipeline("attention_scores_f32");
    REQUIRE(pso != nullptr);

    constexpr uint32_t SEQ_LEN = 32;
    constexpr uint32_t D_K = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(D_K));

    constexpr size_t qk_size = SEQ_LEN * D_K * sizeof(float);
    constexpr size_t out_size = SEQ_LEN * SEQ_LEN * sizeof(float);

    id<MTLBuffer> buf_q     = [device newBufferWithLength:qk_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_k     = [device newBufferWithLength:qk_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_out   = [device newBufferWithLength:out_size options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_seq   = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_dk    = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_scale = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

    auto* q = static_cast<float*>(buf_q.contents);
    auto* k = static_cast<float*>(buf_k.contents);
    *static_cast<uint32_t*>(buf_seq.contents) = SEQ_LEN;
    *static_cast<uint32_t*>(buf_dk.contents) = D_K;
    *static_cast<float*>(buf_scale.contents) = scale;

    // Fill with small deterministic values
    for (uint32_t i = 0; i < SEQ_LEN * D_K; ++i) {
        q[i] = static_cast<float>(i % 17) * 0.1f;
        k[i] = static_cast<float>(i % 13) * 0.1f;
    }

    bool ok = exec.submit([&](void*, void* enc) {
        auto encoder = (__bridge id<MTLComputeCommandEncoder>)enc;
        [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
        [encoder setBuffer:buf_q     offset:0 atIndex:0];
        [encoder setBuffer:buf_k     offset:0 atIndex:1];
        [encoder setBuffer:buf_out   offset:0 atIndex:2];
        [encoder setBuffer:buf_seq   offset:0 atIndex:3];
        [encoder setBuffer:buf_dk    offset:0 atIndex:4];
        [encoder setBuffer:buf_scale offset:0 atIndex:5];

        // Grid covers the full output matrix, tiled at 16x16
        MTLSize grid = MTLSizeMake(SEQ_LEN, SEQ_LEN, 1);
        MTLSize tg = MTLSizeMake(16, 16, 1);
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    });
    REQUIRE(ok);
    exec.flush();

    auto* out = static_cast<float*>(buf_out.contents);

    // CPU reference: scores[i][j] = dot(Q[i], K[j]) * scale
    for (uint32_t i = 0; i < SEQ_LEN; ++i) {
        for (uint32_t j = 0; j < SEQ_LEN; ++j) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < D_K; ++d) {
                dot += q[i * D_K + d] * k[j * D_K + d];
            }
            float expected = dot * scale;
            REQUIRE(out[i * SEQ_LEN + j] == Catch::Approx(expected).epsilon(1e-3));
        }
    }
}
