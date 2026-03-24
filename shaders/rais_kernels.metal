#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Kernel 1: elementwise_add_f32
// Simple vector addition. Used for correctness testing and as a baseline
// for GPU dispatch overhead measurement.
// ---------------------------------------------------------------------------
kernel void elementwise_add_f32(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device float*       result [[buffer(2)]],
    uint                gid    [[thread_position_in_grid]]
) {
    result[gid] = a[gid] + b[gid];
}

// ---------------------------------------------------------------------------
// Kernel 2: rms_norm_f32
// RMS normalization used in transformer inference:
//   output[i] = input[i] / sqrt(mean(input^2) + epsilon) * weight[i]
//
// Grid: one threadgroup per row (grid_size.x = num_rows).
// Each threadgroup computes the RMS for its row using a tree reduction
// in threadgroup memory.
// ---------------------------------------------------------------------------
kernel void rms_norm_f32(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device float*       output  [[buffer(2)]],
    constant uint&      cols    [[buffer(3)]],
    constant float&     epsilon [[buffer(4)]],
    uint                row     [[threadgroup_position_in_grid]],
    uint                lid     [[thread_index_in_threadgroup]],
    uint                tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[1024];

    uint row_offset = row * cols;

    // Each thread accumulates a partial sum of squared values across
    // its stride of the row.
    float partial = 0.0f;
    for (uint i = lid; i < cols; i += tg_size) {
        float v = input[row_offset + i];
        partial += v * v;
    }
    shared_sum[lid] = partial;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction to compute the total sum of squares.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared_sum[lid] += shared_sum[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // RMS = sqrt(mean(x^2) + epsilon)
    float rms = rsqrt(shared_sum[0] / float(cols) + epsilon);

    // Normalize and scale by weight
    for (uint i = lid; i < cols; i += tg_size) {
        output[row_offset + i] = input[row_offset + i] * rms * weight[i];
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: silu_f32
// SiLU activation: output[i] = input[i] * sigmoid(input[i])
//                             = input[i] / (1 + exp(-input[i]))
//
// Vectorized using float4 reads/writes for 4x throughput per thread.
// The CPU-side dispatch must ensure buffer sizes are multiples of 4.
// ---------------------------------------------------------------------------
kernel void silu_f32(
    device const float4* input  [[buffer(0)]],
    device float4*       output [[buffer(1)]],
    uint                 gid    [[thread_position_in_grid]]
) {
    float4 x = input[gid];
    // precise::sigmoid(x) = 1 / (1 + exp(-x))
    float4 sig = 1.0f / (1.0f + exp(-x));
    output[gid] = x * sig;
}

// ---------------------------------------------------------------------------
// Kernel 4: attention_scores_f32
// Scaled dot-product: scores[i][j] = dot(Q[i], K[j]) / sqrt(d_k)
//
// Q shape: [seq_len, d_k], K shape: [seq_len, d_k], output: [seq_len, seq_len]
//
// Uses 16x16 threadgroup tiling. Each threadgroup computes a 16x16 tile
// of the output matrix by iterating over d_k in chunks, loading tiles
// of Q and K^T into threadgroup memory.
//
// The optimal tile size differs between M1 (smaller GPU, 8-16 cores) and
// M3/M4 (up to 40 cores with better caches). The CPU dispatch code should
// benchmark both 16x16 and 32x32 tiles at init time and select accordingly.
// This kernel uses 16x16 as the safe default that works well on all chips.
// ---------------------------------------------------------------------------

constant constexpr uint TILE_SIZE = 16;

kernel void attention_scores_f32(
    device const float* Q         [[buffer(0)]],
    device const float* K         [[buffer(1)]],
    device float*       scores    [[buffer(2)]],
    constant uint&      seq_len   [[buffer(3)]],
    constant uint&      d_k       [[buffer(4)]],
    constant float&     scale     [[buffer(5)]],  // 1.0 / sqrt(d_k), precomputed on CPU
    uint2               gid       [[thread_position_in_grid]],
    uint2               lid       [[thread_position_in_threadgroup]],
    uint2               tgid      [[threadgroup_position_in_grid]]
) {
    // Shared memory tiles for Q and K^T
    threadgroup float q_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float k_tile[TILE_SIZE][TILE_SIZE];

    uint row = tgid.y * TILE_SIZE + lid.y;  // output row (Q index)
    uint col = tgid.x * TILE_SIZE + lid.x;  // output col (K index)

    float sum = 0.0f;

    // Iterate over d_k in chunks of TILE_SIZE
    uint num_tiles = (d_k + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < num_tiles; ++t) {
        uint k_offset = t * TILE_SIZE;

        // Load Q tile: Q[row, k_offset + lid.x]
        if (row < seq_len && (k_offset + lid.x) < d_k) {
            q_tile[lid.y][lid.x] = Q[row * d_k + k_offset + lid.x];
        } else {
            q_tile[lid.y][lid.x] = 0.0f;
        }

        // Load K^T tile: K[col, k_offset + lid.y] (transposed access)
        if (col < seq_len && (k_offset + lid.y) < d_k) {
            k_tile[lid.y][lid.x] = K[col * d_k + k_offset + lid.y];
        } else {
            k_tile[lid.y][lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product for this tile
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += q_tile[lid.y][k] * k_tile[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write scaled result
    if (row < seq_len && col < seq_len) {
        scores[row * seq_len + col] = sum * scale;
    }
}
