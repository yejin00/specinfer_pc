#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

// Minimal types needed for Q4_0_PC
typedef uint16_t ggml_fp16_t;

// FP16 conversion (simplified - use proper implementation if available)
static inline float fp16_to_fp32(ggml_fp16_t h) {
    // Simple conversion - may need proper IEEE 754 half-precision conversion
    uint32_t w = (uint32_t)h << 16;
    return *(float*)&w;
}

static inline ggml_fp16_t fp32_to_fp16(float f) {
    // Simple conversion - may need proper IEEE 754 half-precision conversion
    uint32_t w = *(uint32_t*)&f;
    return (ggml_fp16_t)(w >> 16);
}

// Q4_0_PC quantization function (from ops.cpp)
void quantize_q4_0_pc(const float* src, void* dst, int n_dims, int n_tokens) {
    const size_t scales_size = n_dims * sizeof(ggml_fp16_t);
    const size_t data_size = n_dims * ((n_tokens + 1) / 2);
    
    ggml_fp16_t* scales = (ggml_fp16_t*)dst;
    uint8_t* data = (uint8_t*)((char*)dst + scales_size);
    
    printf("Quantizing: n_dims=%d, n_tokens=%d\n", n_dims, n_tokens);
    printf("  scales_size=%zu, data_size=%zu\n", scales_size, data_size);
    
    // Compute per-dimension scales
    for (int d = 0; d < n_dims; d++) {
        float max_abs = 0.0f;
        for (int t = 0; t < n_tokens; t++) {
            float val = src[t * n_dims + d];
            max_abs = fmaxf(max_abs, fabsf(val));
        }
        float scale = max_abs / 7.0f;  // 4-bit signed: -7 to +7
        scales[d] = fp32_to_fp16(scale);
    }
    
    // Quantize data
    for (int t = 0; t < n_tokens; t++) {
        for (int d = 0; d < n_dims; d++) {
            float val = src[t * n_dims + d];
            float scale = fp16_to_fp32(scales[d]);
            
            int8_t q;
            if (scale < 1e-8f) {
                q = 0;
            } else {
                float scaled = val / scale;
                q = (int8_t)roundf(fmaxf(-7.0f, fminf(7.0f, scaled)));
            }
            
            // Pack into 4-bit
            int data_idx = t * n_dims + d;
            int byte_idx = data_idx / 2;
            int nibble = data_idx % 2;
            
            if (nibble == 0) {
                data[byte_idx] = (q & 0x0F);
            } else {
                data[byte_idx] |= ((q & 0x0F) << 4);
            }
        }
    }
}

// Q4_0_PC dequantization function
void dequantize_q4_0_pc(const void* src, float* dst, int n_dims, int n_tokens) {
    const size_t scales_size = n_dims * sizeof(ggml_fp16_t);
    
    const ggml_fp16_t* scales = (const ggml_fp16_t*)src;
    const uint8_t* data = (const uint8_t*)((const char*)src + scales_size);
    
    for (int t = 0; t < n_tokens; t++) {
        for (int d = 0; d < n_dims; d++) {
            float scale = fp16_to_fp32(scales[d]);
            
            // Unpack 4-bit
            int data_idx = t * n_dims + d;
            int byte_idx = data_idx / 2;
            int nibble = data_idx % 2;
            
            int8_t q;
            if (nibble == 0) {
                q = (int8_t)(data[byte_idx] & 0x0F);
            } else {
                q = (int8_t)((data[byte_idx] >> 4) & 0x0F);
            }
            
            // Sign extend 4-bit to 8-bit
            if (q & 0x08) {
                q |= 0xF0;
            }
            
            dst[t * n_dims + d] = scale * (float)q;
        }
    }
}

int main() {
    // Test parameters
    const int n_dims = 128;
    const int n_tokens = 4;
    const int n_elements = n_dims * n_tokens;
    
    printf("=== Q4_0_PC Quantization Test ===\n");
    printf("n_dims=%d, n_tokens=%d, n_elements=%d\n\n", n_dims, n_tokens, n_elements);
    
    // Allocate buffers
    std::vector<float> src(n_elements);
    std::vector<float> dst(n_elements);
    
    const size_t scales_size = n_dims * sizeof(ggml_fp16_t);
    const size_t data_size = n_dims * ((n_tokens + 1) / 2);
    const size_t quantized_size = scales_size + data_size;
    std::vector<uint8_t> quantized(quantized_size);
    
    // Initialize with test data
    printf("Initializing test data...\n");
    for (int i = 0; i < n_elements; i++) {
        src[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random [-1, 1]
    }
    
    // Print first few values
    printf("Original values (first 8):\n");
    for (int i = 0; i < 8; i++) {
        printf("  src[%d] = %.6f\n", i, src[i]);
    }
    printf("\n");
    
    // Quantize
    printf("Quantizing...\n");
    quantize_q4_0_pc(src.data(), quantized.data(), n_dims, n_tokens);
    printf("\n");
    
    // Dequantize
    printf("Dequantizing...\n");
    dequantize_q4_0_pc(quantized.data(), dst.data(), n_dims, n_tokens);
    printf("\n");
    
    // Compare
    printf("Reconstructed values (first 8):\n");
    for (int i = 0; i < 8; i++) {
        printf("  dst[%d] = %.6f (error: %.6f)\n", i, dst[i], dst[i] - src[i]);
    }
    printf("\n");
    
    // Compute statistics
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int nan_count = 0;
    
    for (int i = 0; i < n_elements; i++) {
        if (std::isnan(dst[i])) {
            nan_count++;
            printf("ERROR: NaN at index %d (token=%d, dim=%d)\n", 
                   i, i / n_dims, i % n_dims);
        } else {
            float error = fabsf(dst[i] - src[i]);
            max_error = fmaxf(max_error, error);
            avg_error += error;
        }
    }
    avg_error /= n_elements;
    
    printf("=== Results ===\n");
    printf("NaN count: %d / %d\n", nan_count, n_elements);
    printf("Max error: %.6f\n", max_error);
    printf("Avg error: %.6f\n", avg_error);
    
    if (nan_count > 0) {
        printf("\n❌ TEST FAILED: NaN values detected!\n");
        return 1;
    } else if (max_error > 0.5f) {
        printf("\n⚠️  WARNING: Large quantization error (expected < 0.5)\n");
        return 1;
    } else {
        printf("\n✅ TEST PASSED: Quantization works correctly\n");
        return 0;
    }
}
