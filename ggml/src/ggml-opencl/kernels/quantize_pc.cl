#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// Per-Channel Q4_0 Quantization for KV Cache
// Each channel (embedding dimension) has one scale value
//------------------------------------------------------------------------------

#define QK4_0 32

//------------------------------------------------------------------------------
// F32 -> Per-Channel Q4_0 quantization kernel
// Input: F32 tensor [n_embd, kv_size]
// Scales: F32 array [n_embd] - pre-computed from calibration
// Output: Quantized data [n_embd * kv_size * 0.5 bytes]
//------------------------------------------------------------------------------

kernel void kernel_quantize_per_channel_q4_0(
        global const float * src,      // [n_embd, kv_size]
        global const float * scales,   // [n_embd] - per-channel scales
        global uchar * dst,            // output quantized data
        int n_embd,                    // number of channels
        int kv_size                    // sequence length
) {
    const int channel_id = get_global_id(0);
    
    if (channel_id >= n_embd) {
        return;
    }
    
    // Get scale for this channel
    const float scale = scales[channel_id];
    const float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 0.0f;
    
    // Source pointer for this channel
    global const float * src_channel = src + channel_id * kv_size;
    
    // Destination pointer for this channel
    // Layout: all channels stored sequentially, each channel has kv_size/2 bytes
    global uchar * dst_channel = dst + channel_id * (kv_size / 2);
    
    // Quantize this channel's values (2 values per byte)
    for (int i = 0; i < kv_size / 2; i++) {
        const float v0 = src_channel[i * 2];
        const float v1 = src_channel[i * 2 + 1];
        
        // Quantize to 4-bit: range [-8, 7]
        const int q0 = (int)(v0 * inv_scale + 8.5f);
        const int q1 = (int)(v1 * inv_scale + 8.5f);
        
        // Clamp to [0, 15]
        const uchar qi0 = (uchar)clamp(q0, 0, 15);
        const uchar qi1 = (uchar)clamp(q1, 0, 15);
        
        // Pack two 4-bit values into one byte
        dst_channel[i] = qi0 | (qi1 << 4);
    }
}

//------------------------------------------------------------------------------
// Per-Channel Q4_0 -> F32 dequantization kernel
//------------------------------------------------------------------------------

kernel void kernel_dequantize_per_channel_q4_0(
        global const uchar * src,      // quantized data
        global const float * scales,   // [n_embd] - per-channel scales
        global float * dst,            // [n_embd, kv_size]
        int n_embd,
        int kv_size
) {
    const int channel_id = get_global_id(0);
    
    if (channel_id >= n_embd) {
        return;
    }
    
    // Get scale for this channel
    const float scale = scales[channel_id];
    
    // Source pointer for this channel
    global const uchar * src_channel = src + channel_id * (kv_size / 2);
    
    // Destination pointer for this channel
    global float * dst_channel = dst + channel_id * kv_size;
    
    // Dequantize this channel's values
    for (int i = 0; i < kv_size / 2; i++) {
        const uchar packed = src_channel[i];
        
        // Extract two 4-bit values
        const int q0 = (packed & 0x0F) - 8;
        const int q1 = (packed >> 4) - 8;
        
        // Dequantize
        dst_channel[i * 2] = (float)q0 * scale;
        dst_channel[i * 2 + 1] = (float)q1 * scale;
    }
}
