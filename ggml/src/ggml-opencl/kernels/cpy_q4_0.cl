#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// Q4_0 quantization/dequantization for KV cache
// Based on CUDA implementation in cpy.cu
//------------------------------------------------------------------------------

#define QK4_0 32

// Q4_0 block structure
typedef struct {
    half d;              // delta (scale)
    uchar qs[QK4_0 / 2]; // nibbles / quants (16 bytes for 32 values)
} block_q4_0;

// Helper function to quantize one block: F32[32] -> Q4_0
static void cpy_blck_f32_q4_0(global const float * xi, global block_q4_0 * dsti) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = xi[j];
        if (amax < fabs(v)) {
            amax = fabs(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f/d : 0.0f;

    dsti->d = (half)d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = xi[j] * id;
        const float x1 = xi[QK4_0/2 + j] * id;

        const uchar xi0 = min((uchar)15, (uchar)max(0, (int)(x0 + 8.5f)));
        const uchar xi1 = min((uchar)15, (uchar)max(0, (int)(x1 + 8.5f)));

        dsti->qs[j] = xi0 | (xi1 << 4);
    }
}

// Helper function to dequantize one block: Q4_0 -> F32[32]
static void cpy_blck_q4_0_f32(global const block_q4_0 * xi, global float * dsti) {
    const float d = (float)(xi->d);

    for (int j = 0; j < QK4_0/2; ++j) {
        const uchar vi = xi->qs[j];
        
        const int vi0 = (vi & 0x0F) - 8;
        const int vi1 = (vi >> 4) - 8;

        dsti[j] = vi0 * d;
        dsti[QK4_0/2 + j] = vi1 * d;
    }
}

//------------------------------------------------------------------------------
// F32 -> Q4_0 quantization kernel
//------------------------------------------------------------------------------

kernel void kernel_cpy_f32_q4_0(
        global float * src0,
        ulong offset0,
        global uchar * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    global char * src_base = (global char*)src0 + offset0;
    global char * dst_base = (global char*)dst + offsetd;

    // Simple approach: treat everything as 1D
    // Total elements in source
    const int total_src_elements = ne00 * ne01 * ne02 * ne03;
    const int total_blocks = total_src_elements / QK4_0;
    
    // Get global work-item ID
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    const int gid_z = get_global_id(2);
    const int global_size_x = get_global_size(0);
    const int global_size_y = get_global_size(1);
    
    // Convert to 1D index
    const int global_id = gid_z * (global_size_x * global_size_y) + gid_y * global_size_x + gid_x;
    const int global_size = get_global_size(0) * get_global_size(1) * get_global_size(2);
    
    // Each work-item processes one or more blocks
    for (int block_idx = global_id; block_idx < total_blocks; block_idx += global_size) {
        // Source: calculate 4D position from flat index
        const int elem_idx = block_idx * QK4_0;
        
        // Bounds check
        if (elem_idx >= total_src_elements) {
            continue;
        }
        
        // elem_idx is the starting element index (not block index)
        // We need to read QK4_0 consecutive elements starting from elem_idx
        const int i03 = elem_idx / (ne00 * ne01 * ne02);
        const int i02 = (elem_idx - i03 * ne00 * ne01 * ne02) / (ne00 * ne01);
        const int i01 = (elem_idx - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01) / ne00;
        const int i00 = elem_idx - i03 * ne00 * ne01 * ne02 - i02 * ne00 * ne01 - i01 * ne00;
        
        // Bounds check for indices
        if (i03 >= ne03 || i02 >= ne02 || i01 >= ne01 || i00 + QK4_0 > ne00) {
            continue;
        }
        
        const size_t src_offset = (size_t)i00 * nb00 + (size_t)i01 * nb01 + (size_t)i02 * nb02 + (size_t)i03 * nb03;
        
        // Destination: flat layout
        const size_t dst_offset = (size_t)block_idx * nb0;
        
        // Quantize
        global const float * src_ptr = (global const float *)(src_base + src_offset);
        global block_q4_0 * dst_ptr = (global block_q4_0 *)(dst_base + dst_offset);
        
        cpy_blck_f32_q4_0(src_ptr, dst_ptr);
    }
}

//------------------------------------------------------------------------------
// Q4_0 -> F32 dequantization kernel
//------------------------------------------------------------------------------

kernel void kernel_cpy_q4_0_f32(
        global uchar * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    global char * src_base = (global char*)src0 + offset0;
    global char * dst_base = (global char*)dst + offsetd;

    // Simple approach: treat everything as 1D
    const int total_src_elements = ne00 * ne01 * ne02 * ne03;
    const int total_blocks = total_src_elements / QK4_0;
    
    // Get global work-item ID
    const int gid_x = get_global_id(0);
    const int gid_y = get_global_id(1);
    const int gid_z = get_global_id(2);
    const int global_size_x = get_global_size(0);
    const int global_size_y = get_global_size(1);
    
    // Convert to 1D index
    const int global_id = gid_z * (global_size_x * global_size_y) + gid_y * global_size_x + gid_x;
    const int global_size = get_global_size(0) * get_global_size(1) * get_global_size(2);
    
    // Each work-item processes one or more blocks
    for (int block_idx = global_id; block_idx < total_blocks; block_idx += global_size) {
        // Bounds check
        const int elem_idx = block_idx * QK4_0;
        if (elem_idx >= total_src_elements) {
            continue;
        }
        
        // Source: flat Q4_0 layout
        const size_t src_offset = (size_t)block_idx * nb00;
        
        // Destination: calculate 4D position from flat index
        const int i3 = elem_idx / (ne0 * ne1 * ne2);
        const int i2 = (elem_idx - i3 * ne0 * ne1 * ne2) / (ne0 * ne1);
        const int i1 = (elem_idx - i3 * ne0 * ne1 * ne2 - i2 * ne0 * ne1) / ne0;
        const int i0 = elem_idx - i3 * ne0 * ne1 * ne2 - i2 * ne0 * ne1 - i1 * ne0;
        
        // Bounds check for destination indices
        if (i3 >= ne3 || i2 >= ne2 || i1 >= ne1 || i0 + QK4_0 > ne0) {
            continue;
        }
        
        const size_t dst_offset = (size_t)i0 * nb0 + (size_t)i1 * nb1 + (size_t)i2 * nb2 + (size_t)i3 * nb3;
        
        // Dequantize
        global const block_q4_0 * src_ptr = (global const block_q4_0 *)(src_base + src_offset);
        global float * dst_ptr = (global float *)(dst_base + dst_offset);
        
        cpy_blck_q4_0_f32(src_ptr, dst_ptr);
    }
}
