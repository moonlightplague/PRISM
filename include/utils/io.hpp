#pragma once

#include "parameters.hpp"
#include <fstream>
#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>

void* readData(char* srcFilePath);

namespace prism{

enum class typetofile {
    hostTofile,
    deviceTofile
};


typedef struct Buffer {
    prism_dtype dtype;
    int tsize, ndim{-1};
    size_t len{1}, bytes{1};
    uint32_t lx{1}, ly{1}, lz{1};
    size_t sty{1}, stz{1};  // stride
    int align{0};
    //DataLocation dl;
    void* d{nullptr};
    void* h{nullptr};
    size_t d_capacity{0};
    size_t h_capacity{0};
    Buffer() {};
    ~Buffer();
    template<itype file_type>
    void load_fromfile(const std::string filename);
    void unload_tofile(const std::string filename, typetofile tf = typetofile::hostTofile);
    void unload_tofile(const std::string  filename, long long numBytes,typetofile tf = typetofile::hostTofile);
    void H2D(long long numBytes = -1);
    void D2H(long long numBytes = -1);
    void H2D_cudaasync(void* stream, long long numBytes = -1);
    void D2H_cudaasync(void* stream, long long numBytes = -1);
    void ensure_host_capacity(size_t numBytes);
    void ensure_device_capacity(size_t numBytes);
    
    template<typename D>
    D len3() const {
        return D(lx, ly, lz);
    }
    template<typename D>
    D st3() const {
        return D(1, sty, stz);
    }
    template<typename... Dim>
    Buffer(prism_dtype type, int align_, Dim... args):dtype(type), align(align_) {
        uint32_t tmp[] = { static_cast<uint32_t>(args)... };
        switch(sizeof...(Dim)) {
            // case 4: lw = tmp[3];
            case 3: lz = tmp[2];
            case 2: ly = tmp[1];
            case 1: lx = tmp[0]; break;
            default:
                //printf("Dim error!\n");
                break;
        }
        ndim = 3;
        if (lz == 1) ndim = 2;
        if (ly == 1) ndim = 1;
        tsize = dtype % 10;
        len = lx * ly * lz;
        sty = lx;
        stz = lx * ly;
        bytes = tsize * (len + align);
        CHECK_CUDA(cudaMallocHost(&h, bytes));
        CHECK_CUDA(cudaMalloc(&d, bytes));
        h_capacity = bytes;
        d_capacity = bytes;
        CHECK_CUDA(cudaMemset(d, 0, bytes));
    }

} inBuffer;

struct Bitplane : Buffer {
    int* prefix_sum_d{nullptr};
    int* aligned_prefix_sum_d{nullptr};
    int* aligned_strides_d{nullptr};
    uint32_t lx{1}, ly{1}, lz{1};
    size_t len{1}, aligned_len{1};
    size_t ori_size{1}, aligned_size{1};
    int align{0};
    int strides[4]{0, 0, 0, 0};
    int aligned_strides[4]{0, 0, 0, 0};
    int prefix_nums[4]{0, 0, 0, 0};
    int aligned_prefix_nums[4]{0, 0, 0, 0};
    int compressed_size[4][32];

    Bitplane(int x, int y, int z) : lx(x), ly(y), lz(z) {
        aligned_len = len = lx * ly * lz;
        ori_size = aligned_size = len * 4;
        h_comput<4>();
        bitplane_malloc();
        dtype = U1;
        tsize = 1;
        bytes = aligned_size;
        d_capacity = aligned_size;
    }
    ~Bitplane();
    void bitplane_malloc();
    void calculate_aligned_buffer_size(size_t alignment);


    template <int LEVEL> __forceinline__ void h_comput(){
        dim3 d_size = dim3(lx, ly, lz);
        int level = 0;
        while(level < LEVEL){
            d_size.x = (d_size.x + 1) >> 1;
            d_size.y = (d_size.y + 1) >> 1;
            d_size.z = (d_size.z + 1) >> 1;
            prefix_nums[level] = d_size.x * d_size.y * d_size.z;
            ++level;
        }
        prefix_nums[LEVEL] = 0;
        int anchor_size = prefix_nums[3];

        prefix_nums[0] -=  prefix_nums[3];
        prefix_nums[1] -=  prefix_nums[3];
        prefix_nums[2] -=  prefix_nums[3];
        prefix_nums[3] -=  prefix_nums[3];

        for(int i = LEVEL - 2; i >= 0; --i) {
            align += (8 - ((prefix_nums[i] - prefix_nums[i+1] + align) % 8)) % 8;
            prefix_nums[i] += align;
        }
        // for (int i = 0; i < LEVEL; ++i) {
        //     printf("prefix_nums[%d]: %u ", i, prefix_nums[i]);
        // }
        align += (8 - ((len - anchor_size - prefix_nums[0] + align) % 8)) % 8;
        // printf("%lu %lu \n", len, align);
        strides[3] = (prefix_nums[2] - prefix_nums[3]) >> 3;
        strides[2] = (prefix_nums[1] - prefix_nums[2]) >> 3;
        strides[1] = (prefix_nums[0] - prefix_nums[1]) >> 3;
        strides[0] = (len - anchor_size - prefix_nums[0] + align) >> 3;
        aligned_len += align;
        
    }

    void unload_tofile(const char* filename, typetofile tf) {
        if(tf == typetofile::deviceTofile) {
            ensure_host_capacity(aligned_size);
            CHECK_CUDA(cudaMemcpy(h, d, aligned_size, cudaMemcpyDeviceToHost));
        }

        std::ofstream outfile(filename, std::ios::binary);
        if (!outfile) {
            std::cerr << "can't open file" << std::endl;
            return ;
        }
        outfile.write(reinterpret_cast<char*>(h), aligned_size);
        outfile.close();
    }

};


template<typename T>
struct olBuffer : Buffer { ///outlier
    uint32_t *d_idx{nullptr}, *h_idx{nullptr};
    uint32_t *d_num{nullptr}, h_num{0};

    olBuffer(){}

    template<typename... Dim>
    olBuffer(prism_dtype dtype, Dim... args)
        : Buffer(dtype, args...)
    {
        cudaMalloc(&d_idx, sizeof(uint32_t) * len);
        cudaMalloc(&d_num, sizeof(uint32_t) * 1);
        cudaMemset(d_num, 0x0, sizeof(uint32_t) * 1);
        cudaMallocHost(&h_idx, sizeof(uint32_t) * len);

    }
    
    ~olBuffer() {
        if (d_idx) cudaFree(d_idx);
        if (d_num) cudaFree(d_num);
        if (h_idx) cudaFreeHost(h_idx);
    }
};
}
