
#include "io.hpp"
#include <fstream>
#include <cuda_runtime.h>
#include <iostream>


namespace prism {

Buffer::~Buffer() {
    if(d)
    cudaFree(d);
    if(h)
    cudaFreeHost(h);
}

void Buffer::ensure_host_capacity(size_t numBytes) {
    if (h_capacity >= numBytes && h != nullptr) {
        return;
    }
    if (h) {
        CHECK_CUDA(cudaFreeHost(h));
        h = nullptr;
    }
    CHECK_CUDA(cudaMallocHost(&h, numBytes));
    h_capacity = numBytes;
}

void Buffer::ensure_device_capacity(size_t numBytes) {
    if (d_capacity >= numBytes && d != nullptr) {
        return;
    }
    if (d) {
        CHECK_CUDA(cudaFree(d));
        d = nullptr;
    }
    CHECK_CUDA(cudaMalloc(&d, numBytes));
    d_capacity = numBytes;
}

template<itype file_type>
void Buffer::load_fromfile(const std::string filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }
    
    file.seekg(0, std::ios::end);
    auto length = file.tellg();
    auto file_length = static_cast<size_t>(length);
    file.seekg(0, std::ios::beg);
    if(file_type == ori_File) {
        if (file_length < bytes) {
            std::cerr << "Error: expected " << bytes << " bytes, but file has " << file_length << " bytes." << std::endl;
            std::cerr << "File read incomplete or corrupted." << std::endl;
            exit(1);
        }
    }
    else  bytes = file_length;
    ensure_host_capacity(bytes);
    ensure_device_capacity(bytes);
    // file.seekg(bytes, std::ios::beg);
    file.read(reinterpret_cast<char*>(h), bytes);
    file.close();
    // dl = DataLocation::OnHost;
}

void Buffer::unload_tofile(const std::string filename, typetofile tf) {
    if(tf == typetofile::deviceTofile) {
        ensure_host_capacity(bytes);
        D2H();
    }

    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "can't open file" << std::endl;
        return ;
    }
    outfile.write(reinterpret_cast<char*>(h), bytes);
    std::cout <<  "decompressed file written: " << filename << '\n';
    outfile.close();
}

void Buffer::unload_tofile(const std::string filename, long long numBytes, typetofile tf) {
    if(tf == typetofile::deviceTofile) {
        ensure_host_capacity(numBytes);
        D2H(numBytes);
    }

    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "can't open file" << std::endl;
        return ;
    }
    std::cout <<  "compressed file written: " << filename << '\n';
    outfile.write(reinterpret_cast<char*>(h), numBytes);
    outfile.close();
}

void Buffer::D2H(long long numBytes) {
    if(numBytes == -1)
        numBytes = bytes;
    CHECK_CUDA(cudaMemcpy(h, d, numBytes, cudaMemcpyDeviceToHost));
}

void Buffer::D2H_cudaasync(void* stream, long long numBytes) {
    if(numBytes == -1)
        numBytes = bytes;
    CHECK_CUDA(cudaMemcpyAsync(h, d, numBytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream));
}

void Buffer::H2D(long long numBytes) {
    if(numBytes == -1)
        numBytes = bytes;
    CHECK_CUDA(cudaMemcpy(d, h, numBytes, cudaMemcpyHostToDevice));
}

void Buffer::H2D_cudaasync(void* stream, long long numBytes)
{
    if(numBytes == -1)
        numBytes = bytes;
    CHECK_CUDA(cudaMemcpyAsync(d, h, numBytes, cudaMemcpyHostToDevice, (cudaStream_t)stream));
}

template void Buffer::load_fromfile<ori_File>(const std::string);
template void Buffer::load_fromfile<cmp_File>(const std::string);

void Bitplane::bitplane_malloc() {
    calculate_aligned_buffer_size(8);

    cudaMalloc(&aligned_strides_d, sizeof(int) * 4);
    cudaMalloc(&prefix_sum_d, sizeof(int) * 4);
    cudaMalloc(&aligned_prefix_sum_d, sizeof(int) * 4);

    cudaMalloc(&d, aligned_size);
    cudaMemset(d, 0, aligned_size);

    cudaMemcpy(aligned_strides_d, aligned_strides, sizeof(int) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(prefix_sum_d, prefix_nums, sizeof(int) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(aligned_prefix_sum_d, aligned_prefix_nums, sizeof(int) * 4, cudaMemcpyHostToDevice);
}

Bitplane::~Bitplane() {
    if (aligned_strides_d) CHECK_CUDA(cudaFree(aligned_strides_d));
    if (prefix_sum_d) CHECK_CUDA(cudaFree(prefix_sum_d));
    if (aligned_prefix_sum_d) CHECK_CUDA(cudaFree(aligned_prefix_sum_d));
}

void Bitplane::calculate_aligned_buffer_size(size_t alignment = 8) {
    const int LEVEL = 4;
    const int segments_per_level = 32;
    aligned_size = 0;
    
    size_t segment_size = strides[3];
    aligned_strides[3] = ((segment_size + alignment - 1) / alignment) * alignment;

    for (int l = LEVEL - 1; l >=0; --l) {
        size_t segment_size = strides[l];
        aligned_strides[l] = ((segment_size + alignment - 1) / alignment) * alignment;
        aligned_prefix_nums[l] = aligned_strides[l] * segments_per_level;
        aligned_size += aligned_prefix_nums[l];
    }
    aligned_prefix_nums[0] =  aligned_prefix_nums[1] +  aligned_prefix_nums[2] +  aligned_prefix_nums[3];
    aligned_prefix_nums[1] =  aligned_prefix_nums[2] +  aligned_prefix_nums[3];
    aligned_prefix_nums[2] =  aligned_prefix_nums[3];
    aligned_prefix_nums[3] =  0;
    // printf("aligned_size:%d\n",aligned_size);
    // printf("\n");
    // for (int i = 0; i < LEVEL; ++i) {
    //     printf("prefix_nums[%d]: %d ", i, prefix_nums[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < LEVEL; ++i) {
    //     printf("aligned_prefix_nums[%d]: %u ", i, aligned_prefix_nums[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < LEVEL; ++i) {
    //     printf("aligned_strides[%d]: %u ", i, aligned_strides[i]);
    // }
    // printf("\n\n");
    // printf("Aligned buffer size: %zu bytes\n", aligned_size);
}

}
