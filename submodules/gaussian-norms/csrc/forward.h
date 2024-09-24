// header file for forward.cu 
// declares functions implented in forward.cu for use in computeGaussNorm.cu 
#ifndef FORWARD_H
#define FORWARD_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

// Utility function to compute the index of the minimum element
__device__ int findMinIndex(const glm::vec3& v);

// Contains the forward pass of norm operation
__device__ void computeNorm(const glm::vec3 scale, float mod, const glm::vec4 rot, glm::vec3& norm);

// Kernel that works on batches of inputs of scales and rotations
__global__ void computeGaussNormsKernel(
    const glm::vec3* scales,
    const glm::vec4* rotations,
    float scale_modifier,
    float* norms,
    int num_gaussians);

// Wrapper function to launch the kernel
cudaError_t computeGaussNorms(
    const glm::vec3* scales,
    const glm::vec4* rotations,
    float scale_modifier,
    float* norms,
    int num_gaussians);

#endif