// header file for backward.cu 
// declares functions implemented in backward.cu so they can be used in computeGaussNorm.cu

#ifndef BACKWARD_H
#define BACKWARD_H

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>


// Backward pass for the conversion of scale and rotation to norms of Gaussian
__device__ void computeNormBackward(
    int idx, 
    const glm::vec3 scale, 
    float mod, 
    const glm::vec4 rot, 
    const glm::vec3* dL_dnorms, 
    glm::vec3* dL_dscales, 
    glm::vec4* dL_drots);

// Kernel that works on batches of inputs of scales and rotations for backward pass
__global__ void computeGaussNormsBackwardKernel(
    const glm::vec3* scales, 
    const glm::vec4* rotations, 
    const glm::vec3* dL_dnorms, 
    glm::vec3* dL_dscales, 
    glm::vec4* dL_drots, 
    float scale_modifier, 
    int num_gaussians);

// Wrapper function to launch the backward kernel
cudaError_t computeGaussNormsBackward(
    const glm::vec3* scales, 
    const glm::vec4* rotations, 
    const glm::vec3* dL_dnorms, 
    glm::vec3* dL_dscales, 
    glm::vec4* dL_drots, 
    float scale_modifier, 
    int num_gaussians);

#endif // BACKWARD_H