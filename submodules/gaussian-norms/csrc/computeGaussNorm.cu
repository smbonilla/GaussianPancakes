// integrates forward and backward computations - include forward.h and backward.h
// functions defined here wrap around forward and backward kernels 
// this will be exposed to 'ext.cpp'
#include "computeGaussNorm.h"
#include "forward.h"
#include "backward.h"
#include "torch/extension.h"
#include <vector>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <tuple>

// utility function for tensor checks
void checkTensorProperties(const torch::Tensor& t, const std::string& name) {
  if (!t.is_contiguous()) {
    throw std::invalid_argument(name + " tensor must be contiguous");
  }
  if (!t.is_cuda()) {
    throw std::invalid_argument(name + " tensor must be on CUDA");
  }
}

torch::Tensor computeGaussNormForward(torch::Tensor scales, torch::Tensor rotations, float scale_modifier) {
    checkTensorProperties(scales, "Scales");
    checkTensorProperties(rotations, "Rotations");

    auto options = torch::TensorOptions().dtype(scales.dtype()).device(scales.device());
    auto norms = torch::empty({scales.size(0), 3}, options);

    const auto* scales_ptr = reinterpret_cast<const glm::vec3*>(scales.data_ptr<float>());
    const auto* rotations_ptr = reinterpret_cast<const glm::vec4*>(rotations.data_ptr<float>());
    auto* norms_ptr = norms.data_ptr<float>();

    int num_gaussians = scales.size(0);
    int threadsPerBlock = 128;
    int blocksPerGrid = (num_gaussians + threadsPerBlock - 1) / threadsPerBlock;

    computeGaussNormsKernel<<<blocksPerGrid, threadsPerBlock>>>(scales_ptr, rotations_ptr, scale_modifier, norms_ptr, num_gaussians);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in computeGaussNormForward: " + std::string(cudaGetErrorString(err)));
    }

    return norms;
}

std::tuple<torch::Tensor, torch::Tensor> computeGaussNormBackward(torch::Tensor scales, torch::Tensor rotations, torch::Tensor dL_dnorms, float scale_modifier) {
    checkTensorProperties(scales, "Scales");
    checkTensorProperties(rotations, "Rotations");
    checkTensorProperties(dL_dnorms, "dL_dNorms");

    auto options = scales.options();
    auto dL_dscales = torch::zeros_like(scales, options);
    auto dL_drots = torch::zeros_like(rotations, options);

    const auto* scales_ptr = reinterpret_cast<const glm::vec3*>(scales.data_ptr<float>());
    const auto* rotations_ptr = reinterpret_cast<const glm::vec4*>(rotations.data_ptr<float>());
    const auto* dL_dnorms_ptr = reinterpret_cast<const glm::vec3*>(dL_dnorms.data_ptr<float>());
    auto* dL_dscales_ptr = reinterpret_cast<glm::vec3*>(dL_dscales.data_ptr<float>());
    auto* dL_drots_ptr = reinterpret_cast<glm::vec4*>(dL_drots.data_ptr<float>());

    int num_gaussians = scales.size(0);
    int threadsPerBlock = 128;
    int blocksPerGrid = (num_gaussians + threadsPerBlock - 1) / threadsPerBlock;

    computeGaussNormsBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(scales_ptr, rotations_ptr, dL_dnorms_ptr, dL_dscales_ptr, dL_drots_ptr, scale_modifier, num_gaussians);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error in computeGaussNormBackward: " + std::string(cudaGetErrorString(err)));
    }

    return std::make_tuple(dL_dscales, dL_drots);
}