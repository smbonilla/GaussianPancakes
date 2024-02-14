// integrates forward and backward computations - include forward.h and backward.h
// functions defined here wrap around forward and backward kernels 
// this will be exposed to 'ext.cpp'
#include "computeGaussNorm.h"
#include "forward.h"
#include "backward.h"
#include "torch/extension.h"
#include <vector>
#include <iostream>

// utility function for tensor checks
void checkTensorProperties(const torch::Tensor& t, const std::string& name) {
  if (!t.is_contiguous()) {
    throw std::invalid_argument(name + " tensor must be contiguous");
  }
  if (!t.is_cuda()) {
    throw std::invalid_argument(name + " tensor must be on CUDA");
  }
}

torch::Tensor computeGaussNormForward(
    torch::Tensor scales,
    torch::Tensor rotations,
    float scale_modifier){
    
    // Ensuring the input tensors are on CUDA and contiguous
    checkTensorProperties(scales, "scales");
    checkTensorProperties(rotations, "rotations");

    // assuming the output tensor norms have the same device and dtype as the scales
    auto options = torch::TensorOptions().dtype(scales.dtype()).device(scales.device());
    auto norms = torch::empty({scales.size(0), 3}, options); 

    // calling the forward kernel
    computeGaussNorms(
        reinterpret_cast<const glm::vec3*>(scales.data_ptr<float>()),
        reinterpret_cast<const glm::vec4*>(rotations.data_ptr<float>()),
        scale_modifier, 
        norms.data_ptr<float>(),
        scales.size(0)
    );

    return norms;
    }