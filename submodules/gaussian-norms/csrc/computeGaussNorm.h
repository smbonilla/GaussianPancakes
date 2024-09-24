// header file for computeGaussNorm.cu 
// declares wrapper functions for forward and backward passes

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>

#include <cuda_runtime_api.h>

// Declarations for the utility function and the main compute functions
void checkTensorProperties(const torch::Tensor& tensor, const std::string& name);

// Declaration for the forward computation
torch::Tensor computeGaussNormForward(
    torch::Tensor scales,
    torch::Tensor rotations,
    float scale_modifier);

// Declaration for the backward computation
std::tuple<torch::Tensor, torch::Tensor> computeGaussNormBackward(
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor dL_dnorms,
    float scale_modifier);