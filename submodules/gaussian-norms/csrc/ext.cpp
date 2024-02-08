// exposes functions to Python 

#include <torch/extension.h>
#include "computeGaussNorm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_gauss_norm_forward", &computeGaussNormForward, "Forward pass for Gaussian Norms");
    m.def("compute_gauss_norm_backward", &computeGaussNormBackward, "Backward pass for Gaussian Norms");
}
