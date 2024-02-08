# imports functions exposed by ext.cpp and wraps them in a torch.autograd.Function

import torch
from torch.autograd import Function
from ._C import compute_gauss_norm_forward, compute_gauss_norm_backward

class ComputeGaussNormFunction(Function):
    @staticmethod
    def forward(ctx, input_scale, input_rotation, scale_modifier):
        """
        Forward pass of gaussian-norms.

        Parameters:
        - ctx: Context object used to stash information for backward computation
        - input_scale: Tensor containing scale parameters for the Gaussian
        - input_rotation: Tensor containing rotation parameters for the Gaussian
        - scale_modifier: Scalar or Tensor modifying the scale in some way

        Returns:
        - normal: (num points, 3) containing the normal of each Gaussian
        """
        normal = compute_gauss_norm_forward(scale_modifier*input_scale, input_rotation)
        ctx.save_for_backward(input_scale, input_rotation, scale_modifier, normal)
        return normal

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of gaussian-norms.

        Parameters:
        - ctx: Context object used to stash information for backward computation
        - grad_output: Tensor containing the gradient of the loss with respect to the output of your forward computation

        Returns:
        - grad_input_scale: Tensor containing the gradient of the loss with respect to the input_scale
        - grad_input_rotation: Tensor containing the gradient of the loss with respect to the input_rotation
        - None: no gradient for scale_modifier
        """

        input_scale, input_rotation, scale_modifier, normal = ctx.saved_tensors
        grad_input_scale, grad_input_rotation = compute_gauss_norm_backward(grad_output, input_scale*scale_modifier, input_rotation, normal)
        return grad_input_scale, grad_input_rotation, None

# Alias for ease of use
compute_gauss_norm = ComputeGaussNormFunction.apply
