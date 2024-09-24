import torch 
from gaussian_norms import compute_gauss_norm
import time

def compute_gauss_norm_pytorch(scaling, r, scale_modifier):
    #norm = r.norm(p=2, dim=1, keepdim=True)
    #q = r / norm
    q = r

    # Precompute repeated operations
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    rx, ry, rz = r*x, r*y, r*z
    
    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - rz)
    R[:, 0, 2] = 2 * (xz + ry)
    R[:, 1, 0] = 2 * (xy + rz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - rx)
    R[:, 2, 0] = 2 * (xz - ry)
    R[:, 2, 1] = 2 * (yz + rx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)

    _, min_index = torch.min(scaling * scale_modifier, dim=1)
    normals = torch.zeros_like(scaling)
    for i, idx in enumerate(min_index):
        normals[i] = R[i, :, idx]

    # Directly select the unit vector for the minimum scale direction
    unit_vectors = torch.eye(3, device=r.device)
    selected_vectors = unit_vectors[min_index]

    # Apply rotation to the selected vectors
    surface_normal = torch.einsum('bij,bj->bi', R, selected_vectors)

    # Check if both methods produce the same result
    assert torch.allclose(surface_normal, normals, atol=1e-6), "Surface normals do not match"

    return normals

def main():
    num_gaussians = 10
    input_scale = torch.randn((num_gaussians, 3), device='cuda', requires_grad=True, dtype=torch.float32)
    input_rotation = torch.randn((num_gaussians, 4), device='cuda', requires_grad=True, dtype=torch.float32)
    scale_modifier = torch.tensor([2.0], device='cuda', requires_grad=False, dtype=torch.float32)

    # Normalize rotation before passing to the CUDA function <- MUST DO THIS when implementing!
    input_rotation_normalized = input_rotation / input_rotation.norm(p=2, dim=1, keepdim=True)

    # Separate computations for CUDA and PyTorch to ensure independent gradient tracking
    input_scale_cuda = input_scale.clone().detach().requires_grad_(True)
    input_rotation_cuda = input_rotation_normalized.clone().detach().requires_grad_(True)
    
    input_scale_pytorch = input_scale.clone().detach().requires_grad_(True)
    input_rotation_pytorch = input_rotation_normalized.clone().detach().requires_grad_(True)

    cuda_start = time.time()

    # Compute normals using CUDA implementation
    cuda_normal = compute_gauss_norm(input_scale_cuda, input_rotation_cuda, scale_modifier)
    loss_cuda = cuda_normal.sum()
    loss_cuda.backward()

    cuda_end = time.time()

    print('CUDA time: ', cuda_end - cuda_start)

    pytorch_start = time.time()

    # Compute normals using PyTorch implementation
    pytorch_normal = compute_gauss_norm_pytorch(input_scale_pytorch, input_rotation_pytorch, scale_modifier)
    loss_pytorch = pytorch_normal.sum()
    loss_pytorch.backward()

    pytorch_end = time.time()

    print('PyTorch time: ', pytorch_end - pytorch_start)

    # print difference between the two methods
    print('Difference in normals: ', torch.norm(cuda_normal - pytorch_normal))
    if input_scale_pytorch.grad is None:
        print('PyTorch Scale gradient is None - dummies')
    else:
        print('Difference in scale gradients: ', torch.norm(input_scale_cuda.grad - input_scale_pytorch.grad))
    difference = torch.abs(input_rotation_pytorch.grad - input_rotation_cuda.grad)
    mean_difference = difference.mean() 
    print('Mean difference in rotation gradients:', mean_difference.item())


if __name__ == "__main__":
    main()