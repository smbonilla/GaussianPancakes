// contains backward pass of norm operation 
#include "backward.h"
#include <stdio.h>
#include "utils.h"


// Utility function to compute the index of the minimum element
// __device__ int findMinIndex(const glm::vec3& v){
//     if (v.x < v.y) {
//         return (v.x < v.z) ? 0 : 2;
//     } else {
//         return (v.y < v.z) ? 1 : 2;
//     }
// }

// Backward pass for the conversion of scale and rotation to a 
// norms of gaussian
__device__ void computeNormBackward(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const glm::vec3* dL_dnorms, glm::vec3* dL_dscales, glm::vec4* dL_drots) 
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// transposing the rotation matrix 
	glm::mat3 Rt = glm::transpose(R);

	// Find the minimum index
    int minIndex = findMinIndex(scale);

	// Grab the gradient of the loss w.r.t. the norm (current index)
	glm::vec3 dL_dnorm = dL_dnorms[idx];
	
	// Calculate dL_dscales

	// Backpropagate the gradients through the rotation
	glm::vec3 dL_dselectedVecRot = Rt * dL_dnorm;

	// backpropagate the gradient through scale modification 
	glm::vec3 gradientThroughScale = dL_dselectedVecRot; 

	// adjust gradient based on modulation and selection 
	glm::vec3 dL_dscale = {0.0f, 0.0f, 0.0f}; // initialize 
	dL_dscale[minIndex] = gradientThroughScale[minIndex] * mod;

	// compute gradient with respect to mod - not used ever?
	float dL_dmod = glm::dot(gradientThroughScale, scale);

	// computed gradient back to output pointers
	dL_dscales[idx] = dL_dscale;

	// Calculate dL_drots

	// initialize d(norm)/dr, d(norm)/dx, d(norm)/dy, d(norm)/dz
	glm::vec3 dnorm_dr = {0.0f, 0.0f, 0.0f};
	glm::vec3 dnorm_dx = {0.0f, 0.0f, 0.0f};
	glm::vec3 dnorm_dy = {0.0f, 0.0f, 0.0f};
	glm::vec3 dnorm_dz = {0.0f, 0.0f, 0.0f};

	// Calculate dnorm_dq depending on which index is selected
	// dnorm_dq = {d(norm)/dr, d(norm)/dx, d(norm)/dy, d(norm)/dz}
	switch (minIndex)
	{
		case 1:
			// Case 1 -> norm = {1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y)}
			dnorm_dr = {0.0f, -2.0f * z, 2.0f * y};
			dnorm_dx = {0.0f, 2.0f * y, 2.0f * z};
			dnorm_dy = {-4.0f * y, 2.0f * x, -2.0f * r};
			dnorm_dz = {-4.0f * z, 2.0f * r, 2.0f * x};
			
			break;
		
		case 2:
			// Case 2 -> norm = {2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x)}
			dnorm_dr = {-2.0f * z, 0.0f, 2.0f * x};
			dnorm_dx = {2.0f * y, -4.0f * x, 2.0f * r};
			dnorm_dy = {2.0f * x, 0.0f, 2.0f * z};
			dnorm_dz = {-2.0f * r, -4.0f * z, 2.0f * y};

			break;

		case 3:
			// Case 3 -> norm = {2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y)}
			dnorm_dr = {2.0f * y, -2.0f * x, 0.0f};
			dnorm_dx = {2.0f * z, -2.0f * r, -4.0f * x};
			dnorm_dy = {2.0f * r, 2.0f * z, -4.0f * y};
			dnorm_dz = {2.0f * x, 2.0f * y, 0.0f};

			break;
	}

	// dL_dq = dL_dnorm * dnorm_dq
	glm::vec4 dL_dq = {0.0f, 0.0f, 0.0f, 0.0f};
	dL_dq.x = glm::dot(dL_dnorm, dnorm_dr);
	dL_dq.y = glm::dot(dL_dnorm, dnorm_dx);
	dL_dq.z = glm::dot(dL_dnorm, dnorm_dy);
	dL_dq.w = glm::dot(dL_dnorm, dnorm_dz);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}

// kernel that works on batches of inputs of scales and rotations 
__global__ void computeGaussNormsBackwardKernel(
	const glm::vec3* scales, 
	const glm::vec4* rotations,
	const glm::vec3* dL_dnorms,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots,
	float scale_modifier, 
	int num_gaussians) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_gaussians) {
        return;
    }

	glm::vec3 scale = scales[idx];
	glm::vec4 rot = rotations[idx];

	computeNormBackward(idx, scale, scale_modifier, rot, dL_dnorms, dL_dscales, dL_drots);


	}

// Wrapper function to launch the backward kernel
cudaError_t computeGaussNormsBackward(
    const glm::vec3* scales, 
    const glm::vec4* rotations, 
    const glm::vec3* dL_dnorms, 
    glm::vec3* dL_dscales, 
    glm::vec4* dL_drots, 
    float scale_modifier, 
    int num_gaussians) {

    int threadsPerBlock = 128;
    int blocksPerGrid = (num_gaussians + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the backward kernel
    computeGaussNormsBackwardKernel<<<blocksPerGrid, threadsPerBlock>>>(
        scales, rotations, dL_dnorms, dL_dscales, dL_drots, scale_modifier, num_gaussians);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in computeGaussNormsBackward: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Synchronize the device to ensure all the computation is done
    // This is typically used for debugging and might be omitted in production code
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
    }

    return err;
}