
#include "forward.h"
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

// contains forward pass of norm operation 
__device__ void computeNorm(const glm::vec3 scale, float mod, const glm::vec4 rot, glm::vec3& norm)
{

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

    // Compute the scaling factors and find the minimum index
    glm::vec3 scaled = mod * scale;
    int minIndex = findMinIndex(scaled);

    // Create a 3x3 identity matrix for basis vectors 
    glm::mat3 basis = glm::mat3(1.0f);

    // Select the basis vector corresponding to the minimum scale axis 
    glm::vec3 selectedVector = basis[minIndex];

    // Apply rotation to the selected vector to compute surface normal
    //norm = R * selectedVector;
    // DIrectly select the correct column from R absed on minIndex
    switch(minIndex) {
        case 0:
            norm = glm::vec3(R[0][0], R[1][0], R[2][0]); // First column for X-axis
            break;
        case 1:
            norm = glm::vec3(R[0][1], R[1][1], R[2][1]); // Second column for Y-axis
            break;
        case 2:
            norm = glm::vec3(R[0][2], R[1][2], R[2][2]); // Third column for Z-axis
            break;
    }
}

// kernel that works on batches of inputs of scales and rotations 
__global__ void computeGaussNormsKernel(
    const glm::vec3* scales, 
    const glm::vec4* rotations,
    float scale_modifier, 
    float* norms,
    int num_gaussians)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_gaussians) {
        return;
    }

    glm::vec3 scale = scales[idx];
    glm::vec4 rot = rotations[idx];
    glm::vec3 norm;

    computeNorm(scale, scale_modifier, rot, norm);

    // store compute norms 
    norms[3 * idx] = norm.x;
    norms[3 * idx + 1] = norm.y;
    norms[3 * idx + 2] = norm.z;
}

// wrapper function to launch the kernel
cudaError_t computeGaussNorms(
    const glm::vec3* scales, 
    const glm::vec4* rotations, 
    float scale_modifier,
    float* norms,
    int num_gaussians)
{
    int threadsPerBlock = 128;
    int blocksPerGrid = (num_gaussians + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel 
    computeGaussNormsKernel<<<blocksPerGrid, threadsPerBlock>>>(scales, rotations, scale_modifier, norms, num_gaussians);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in computeGaussNorms: %s\n", cudaGetErrorString(err));
    }

    // Synchronize the device to ensure all the computation is done
    // only for debugging purposes
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Cuda error after synchronization: %s\n", cudaGetErrorString(err));
    }

    // return the last error encountered
    return err;
}
