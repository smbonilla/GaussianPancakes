
#include "forward.h"

// contains forward pass of norm operation 
__device__ void computeNorm(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D, glm::vec3& norm)
{
    
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

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

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];

    // Determine the index of the maximum scale factor
    int maxIndex = 0;
    float maxValue = scale.x * mod; 
    if ((scale.y * mod) > maxValue) {
        maxIndex = 1;
        maxValue = scale.y * mod;
    }
    if ((scale.z * mod) > maxValue) {
        maxIndex = 2;
    }

    // Compute the normal vector based on the largest scale factor
    switch (maxIndex) {
        case 0: // Largest scale factor is scale.x
            norm = glm::vec3(cov3D[0], cov3D[1], cov3D[2]);
            break;
        case 1: // Largest scale factor is scale.y
            norm = glm::vec3(cov3D[1], cov3D[3], cov3D[4]);
            break;
        case 2: // Largest scale factor is scale.z
            norm = glm::vec3(cov3D[2], cov3D[4], cov3D[5]);
            break;
    }

    // Normalize the normal vector
    norm = glm::normalize(norm);
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
    float cov3D[6];
    glm::vec3 norm;

    computeNorm(scale, scale_modifier, rot, cov3D, norm);

    // store compute norms 
    norms[3 * idx] = norm.x;
    norms[3 * idx + 1] = norm.y;
    norms[3 * idx + 2] = norm.z;
}

// wrapper function to launch the kernel
void computeGaussNorms(
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

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Cuda error after synchronization: %s\n", cudaGetErrorString(err));
    }
}
