// contains backward pass of norm operation 
#include "backward.h"

// Utility function to compute the index of the minimum element
__device__ int findMinIndex(const glm::vec3& v){
    if (v.x < v.y) {
        return (v.x < v.z) ? 0 : 2;
    } else {
        return (v.y < v.z) ? 1 : 2;
    }
}

// Backward pass for the conversion of scale and rotation to a 
// norms of gaussian
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const glm::vec3* dL_dnorms, glm::vec3* dL_dscales, glm::vec4* dL_drots) 
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

	// Compute the scaling factors and find the minimum index
    glm::vec3 scaled = mod * scale;
    int minIndex = findMinIndex(scaled);

	// gradient w.r.t. the selected basis vector after rotation
	glm::vec3 dL_dselectedVec = glm::vec3(dL_dnorms[3 * idx + 0], dL_dnorms[3 * idx + 1], dL_dnorms[3 * idx + 2]);

	// Backpropagate the gradients through the rotation
	glm::vec3 dL_dselectedVecRot = Rt * dL_dselectedVec;

	// backpropagate the gradient through scale modification 
	glm::vec3 gradientThroughScale = dL_dselectedVecRot; 

	// adjust gradient based on modulation and selection 
	glm::vec3 dL_dscale = {0.0f, 0.0f, 0.0f}; // initialize 
	dL_dscale[minIndex] = gradientThroughScale[minIndex] * mod;

	// compute gradient with respect to mod 
	float dL_dmod = glm::dot(gradientThroughScale, scale);

	// computed gradient back to output pointers
	dL_dscales[idx] = dL_dscale;

	// Computing gradients w.r.t. rotation `rot` involves quaternion calculus.
	glm::vec4 dL_dq = calculateQuaternionGradient(Rt, dL_dselectedVecRot);
	// You need to account for how changes in each quaternion component affect `R` and thus `norm`.
	// The provided example already illustrates calculating dL_dq for quaternion components,
	// but ensure it matches the specifics of how `R` is used in your forward pass.

	// Finalize gradients for `rot`
	// Adjust the provided dL_dq calculation to your specific needs, considering the influence of quaternion changes on `norm`.

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}