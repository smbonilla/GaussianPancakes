#pragma once

#include <glm/glm.hpp>

__device__ inline int findMinIndex(const glm::vec3& v){
    if (v.x < v.y) {
        return (v.x < v.z) ? 0 : 2;
    } else {
        return (v.y < v.z) ? 1 : 2;
    }
}
