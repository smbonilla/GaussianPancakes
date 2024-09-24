
# 
# Date:         25.01.2024
# Author:       Sierra Bonilla
# Description:  This file contains obj utils (meshlab -> obj w color)
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# 

import numpy as np
from utils.graphics_utils import BasicPointCloud 

def read_obj(obj_path):
    """Reads a .obj file and returns a numpy array with vertices and a numpy array with vertex colors.

    params:
        obj_path (str): path to .obj file

    returns:
        tuple: (vertices, vertex_colors) where vertices is a numpy array of shape (N, 3) and vertex_colors is a numpy array of shape (N, 3)
    """
    vertices = []
    vertex_colors = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(v) for v in line.split()[1:4]])
                vertex_colors.append([float(v) for v in line.split()[4:7]])
    return np.array(vertices), np.array(vertex_colors)

def downsample_obj(pointcloud: BasicPointCloud, voxel_size: float) -> BasicPointCloud:
    voxel_indices = np.floor(pointcloud.points / voxel_size).astype(int)

    voxel_indices = voxel_indices[:, 0] + voxel_indices[:, 1] * 1e3 + voxel_indices[:, 2] * 1e6
    unique_voxel_indices, inv = np.unique(voxel_indices, return_inverse=True)

    downsampled_points = np.array([pointcloud.points[inv == i].mean(axis=0) for i in range(len(unique_voxel_indices))])
    downsampled_colors = np.array([pointcloud.colors[inv == i].mean(axis=0) for i in range(len(unique_voxel_indices))])
    downsampled_normals = np.array([pointcloud.normals[inv == i].mean(axis=0) for i in range(len(unique_voxel_indices))])

    return BasicPointCloud(points=downsampled_points, colors=downsampled_colors, normals=downsampled_normals)
