
# 
# Date:         25.01.2024
# Author:       Sierra Bonilla
# Description:  This file contains the rnnslam loader 
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# 

from typing import NamedTuple
from PIL import Image

import numpy as np
import os
import sys

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import focal2fov, fov2focal

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    width: int
    height: int

def readRNNSLAMcamerasdef(cam_extrinsics, cam_intrinsics, images_folder, depths_folders):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        depths_path = os.path.join(depths_folder, os.path.basename(extr.name))

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


class RNNSLAM_Dataset(object): # may need to be changed to rnnslam for simulation, in-vivo, and phantom separately b/c camera intrinsics are different
    def __init__(self, source_path, images, depths):
        self.source_path = source_path
        self.images = images
        self.depths = depths

        self.load_meta()
    
    def load_meta(self):
        "Load data from dataset."
        try:
            # Read TUM poses & hard set camera intrinsics 
            cameras_extrinsic_file = os.path.join(self.source_path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.source_path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.source_path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.source_path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        reading_dir = "images" if self.images == None else self.images
        depths_dir = "depths" if self.depths == None else self.depths