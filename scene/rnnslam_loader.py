
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
from scipy.spatial.transform import Rotation as R

import numpy as np
import os
import sys
import math

from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary
from utils.graphics_utils import focal2fov, fov2focal
from scene.colmap_loader import Camera

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
    depth_path: str
    depth_name: str
    width: int
    height: int

class BaseImage(NamedTuple):
    id: int
    qvec: np.array
    tvec: np.array
    camera_id: int
    name: str
    depth_name: str
    xys: np.array
    point3D_ids: np.array

class Image2(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def build_CAMERA(id, model, width, height, params):
    """
    Builds a COLMAP Camera object with the given parameters.
    """
    return Camera(id=id, model=model, width=width, height=height, params=params)

def sort_key(filename):
    main_part, decimal_part, _ = filename.split('.')
    decimal_part = decimal_part.split('.')[0] 
    return int(main_part), int(decimal_part)

def covert_row_of_TUM(TUM_pose_row, camera_id=1, image_name='placeholder', depth_name='placeholder', id=None):
    """
    Convert row of TUM pose to COLMAP pose.

        TUM POSE FORMAT: timestamp tx ty tz qx qy qz qw

        COLMAP POSE FORMAT: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    :return: colmap image object with empty fields for xys and point3D_ids
    """

    # TUM_pose_row = [id, tx, ty, tz, qx, qy, qz, qw]
    if len(str(TUM_pose_row[0]).split('.')) == 3 or str(TUM_pose_row[0]).startswith('1305'):
        id = id
    else:
        id = int(TUM_pose_row[0])
    tx = TUM_pose_row[1]
    ty = TUM_pose_row[2]
    tz = TUM_pose_row[3]
    qx = TUM_pose_row[4]
    qy = TUM_pose_row[5]
    qz = TUM_pose_row[6]
    qw = TUM_pose_row[7]
    
    norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx = qx / norm
    qy = qy / norm
    qz = qz / norm
    qw = qw / norm

    rotmat = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # Check determinant is very close to 1 otherwise assert error
    if np.linalg.det(rotmat) < 0.999 or np.linalg.det(rotmat) > 1.001:
        print('Determinant of Rotation Matrix:')
        print(np.linalg.det(rotmat))
        print('Determinant of Rotation Matrix is not close to 1! Check rotation is proper.')
        assert False

    colmap_rotmat = np.transpose(rotmat) 

    q =  R.from_matrix(colmap_rotmat).as_quat()

    qw = q[3]
    qx = q[0]
    qy = q[1]
    qz = q[2]

    tvec = [tx, ty, tz]

    # convert tvec to colmap format
    tvec = np.dot(-colmap_rotmat,np.transpose(np.array(tvec)))

    # colmap expects (w, x, y, z)
    qvec_new = [qw, qx, qy, qz]

    image = Image2(id=id, qvec=qvec_new, tvec=tvec, camera_id=camera_id, name=image_name, depth_name=depth_name,xys=[], point3D_ids=[]) 

    return image  

def read_rnnslam_extrinsics(extrinsics_file, image_files, depth_files):
    """
    expecting extrinsics file to be txt file with format:

    %% TUM pose format
    % format: ’timestamp tx ty tz qx qy qz qw
    % [R_wc, t_wc] from camera to the world: Pw = R_wc*Pc+t_wc
    """
    try:
        RNNSLAM_txt = np.loadtxt(extrinsics_file)

        assert len(RNNSLAM_txt) == len(image_files), "Number of images and extrinsics do not match!"
        assert len(RNNSLAM_txt) == len(depth_files), "Number of depths and extrinsics do not match!"

        images_dict = {}
        for idx, row in enumerate(RNNSLAM_txt): 
            if not math.isnan(row[0]):
                image = covert_row_of_TUM(row, image_name=image_files[idx], depth_name=depth_files[idx], id=idx)
                images_dict[image.id] = image
    except Exception as e:
        print('Error reading extrinsics file: {}'.format(e))

    return images_dict

def read_rnnslam_intrinsics(intrinsics_file):
    """
    creating camera based on if phantom, in-vivo, or simulation
    """
    return read_intrinsics_text(intrinsics_file)

def readRNNSIM(extrinsics_file, intrinsics_file, images_folder, depths_folder):
    # all image files that are png or jpg in images_folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.png') or f.endswith('.jpg')]
    depth_files = [f for f in os.listdir(depths_folder) if f.endswith('.png') or f.endswith('.jpg')]

    # if image files are of the format '1_color.png' 
    if len(image_files[0].split("_")) == 2:
        image_files.sort(key=lambda f: int(f.split("_")[0]))
        depth_files.sort(key=lambda f: int(f.split("_")[0]))
    # if image files are of the format 'frame018250.jpg' or 'frame018251.jpg' 
    elif len(image_files[0].split("frame")) == 2:
        image_files.sort(key=lambda f: int(f.split("frame")[1].split(".")[0]))
        depth_files.sort(key=lambda f: int(f.split(".")[0]))
    # if image files are of the format '1.png'
    elif len(image_files[0].split(".")) == 2:
        image_files.sort(key=lambda f: int(f.split(".")[0]))
        depth_files.sort(key=lambda f: int(f.split(".")[0]))
    # if image files are of the format '1305031102.175304.png'
    elif len(image_files[0].split(".")) == 3:
        image_files.sort(key=sort_key)
        depth_files.sort(key=sort_key)
    else:
        assert False, "Image file format not recognized!"

    try:
        cam_extrinsics = read_rnnslam_extrinsics(extrinsics_file, image_files, depth_files)
        cam_intrinsics = read_rnnslam_intrinsics(intrinsics_file)
    except Exception as e:
        print('Error reading extrinsics file: {}'.format(e))
        return None

    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{} for RNNSIM data".format(idx+1, len(cam_extrinsics)))
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

        depth_path = os.path.join(depths_folder, os.path.basename(extr.depth_name))
        depth_name = os.path.basename(depth_path).split(".")[0]
        depth = Image.open(depth_path)
        depth = np.array(depth)
        depth = depth / np.max(depth)
        # inverse so that white is closer and black is further
        # depth = 1 - depth
        depth = (depth * 255).astype(np.float32)
        depth = Image.fromarray(depth) 

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, depth=depth, 
                              depth_path=depth_path, depth_name=depth_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


# class RNNSLAM_Dataset(object): # may need to be changed to rnnslam for simulation, in-vivo, and phantom separately b/c camera intrinsics are different
#     def __init__(self, source_path, images, depths):
#         self.source_path = source_path
#         self.images = images
#         self.depths = depths

#         self.load_meta()
    
#     def load_meta(self):
#         "Load data from dataset."
#         try:
#             # Read TUM poses & hard set camera intrinsics 
#             cameras_extrinsic_file = os.path.join(self.source_path, "sparse/0", "images.bin")
#             cameras_intrinsic_file = os.path.join(self.source_path, "sparse/0", "cameras.bin")
#             cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
#             cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
#         except:
#             cameras_extrinsic_file = os.path.join(self.source_path, "sparse/0", "images.txt")
#             cameras_intrinsic_file = os.path.join(self.source_path, "sparse/0", "cameras.txt")
#             cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#             cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

#         reading_dir = "images" if self.images == None else self.images
#         depths_dir = "depths" if self.depths == None else self.depths