# Gaussian Viewing Utilities
#
# Author: Sierra Bonilla
# Date: 2023-11-10

import os
import json
import numpy as np
import torch
from system_utils import searchForMaxIteration
from gaussian_renderer import render, GaussianModel 
from graphics_utils import focal2fov 
from scene.cameras import Camera as GSCamera
from PIL import Image
import cv2 

class PipelineParamsNoparse:
    """ Same as PipelineParams but without argument parser. """
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False                      
        self.debug = False

def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    """
    Load a checkpoint from a model path.

    :param
        model_path: path to model
        sh_degree: degree of spherical harmonics
        iteration: iteration of checkpoint to load (-1 for max)
    
    :return
        gaussians: GaussianModel object    
    """
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(checkpt_dir, f"iteration_{iteration}", "point_cloud.ply")
    
    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)                                                 
    return gaussians

def load_camera(model_path, idx=0):
    """
    Load one of the default cameras for the scene. 
    
    :param
        model_path: path to model
        idx: index of camera to load
    
    :return
        camera: GSCamera object
    """
    cam_path = os.path.join(model_path, 'cameras.json')
    if not os.path.exists(cam_path):
        print(f'Could not find saved cameras for the scene at {cam_path}')
        return None 
    with open(cam_path, 'r') as f:
        data = json.load(f)
        raw_camera = data[idx]

    tmp = np.zeros((4,4))
    tmp[:3,:3] = raw_camera['rotation']
    tmp[:3, 3] = raw_camera['translation']
    tmp[3,3] = 1

    C2W = np.linalg.inv(tmp)

    R = C2W[:3,:3].transpose()
    T = C2W[:3, 3]

    width = raw_camera['width']
    height = raw_camera['height']
    fovx = focal2fov(raw_camera['fx'], width)
    fovy = focal2fov(raw_camera['fy'], height)

    return GSCamera(colmap_id=idx, R=R, T=T, FoVx=fovx, FoVy=fovy, image=torch.zeros((3, height, width)), gt_alpha_mask=None, image_name ='fake', uid=0)

def new_camera(previous_camera, R, T):
    """
    Create a new camera view from previous camera with applying new rotation and translation to previous rotation and translation. 

    :param
        previous_camera: GSCamera object
        R: rotation matrix
        T: translation vector
    
    :return 
        new_camera: GSCamera object
    """
    C2W = np.zeros((4,4))
    C2W[:3,:3] = R
    C2W[:3, 3] = T
    C2W[3,3] = 1

    C2W = np.linalg.inv(C2W)

    R = previous_camera.R*C2W[:3,:3].transpose()
    T = previous_camera.T*C2W[:3, 3]

    width = previous_camera.width
    height = previous_camera.height
    fovx = previous_camera.FoVx
    fovy = previous_camera.FoVy

    return GSCamera(colmap_id=previous_camera.colmap_id, R=R, T=T, FoVx=fovx, FoVy=fovy, image=torch.zeros((3, height, width)), gt_alpha_mask=None, image_name ='fake', uid=0)

def circular_motion_Z(num_views):
    """
    Create a circular motion about Z axis Rotation and Translation vectors around a fixed center point. 

    :param
        num_views: number of views to create
    
    :return
        R, T: rotation and translation vectors
    """
    R = []
    T = []
    for i in range(num_views):
        theta = 2*np.pi*i/num_views
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        T = np.array([0, 0, 0])
        R.append(R)
        T.append(T)
    return R, T

def circular_cameras(center_camera, num_views):
    """
    Create a circular motion about Z axis Rotation and Translation vectors around a fixed center point. 

    :param
        center_camera: GSCamera object
        num_views: number of views to create
    
    :return
        cameras: list of GSCamera objects
    """
    cameras = []
    R, T = circular_motion_Z(num_views)
    for i in range(num_views):
        cameras.append(new_camera(center_camera, R[i], T[i]))
    return cameras

def render_circular_path_video(model_path, center_camera_idx, num_views, output_path, fps, iteration=-1, sh_degree=3):
    """
    render a circular path video of the Gaussian model around the center_camera location provided. 

    :param
        model_path: path to model
        center_camera_idx: index to camera to use as center of circular path 
        num_views: number of views to create
        output_path: path to save the video
        fps: frames per second of the video
        iteration: iteration of checkpoint to load (-1 for max)
        sh_degree: degree of spherical harmonics
    
    :return
        None
    """
    # Load model and cameras 
    gaussians = load_checkpoint(model_path, sh_degree, iteration)
    center_camera = load_camera(model_path, center_camera_idx)
    cameras = circular_cameras(center_camera, num_views)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipeline = PipelineParamsNoparse()

    # Render
    images = []
    for camera in cameras:
        render_res = render(camera, gaussians, pipeline, background)
        rendering = render_res["render"]
        image = Image.fromarray((rendering.permute(1, 2, 0) * 255).to(torch.uint8), 'RGB')
        images.append(image)

    # Save video from frames in images
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (images[0].shape[1], images[0].shape[0]))

    try:
        for image in images:
                frame = cv2.imread(image)
                out.write(frame)
        out.release()
        print(f'Video saved to {output_path}')

    except Exception as e:
        print('There seems to be an issue with loading the images.')
        print(e)
        return
    
