# Gaussian Viewing Utilities
#
# Author: Sierra Bonilla
# Date: 2023-11-10

import os
import json
import numpy as np
import torch
from utils.system_utils import searchForMaxIteration
from gaussian_renderer import render, GaussianModel 
from utils.graphics_utils import focal2fov 
from scene.cameras import Camera as GSCamera
from PIL import Image
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt

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
    tmp[:3, 3] = raw_camera['position']
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

    R = np.dot(R, previous_camera.R)
    T = previous_camera.T + T

    width = previous_camera.image_width
    height = previous_camera.image_height
    fovx = previous_camera.FoVx
    fovy = previous_camera.FoVy

    return GSCamera(colmap_id=previous_camera.colmap_id, R=R, T=T, FoVx=fovx, FoVy=fovy, image=torch.zeros((3, height, width)), gt_alpha_mask=None, image_name ='fake', uid=0)

def calculate_tilt_rotation(tilt_angle):
    """
    Calculate rotation matrix for a tilt angle.
    
    :param
        tilt_angle: angle to tilt camera
    
    :return
        R: rotation matrix
    """
    R_tilt = np.array([
        [np.cos(tilt_angle), 0, np.sin(tilt_angle)],
        [0, 1, 0],
        [-np.sin(tilt_angle), 0, np.cos(tilt_angle)]
    ])
    return R_tilt

def create_spiral_poses(num_views, tilt_angle, radius=0.5, height=0.5, n_loops=2):
    """
    Create poses along a spiral path with a given number of views and a tilt angle.

    :param
        num_views: number of views to create
        tilt_angle: angle to tilt camera (radians)
        radius: radius of the base circle of the spiral
        height: height of the spiral from start to end
        n_loops: number of loops in the spiral

    :return
        R_list, T_list: Lists of rotation and translation matrices for each pose
    """
    R_list = []
    T_list = []

    # Tilt rotation is constant for all poses
    R_tilt = calculate_tilt_rotation(tilt_angle)

    for i in range(num_views):
        # Calculate the progress along the spiral path
        t = n_loops * (2 * np.pi) * (i / num_views)

        # Rotation around Z-axis
        # R_z = np.array([
        #     [np.cos(t), -np.sin(t), 0],
        #     [np.sin(t),  np.cos(t), 0],
        #     [0,         0,          1]
        # ])
        R_z = np.array([(np.cos(t) * 0.5) - 2, -np.sin(t) - 0.5, -np.sin(0.5 * t) * 0.75]) * radius

        # reshape to 3x3 matrix
        R_z = np.reshape(R_z, (3, 3))

        # Combine the Z-axis rotation with the tilt rotation
        R_combined = R_z

        # Calculate the translation vector
        T = np.array([0, 0, 0])

        # Add the combined rotation and translation to the lists
        R_list.append(R_combined)
        T_list.append(T)

    return R_list, T_list

def circular_conical_cameras(center_camera, num_views, tilt_angle):
    """
    Create a circular motion about Z axis Rotation and Translation vectors around a fixed center point. 

    :param
        center_camera: GSCamera object
        num_views: number of views to create
        tilt_angle: angle to tilt camera (radians)
    
    :return
        cameras: list of GSCamera objects
    """
    cameras = []
    R, T = create_spiral_poses(num_views, tilt_angle)
    for i in range(num_views):
        cameras.append(new_camera(center_camera, R[i], T[i]))
    return cameras

def render_circular_conical_path_video(model_path, center_camera_idx, num_views, tilt_angle, output_path, fps, iteration=-1, sh_degree=3):
    """
    render a circular path video of the Gaussian model around the center_camera location provided. 

    :param
        model_path: path to model
        center_camera_idx: index to camera to use as center of circular path 
        num_views: number of views to create
        tilt_angle: angle to tilt camera (radians)
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
    cameras = circular_conical_cameras(center_camera, num_views, tilt_angle)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipeline = PipelineParamsNoparse()

    # Render
    images = []
    for camera in cameras:
        render_res = render(camera, gaussians, pipeline, background)
        rendering = render_res["render"]
        image = (rendering.permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu().numpy()
        images.append(image)

    # Save video from frames in images
    try:
        clip = ImageSequenceClip(list(images), fps=fps)
        clip.write_videofile(output_path, codec='libx264', audio=False)
        print(f'Video saved to {output_path}')

    except Exception as e:
        print('There seems to be an issue with loading the images.')
        print(e)
        return
    
def render_mask_gaussians(model_path, percent_of_gaussians, camera_index):
    """
    Render an image of the scene from camera index specified with specific percent of gaussians visisble.

    :param
        gaussians: GaussianModel object
        percent_of_gaussians: percent of gaussians to render
        camera_index: index of camera to render from
    
    :return
        image: image of scene
    """
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    scaling = gaussians._scaling.max(dim=1)[0]
    scaling_max = scaling.max().item()
    scaling_min = scaling.min().item()

    mask = scaling < (percent_of_gaussians * (scaling_max - scaling_min) + scaling_min)
    tmp_gaussians = GaussianModel(gaussians.max_sh_degree)
    tmp_gaussians._xyz = gaussians._xyz[mask, :]
    tmp_gaussians._features_dc = gaussians._features_dc[mask, ...]
    tmp_gaussians._features_rest = gaussians._features_rest[mask, ...]
    tmp_gaussians._opacity = gaussians._opacity[mask, ...]
    tmp_gaussians._scaling = gaussians._scaling[mask, ...]
    tmp_gaussians._rotation = gaussians._rotation[mask, ...]
    tmp_gaussians.active_sh_degree = gaussians.max_sh_degree

    cam = load_camera(model_path, camera_index)
    render_res = render(cam, tmp_gaussians, pipeline, background)
    rendering = render_res["render"]
    return (rendering.permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu().numpy()


def normalize(x):
    """
    Normalize a vector.
    
    :param
        x: vector to normalize
        
    :return
        normalized vector"""
    return x / np.linalg.norm(x)


def generate_spiral_points(n_poses, n_rotations, radius, plotting=False):
    """
    Generate points on a 2D spiral.

    :param
    n_poses (int): Number of points on the spiral.
    n_rotations (int): Number of rotations in the spiral.
    radius (float): Maximum radius of the spiral.

    :return
    np.array: Array of points on the spiral.
    """

    # The total angle for the spiral
    theta_max = 2 * np.pi * n_rotations

    # Linearly spaced values of theta
    theta = np.linspace(0, theta_max, n_poses)

    # Linearly increasing radius
    r = np.linspace(0, radius, n_poses)

    # Cartesian coordinates of the spiral
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    if plotting: 

        # Plotting the spiral
        plt.figure(figsize=(6, 6))
        plt.plot(x, y, marker='o')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('2D Spiral')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    return np.column_stack((x, y))


def generate_spiral_points3D(direction, translation, n_poses=120, n_rotations=2, radius=0.5, plotting=False):
    """
    Generate a spiral path in 3D space. 

    :param
        direction: vector perpendicular to the camera plane
        translation: translation vector of the camera
        n_poses: number of poses to create along the path
        n_rotations: number of rotations in the spiral
        radius: maximum radius of the spiral
    """

    # define a vector perpendicular to the direction vector
    if direction[1] != 0:
        perpendicular = np.array([direction[2], 0, -direction[0]])
    elif direction[2] != 0:
        perpendicular = np.array([0, direction[0], -direction[1]])
    else:
        perpendicular = np.array([direction[1], -direction[0], 0])
    
    # normalize the perpendicular vector
    V_norm = normalize(perpendicular) 

    # find second vector perpendicular to the direction vector
    U = np.cross(direction, V_norm)

    # normalize the second perpendicular vector
    U_norm = normalize(U)

    # generate rotation matrix R = [V_norm, U_norm, direction]
    R = np.column_stack((V_norm, U_norm, direction))

    # generate points on a 2D spiral
    points = generate_spiral_points(n_poses, n_rotations, radius)

    # reshape points to 3D [x, y, 0]
    points = np.column_stack((points, np.zeros((n_poses, 1))))

    # rotate the points to transform 2D spiral to camera plane
    rotated_points = np.dot(points, R)

    # add translation to the rotated points
    rotated_points = rotated_points + translation

    if plotting:
        # Plotting the spiral
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Spiral')
        ax.grid(True)
        plt.show()

    return rotated_points 

def render_spiral_path_video(model_path, center_camera_idx, fps, output_path, n_poses=120, n_rotations=2, radius=0.5, iteration=-1, sh_degree=3):
    """
    render a spiral path video of the Gaussian model around the center_camera location provided. 

    :param
        model_path: path to model
        center_camera_idx: index to camera to use as center of circular path 
        fps: frames per second of the video
        output_path: path to save the video
        n_poses: number of poses to create along the path
        n_rotations: number of rotations in the spiral
        radius: maximum radius of the spiral
        iteration: iteration of checkpoint to load (-1 for max)
        sh_degree: degree of spherical harmonics
    
    :return
        None
    """
    # Load model and cameras 
    gaussians = load_checkpoint(model_path, sh_degree, iteration)
    center_camera = load_camera(model_path, center_camera_idx)

    # Get the direction vector of the camera
    direction = center_camera.R[:, 2]
    translation = center_camera.T

    # Generate spiral path in camera plane
    points = generate_spiral_points3D(direction, translation, n_poses, n_rotations, radius)

    # Create a camera for each point on the spiral path
    cameras = []
    R = np.eye(3)
    for i in range(n_poses):
        cameras.append(new_camera(center_camera, R, points[i] - translation))

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipeline = PipelineParamsNoparse()

    # Render
    images = []
    for camera in cameras:
        render_res = render(camera, gaussians, pipeline, background)
        rendering = render_res["render"]
        image = (rendering.permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu().numpy()
        images.append(image)

    # Save video from frames in images
    try:
        clip = ImageSequenceClip(list(images), fps=fps)
        clip.write_videofile(output_path, codec='libx264', audio=False)
        print(f'Video saved to {output_path}')

    except Exception as e:
        print('There seems to be an issue with loading the images.')
        print(e)
        return
    
def rotate_about_x_axis(R, degree):
    """
    Rotate a given rotation matrix about the x-axis by a given degree.

    :param
        R: rotation matrix
        degree: degree to rotate by
    
    :return
        rotated rotation matrix
    """
    rad = np.radians(degree)
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad), np.cos(rad)]
    ])
    # R_x = np.array([
    #     [np.cos(rad), -np.sin(rad), 0],
    #     [np.sin(rad), np.cos(rad), 0],
    #     [0, 0, 1]
    # ])
    # R_x = np.array([
    #     [np.cos(rad), 0, np.sin(rad)],
    #     [0, 1, 0],
    #     [-np.sin(rad), 0, np.cos(rad)]
    # ])
    # transpose R first
    # R = R.transpose()
    return np.dot(R, R_x)

def rotate_about_x_path(model_path, center_camera_idx, degree, num_steps=120, num_repeats=1):
    """
    Return a list of new cameras that rotate from -degree to +degree 
    about the x-axis in num_steps repeated num_repeats times.

    :param
        model_path: path to model
        center_camera_idx: index to camera to use as center of the rotating path
        degree: degree to rotate by
        num_steps: number of steps to rotate in
        num_repeats: number of times to repeat the rotation
    
    :return
        list of new cameras
    """
    center_camera = load_camera(model_path, center_camera_idx)

    R_center = center_camera.R
    T_center = center_camera.T  
    cameras = []
    for i in range(num_repeats):
        for j in range(num_steps):
            degree_step = 2 * degree / num_steps 
            R = rotate_about_x_axis(R_center, -degree + (j * degree_step))
            fovx = center_camera.FoVx
            fovy = center_camera.FoVy
            height = center_camera.image_height
            width = center_camera.image_width
            new_camera = GSCamera(colmap_id=center_camera.colmap_id, R=R, T=T_center, FoVx=fovx, FoVy=fovy, image=torch.zeros((3, height, width)), gt_alpha_mask=None, image_name ='fake', uid=0)
            cameras.append(new_camera)
    return cameras

def render_rotate_about_x_path_video(model_path, center_camera_idx, degree, fps, output_path, num_steps=120, num_repeats=1):
    """
    render a rotation about x-axis path video of the Gaussian model around the center_camera location provided. 

    :param
        model_path: path to model
        center_camera_idx: index to camera to use as center of circular path 
        degree: degree to rotate by
        fps: frames per second of the video
        output_path: path to save the video
        num_steps: number of steps to rotate in
        num_repeats: number of times to repeat the rotation
        iteration: iteration of checkpoint to load (-1 for max)
        sh_degree: degree of spherical harmonics
    
    :return
        None
    """
    # Load model and cameras 
    gaussians = load_checkpoint(model_path)
    cameras = rotate_about_x_path(model_path, center_camera_idx, degree, num_steps, num_repeats)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    pipeline = PipelineParamsNoparse()

    # Render
    images = []
    for camera in cameras:
        render_res = render(camera, gaussians, pipeline, background)
        rendering = render_res["render"]
        image = (rendering.permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu().numpy()
        images.append(image)

    # Save video from frames in images
    try:
        clip = ImageSequenceClip(list(images), fps=fps)
        clip.write_videofile(output_path, codec='libx264', audio=False)
        print(f'Video saved to {output_path}')

    except Exception as e:
        print('There seems to be an issue with loading the images.')
        print(e)
        return