from typing import NamedTuple
import numpy as np
import os
import math
import json
import torch
from scene.cameras import Camera 

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

def normalize(x):
    return x / np.linalg.norm(x)

# cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                         image_path=image_path, image_name=image_name, 
#                         depth=depth, depth_name=depth_name, depth_path=depth_path,
#                         width=width, height=height)
# cam_infos.append(cam_info)
def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    return c2w

def camera_path_spiral(c2ws, focal=1, zrate=.1, rots=3, N=300):
    c2w = poses_avg(c2ws)
    up = normalize(c2ws[:, :3, 1].sum(0))
    tt = c2ws[:,:3,3]
    rads = np.percentile(np.abs(tt), 90, 0)
    rads[:] = rads.max() * .05
    
    render_poses = []
    rads = np.array(list(rads) + [1.])
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    render_poses = np.stack(render_poses, axis=0)
    render_poses = np.concatenate([render_poses, np.zeros_like(render_poses[..., :1, :])], axis=1)
    render_poses[..., 3, 3] = 1
    render_poses = np.array(render_poses, dtype=np.float32)
    return render_poses

def spiral_cam_info(model_path, focal=1):
    cam_path = os.path.join(model_path, 'cameras.json')
    if not os.path.exists(cam_path):
        print(f'Could not find saved cameras for the scene at {cam_path}')
        return None 
    with open(cam_path, 'r') as f:
        data = json.load(f)

    c2ws = np.zeros((len(data), 4, 4))

    for i, cam in enumerate(data):
        tmp = np.zeros((4,4))
        tmp[:3,:3] = cam['rotation']
        tmp[:3, 3] = cam['position']
        tmp[3,3] = 1

        c2w = np.linalg.inv(tmp)

        R = c2w[:3,:3].transpose()
        T = c2w[:3, 3]

        c2ws[i, :3, :3] = R
        c2ws[i, :3, 3] = T
    
    width = cam['width']
    height = cam['height']
    fovx = focal2fov(cam['fx'], width)
    fovy = focal2fov(cam['fy'], height)
    
    render_poses = camera_path_spiral(c2ws, focal=focal)

    cam_infos = []

    for i, pose in enumerate(render_poses):
        #GSCamera(colmap_id=idx, R=R, T=T, FoVx=fovx, FoVy=fovy, image=torch.zeros((3, height, width)), depth=torch.zeros((3, height, width)), gt_alpha_mask=None, image_name ='fake', uid=0)
        cam = Camera(colmap_id=i, R=pose[:3, :3], T=pose[:3, 3], FoVx=fovx, FoVy=fovy, image=torch.zeros((3, height, width)), depth=torch.zeros((3, height, width)), gt_alpha_mask=None, image_name='fake', uid=0)
        cam_infos.append(cam)
    
    return cam_infos
