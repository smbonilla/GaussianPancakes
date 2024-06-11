#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from pytorch_msssim import ms_ssim
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def rescale_image(img, new_min=0, new_max=255):
    old_min, old_max = img.getextrema()
    scale = (new_max - new_min) / (old_max - old_min)
    result_img = img.point(lambda i: (i - old_min) * scale + new_min)
    return result_img

def normalize_tensor_to_range(tensor, target_min, target_max):
    current_min = tensor.min()
    current_max = tensor.max()
    if current_min == current_max:
        return tensor.clone().fill_(target_min)
    normalized_tensor = (tensor - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
    return normalized_tensor

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        # fname does not end with "_depth.png"
        if not fname.endswith("_depth.png"): 
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)
    return renders, gts, image_names

def readDepths(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if fname.endswith("_depth.png"): 
            render = Image.open(renders_dir / fname).convert("L")
            gt = Image.open(gt_dir / fname).convert("L")

            # ---- testing next two line ----
            # make sure both are between 0-255
            render = rescale_image(render)
            #gt = rescale_image(gt)

            render_tensor = tf.to_tensor(render)
            gt_tensor = tf.to_tensor(gt)
            # render_tensor = normalize_tensor_to_range(render_tensor, 0, 1)
            
            renders.append(render_tensor.unsqueeze(0).cuda())
            gts.append(gt_tensor.unsqueeze(0).cuda())
            image_names.append(fname)
        
    return renders, gts, image_names

def evaluate(model_paths, train_times):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                depths, gt_depths, depth_image_names = readDepths(renders_dir, gt_dir)

                depths_mse = []
                depths_ssim = []

                # save one depth and one gt depth to png and then break
                for i in range(len(depths)):
                    depth = depths[i]
                    gt_depth = gt_depths[i]
                    depths_mse.append(torch.mean((depth - gt_depth) ** 2).item())
                    depths_ssim.append(ssim(depth, gt_depth).item())

                ssims = []
                msssims = []
                psnrs = []
                lpipss = []

                num_samples = len(renders)

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    msssims.append(ms_ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                
                SSIM_avg, SSIM_std = torch.tensor(ssims).mean(), torch.tensor(ssims).std()
                MS_SSIM_avg, MS_SSIM_std = torch.tensor(msssims).mean(), torch.tensor(msssims).std()
                PSNR_avg, PSNR_std = torch.tensor(psnrs).mean(), torch.tensor(psnrs).std()
                LPIPS_avg, LPIPS_std = torch.tensor(lpipss).mean(), torch.tensor(lpipss).std()
                DepthMSE_avg, DepthMSE_std = torch.tensor(depths_mse).mean(), torch.tensor(depths_mse).std()
                DepthSSIM_avg, DepthSSIM_std = torch.tensor(depths_ssim).mean(), torch.tensor(depths_ssim).std()    

                print("  SSIM : {:>12.7f}".format(SSIM_avg, ".5"))
                print("  MS-SSIM : {:>12.7f}".format(MS_SSIM_avg, ".5"))
                print("  PSNR : {:>12.7f}".format(PSNR_avg, ".5"))
                print("  LPIPS: {:>12.7f}".format(LPIPS_avg, ".5"))
                print("  Depth MSE: {:>12.7f}".format(DepthMSE_avg, ".5"))
                print("  Depth SSIM: {:>12.7f}".format(DepthSSIM_avg, ".5"))

                # if train_times is not an empty list, add the training time to the dictionary
                if train_times:
                    full_dict[scene_dir][method].update({"SSIM": SSIM_avg.item(),
                                                     "SSIM_std": SSIM_std.item(),
                                                    "MS-SSIM": MS_SSIM_avg.item(),
                                                    "MS-SSIM_std": MS_SSIM_std.item(),
                                                        "PSNR": PSNR_avg.item(),
                                                        "PSNR_std": PSNR_std.item(),
                                                        "LPIPS": LPIPS_avg.item(), 
                                                        "LPIPS_std": LPIPS_std.item(),
                                                        "DepthMSE": DepthMSE_avg.item(),
                                                        "DepthMSE_std": DepthMSE_std.item(),
                                                        "DepthSSIM": DepthSSIM_avg.item(),
                                                        "DepthSSIM_std": DepthSSIM_std.item(),
                                                        "NumSamples": num_samples,
                                                        "TrainTime": train_times[model_paths.index(scene_dir)]})
                else:
                    full_dict[scene_dir][method].update({"SSIM": SSIM_avg.item(),
                                                        "SSIM_std": SSIM_std.item(),
                                                        "MS-SSIM": MS_SSIM_avg.item(),
                                                        "MS-SSIM_std": MS_SSIM_std.item(),
                                                            "PSNR": PSNR_avg.item(),
                                                            "PSNR_std": PSNR_std.item(),
                                                            "LPIPS": LPIPS_avg.item(), 
                                                            "LPIPS_std": LPIPS_std.item(),
                                                            "DepthMSE": DepthMSE_avg.item(),
                                                            "DepthMSE_std": DepthMSE_std.item(),
                                                            "DepthSSIM": DepthSSIM_avg.item(),
                                                            "DepthSSIM_std": DepthSSIM_std.item(),
                                                            "NumSamples": num_samples})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                        "MS-SSIM": {name: msssim for msssim, name in zip(torch.tensor(msssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                            "DepthMSE": {name: mse for mse, name in zip(torch.tensor(depths_mse).tolist(), depth_image_names)},
                                                            "DepthSSIM": {name: ssim for ssim, name in zip(torch.tensor(depths_ssim).tolist(), depth_image_names)},
                                                            "NumSamples": num_samples})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--train_times', '-t', required=False, nargs="+", type=float, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths, args.train_times)
