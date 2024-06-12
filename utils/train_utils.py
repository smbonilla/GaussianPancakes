import os
import torch
import uuid
from argparse import Namespace
from .image_utils import psnr
from pytorch_msssim import ms_ssim
from .loss_utils import ssim
import lpips
from scene import Scene
import numpy as np
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False

def prepare_output_and_logger(args):
    """
    Prepares the output folder and Tensorboard logger.

    Parameters:
        args (Namespace): Arguments from the command line.
    """ 
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10]) 
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, verbose):
    """
    Logs training progress and evaluates the model at specific iterations.

    Parameters:
        tb_writer (SummaryWriter): TensorBoard writer for logging.
        iteration (int): Current iteration number.
        Ll1 (Tensor): L1 loss for logging.
        loss (Tensor): Total loss for logging.
        l1_loss (function): L1 loss function.
        elapsed (float): Elapsed time for the iteration.
        testing_iterations (list): Iterations at which to perform evaluation.
        scene (Scene): Scene object containing training and test data.
        renderFunc (function): Function to render images.
        renderArgs (tuple): Additional arguments for the render function.
        verbose (bool): If True, perform detailed evaluation and logging.
    """
    if tb_writer:
        # log basic metrics
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        lpips_model = lpips.LPIPS(net='vgg').cuda() if verbose else None

        for config in validation_configs:
            if not config['cameras']:
                continue

            l1_test, psnr_test = [], []
            lpips_test, ssim_test, mssim_test = ([] for _ in range(3)) if verbose else (None, None, None)

            for idx, viewpoint in enumerate(config['cameras']):
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                if tb_writer and idx < 5:
                    tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(f"{config['name']}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)

                l1_test.append(l1_loss(image, gt_image).mean().double())
                psnr_test.append(psnr(image, gt_image).mean().double())

                if verbose:
                    lpips_test.append(lpips_model(image, gt_image).mean().double())
                    ssim_test.append(ssim(image, gt_image).mean().double())
                    mssim_test.append(ms_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), data_range=1.0).mean().double())
            
            l1_test_avg, psnr_test_avg = sum(l1_test) / len(l1_test), sum(psnr_test) / len(psnr_test)

            if verbose:
                # Calculate verbose metrics averages and stds
                lpips_test_avg, ssim_test_avg, mssim_test_avg = sum(lpips_test) / len(lpips_test), sum(ssim_test) / len(ssim_test), sum(mssim_test) / len(mssim_test)
                lpips_test_std, ssim_test_std, mssim_test_std = torch.std(torch.tensor(lpips_test)), torch.std(torch.tensor(ssim_test)), torch.std(torch.tensor(mssim_test))
                
                # Print verbose metrics
                print(f"\n[ITER {iteration}] Evaluating {config['name']} avg: L1 {l1_test_avg} PSNR {psnr_test_avg} LPIPS {lpips_test_avg} SSIM {ssim_test_avg} MSSIM {mssim_test_avg}")
                print(f"[ITER {iteration}] Evaluating {config['name']} std: L1 {torch.std(torch.tensor(l1_test))} PSNR {torch.std(torch.tensor(psnr_test))} LPIPS {lpips_test_std} SSIM {ssim_test_std} MSSIM {mssim_test_std}")

            # Log basic and verbose metrics
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test_avg, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test_avg, iteration)
                if verbose:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test_avg, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test_avg, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - mssim', mssim_test_avg, iteration)

        if tb_writer:
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def save_image(tensor, filename, source_path):
    """
    Saves a tensor as an image.

    Parameters:
        tensor (Tensor): Tensor to save as an image.
        filename (str): Name of the file to save the image to.
        source_path (str): Path to the folder where the image should be saved.
    """
    array = tensor.detach().cpu().numpy()
    if array.shape[0] == 1: 
        array = np.squeeze(array, axis=0)
    else:
        array = array.transpose(1, 2, 0)  # Convert from CHW to HWC for RGB images.
    array = (array * 255).astype(np.uint8)
    Image.fromarray(array).save(os.path.join(source_path, filename))

def save_example_images(image, gt_image, depth, gt_depth, iteration, source_path):
    """
    Saves example images for debugging purposes.
    
    Parameters:
        image (Tensor): Rendered image.
        gt_image (Tensor): Ground truth image.
        depth (Tensor): Rendered depth map.
        gt_depth (Tensor): Ground truth depth map.
        iteration (int): Current iteration number.
        source_path (str): Path to the folder where the images should be saved.
    """
    save_image(image, "render_" + str(iteration) + ".png", source_path)
    save_image(gt_image, "gt_" + str(iteration) + ".png", source_path)
    save_image(depth, "depth_" + str(iteration) + ".png", source_path)
    save_image(gt_depth, "gt_depth_" + str(iteration) + ".png", source_path)