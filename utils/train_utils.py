import os
import torch
import uuid
from argparse import Namespace
from .image_utils import psnr
from pytorch_msssim import ms_ssim
from .loss_utils import ssim
import lpips
from scene import Scene

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10]) # CHANGE TO OUTPUT TO RAID NOT HOME
        
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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        
        lpips_model = lpips.LPIPS(net='vgg').cuda()

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = []
                psnr_test = []
                lpips_test = []
                ssim_test = []
                mssim_test = []
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test.append(l1_loss(image, gt_image).mean().double())
                    psnr_test.append(psnr(image, gt_image).mean().double())
                    lpips_test.append(lpips_model(image, gt_image).mean().double())
                    ssim_test.append(ssim(image, gt_image).mean().double())
                    mssim_test.append(ms_ssim(image.unsqueeze(0), gt_image.unsqueeze(0), data_range=1.0).mean().double())

                psnr_test_avg = sum(psnr_test) / len(psnr_test)
                l1_test_avg = sum(l1_test) / len(l1_test)
                lpips_test_avg = sum(lpips_test) / len(lpips_test)
                ssim_test_avg = sum(ssim_test) / len(ssim_test)
                mssim_test_avg = sum(mssim_test) / len(mssim_test)
                psnr_test_std = torch.std(torch.tensor(psnr_test))
                l1_test_std = torch.std(torch.tensor(l1_test))
                lpips_test_std = torch.std(torch.tensor(lpips_test))
                ssim_test_std = torch.std(torch.tensor(ssim_test))
                mssim_test_std = torch.std(torch.tensor(mssim_test))

                print("\n[ITER {}] Evaluating {} avg: L1 {} PSNR {} LPIPS {} SSIM {} MSSIM {}".format(iteration, config['name'], l1_test_avg, psnr_test_avg, lpips_test_avg, ssim_test_avg, mssim_test_avg))
                print("[ITER {}] Evaluating {} std: L1 {} PSNR {} LPIPS {} SSIM {} MSSIM {}".format(iteration, config['name'], l1_test_std, psnr_test_std, lpips_test_std, ssim_test_std, mssim_test_std))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test_avg, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test_avg, iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()