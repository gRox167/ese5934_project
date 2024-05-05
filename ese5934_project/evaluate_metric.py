import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def Evaluate_MT1(Ground_Truth_IMG, Predicted_IMG):
    gt_np = torch.view_as_complex(Ground_Truth_IMG).abs()[0].cpu().numpy()
    pred_np = torch.view_as_complex(Predicted_IMG).abs().cpu().numpy()

    # Define the data range as the maximum value found in the ground truth image
    data_range = np.max(gt_np) - np.min(gt_np)

    # Calculate PSNR
    psnr_value = psnr(gt_np, pred_np, data_range=data_range)

    # Calculate SSIM
    ssim_value = ssim(gt_np, pred_np, data_range=data_range)

    print(f"PSNR Value mt1: {psnr_value}")
    print(f"SSIM Value mt1: {ssim_value}")
    return psnr_value, ssim_value


def Evaluate_MT2(Ground_Truth_IMG, Predicted_IMG):
    gt_np = torch.view_as_complex(Ground_Truth_IMG).abs()[0].cpu().numpy()
    pred_np = torch.view_as_complex(Predicted_IMG).abs().cpu().numpy()

    gt_np_std = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min())
    pred_np_std = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())
    data_range_std = gt_np_std.max() - gt_np_std.min()

    # Calculate PSNR
    psnr_value = psnr(gt_np_std, pred_np_std, data_range=data_range_std)

    # Calculate SSIM
    ssim_value = ssim(gt_np_std, pred_np_std, data_range=data_range_std)

    print(f"PSNR Value mt2: {psnr_value}")
    print(f"SSIM Value mt2: {ssim_value}")
    return psnr_value, ssim_value
