import cv2
import numpy as np
import os
import glob
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def calculate_rase(gt, pred):
    """
    计算RASE(相对绝对误差平均值)
    :param gt: 真实图像(ndarray)
    :param pred: 预测图像(ndarray)
    :return: RASE值
    """
    # 确保图像为浮点类型
    gt = gt.astype(np.float64)
    pred = pred.astype(np.float64)
    
    # 避免除以零
    gt_nonzero = np.where(gt == 0, 1e-10, gt)
    
    # 计算相对绝对误差
    relative_abs_error = np.abs((gt_nonzero - pred) / gt_nonzero)
    
    # 计算平均值
    rase = np.mean(relative_abs_error)
    return rase

def process_images(input_dir, output_dir, hr_dir, result_file):
    """
    处理图像并计算指标
    :param input_dir: 输入图像目录(包含子文件夹)
    :param output_dir: 输出图像目录
    :param hr_dir: 高分辨率参考图像目录
    :param result_file: 结果保存文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有输入图像路径
    input_paths = glob.glob(os.path.join(input_dir, '**', '*.png'), recursive=True)
    input_paths += glob.glob(os.path.join(input_dir, '**', '*.jpg'), recursive=True)
    
    results = []
    
    for input_path in input_paths:
        # 读取输入图像
        img = cv2.imread(input_path)
        if img is None:
            continue
            
        # 检查尺寸
        if img.shape[0] != 256 or img.shape[1] != 256:
            print(f"跳过非256x256图像: {input_path}")
            continue
            
        # 放大图像到512x512
        upscaled = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        
        # 创建输出路径
        rel_path = os.path.relpath(input_path, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存放大后的图像
        cv2.imwrite(output_path, upscaled)
        
        # 获取对应的HR图像路径
        hr_path = os.path.join(hr_dir, rel_path)
        if not os.path.exists(hr_path):
            print(f"未找到HR图像: {hr_path}")
            continue
            
        # 读取HR图像
        hr_img = cv2.imread(hr_path)
        if hr_img is None:
            continue
            
        # 确保HR图像为512x512
        if hr_img.shape[0] != 512 or hr_img.shape[1] != 512:
            hr_img = cv2.resize(hr_img, (512, 512))
        
        # 转换为YUV颜色空间并提取Y通道
        hr_y = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YUV)[:, :, 0]
        upscaled_y = cv2.cvtColor(upscaled, cv2.COLOR_BGR2YUV)[:, :, 0]
        
        # 计算指标
        psnr_value = psnr(hr_y, upscaled_y, data_range=255)
        ssim_value = ssim(hr_y, upscaled_y, data_range=255)
        rase_value = calculate_rase(hr_y, upscaled_y)
        
        # 保存结果
        results.append({
            'image': rel_path,
            'psnr': psnr_value,
            'ssim': ssim_value,
            'rase': rase_value
        })
        print(f"处理完成: {rel_path} - PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, RASE: {rase_value:.4f}")
    
    # 保存结果到文件
    with open(result_file, 'w') as f:
        f.write("Image\tPSNR\tSSIM\tRASE\n")
        for res in results:
            f.write(f"{res['image']}\t{res['psnr']:.4f}\t{res['ssim']:.4f}\t{res['rase']:.4f}\n")
    
    # 计算平均值
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    avg_rase = np.mean([r['rase'] for r in results])
    
    with open(result_file, 'a') as f:
        f.write("\n")
        f.write(f"Average\t{avg_psnr:.4f}\t{avg_ssim:.4f}\t{avg_rase:.4f}\n")
    
    print(f"处理完成! 共处理 {len(results)} 张图像")
    print(f"平均值 - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, RASE: {avg_rase:.4f}")

if __name__ == "__main__":
    # 配置路径
    input_dir = "D:/cp_Training_data/6_5/experiment1/PNSRSSIMRASE/202220/202220_lr1/X2/202220lr2"  # 包含256x256图像的文件夹
    output_dir = "D:/cp_Training_data/6_5/experiment1/PNSRSSIMRASE/SRGAN2/202220"  # 放大图像保存位置
    hr_dir = "D:/cp_Training_data/6_5/experiment1/PNSRSSIMRASE/202220/202220_gt/202220gt"  # 高分辨率参考图像
    result_file = "image_metrics.txt"  # 结果保存文件
    
    process_images(input_dir, output_dir, hr_dir, result_file)