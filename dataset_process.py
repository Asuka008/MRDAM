import os
import random
from PIL import Image
import numpy as np
from shutil import copyfile

# 配置参数
base_dir = "D:/cp_Training_data/SRGAN"  # 原始800x800图像路径
output_dir = "D:/cp_Training_data/Sunflower4"          # 输出路径
hr_size = 512               # HR图像尺寸
scale_factor = 2            # 下采样倍数
train_ratio = 0.8            # 训练集比例
num_train_crops = 2         # 每张训练图随机裁剪次数

# 创建输出文件夹结构
os.makedirs(f'{output_dir}/train_HR', exist_ok=True)
os.makedirs(f'{output_dir}/train_LR', exist_ok=True)
os.makedirs(f'{output_dir}/valid_HR', exist_ok=True)
os.makedirs(f'{output_dir}/valid_LR', exist_ok=True)

# 获取所有图像文件
all_files = [f for f in os.listdir(base_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
random.shuffle(all_files)

# 分割训练集和验证集
split_idx = int(len(all_files) * train_ratio)
train_files = all_files[:split_idx]
valid_files = all_files[split_idx:]

def process_image(img_path, save_hr_dir, save_lr_dir, is_train):
    """处理单张图像并保存结果"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    
    if is_train:
        # 训练集：随机裁剪多个256x256区域
        for i in range(num_train_crops):
            # 随机左上角坐标
            left = random.randint(0, w - hr_size)
            top = random.randint(0, h - hr_size)
            # 裁剪HR
            hr_crop = img.crop((left, top, left+hr_size, top+hr_size))
            # 生成LR
            lr_crop = hr_crop.resize((hr_size//scale_factor, hr_size//scale_factor), 
                                   Image.BICUBIC)
            # 保存
            base_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_{i}"
            hr_crop.save(f"{save_hr_dir}/{base_name}.jpg")
            lr_crop.save(f"{save_lr_dir}/{base_name}.jpg")
    else:
        # 验证集：中心裁剪单个256x256区域
        center_x, center_y = w//2, h//2
        left = center_x - hr_size//2
        top = center_y - hr_size//2
        hr_crop = img.crop((left, top, left+hr_size, top+hr_size))
        lr_crop = hr_crop.resize((hr_size//scale_factor, hr_size//scale_factor),
                               Image.BICUBIC)
        # 保存
        base_name = os.path.basename(img_path)
        hr_crop.save(f"{save_hr_dir}/{base_name}")
        lr_crop.save(f"{save_lr_dir}/{base_name}")

# 处理训练集
for f in train_files:
    process_image(
        img_path=os.path.join(base_dir, f),
        save_hr_dir=f'{output_dir}/train_HR',
        save_lr_dir=f'{output_dir}/train_LR',
        is_train=True
    )

# 处理验证集
for f in valid_files:
    process_image(
        img_path=os.path.join(base_dir, f),
        save_hr_dir=f'{output_dir}/valid_HR',
        save_lr_dir=f'{output_dir}/valid_LR',
        is_train=False
    )

print(f"处理完成！生成数据分布：")
print(f"训练集 HR: {len(os.listdir(f'{output_dir}/train_HR'))} 张")
print(f"验证集 HR: {len(os.listdir(f'{output_dir}/valid_HR'))} 张")