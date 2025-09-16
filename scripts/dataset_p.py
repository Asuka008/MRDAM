import cv2
import os

def downsample_images():
    """
    对预设路径中的图片进行2倍和4倍下采样处理
    """
    # 在这里直接修改路径↓↓↓↓↓↓
    input_dir = "D:/progamming/papers/SRGAN-PyTorch-main/data/Sunflower8/test/10"  # 输入图片目录路径
    output_root = "D:/progamming/papers/SRGAN-PyTorch-main/data/Sunflower8"  # 输出根目录路径
    # 在这里直接修改路径↑↑↑↑↑↑

    # 创建输出子目录
    output_2x = os.path.join(output_root, '2x')
    output_4x = os.path.join(output_root, '4x')
    os.makedirs(output_2x, exist_ok=True)
    os.makedirs(output_4x, exist_ok=True)
    
    # 支持的图片格式
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    # 遍历输入目录
    for filename in os.listdir(input_dir):
        # 检查文件格式
        if filename.lower().endswith(valid_exts):
            img_path = os.path.join(input_dir, filename)
            
            # 读取图片（保留Alpha通道）
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"无法读取图像：{img_path}")
                continue
            
            # 获取原始尺寸
            h, w = img.shape[:2]
            
            # 处理2倍下采样
            new_size_2x = (w // 2, h // 2)
            if new_size_2x[0] > 0 and new_size_2x[1] > 0:
                img_2x = cv2.resize(img, new_size_2x, interpolation=cv2.INTER_AREA)
                # 直接使用原文件名保存
                output_path = os.path.join(output_2x, filename)
                cv2.imwrite(output_path, img_2x)
            else:
                print(f"跳过2倍下采样 - {filename} 尺寸过小")
            
            # 处理4倍下采样
            new_size_4x = (w // 4, h // 4)
            if new_size_4x[0] > 0 and new_size_4x[1] > 0:
                img_4x = cv2.resize(img, new_size_4x, interpolation=cv2.INTER_AREA)
                # 直接使用原文件名保存
                output_path = os.path.join(output_4x, filename)
                cv2.imwrite(output_path, img_4x)
            else:
                print(f"跳过4倍下采样 - {filename} 尺寸过小")

if __name__ == "__main__":
    downsample_images()
    print("处理完成！")