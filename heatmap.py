import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import Generator

class ActivationVisualizer(Generator):
    def __init__(self, scale_factor):
        super().__init__(scale_factor)
        
        # 注册钩子获取激活图
        self.activations = []
        def hook_fn(module, input, output):
            self.activations.append(output.detach())
        
        # 选择可视化层（示例使用第4个残差块后的激活）
        self.block7.register_forward_hook(hook_fn)

    def get_heatmap(self, x):
        self.activations = []  # 清空历史激活
        with torch.no_grad():
            _ = self.forward(x)
        
        # 获取激活并处理
        activation = self.activations[0].cpu().numpy()
        heatmap = np.mean(activation, axis=1).squeeze()  # 通道维度平均
        
        # 归一化处理
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        return heatmap

def process_image(input_path, output_path, checkpoint_path, scale_factor=4):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = ActivationVisualizer(scale_factor).eval()  # 创建实例并设置eval模式
    model = model.to(device)
    
    # 加载权重（直接使用传入的checkpoint路径）
    model.load_state_dict(torch.load(checkpoint_path))
    
    # 读取图像（使用PIL）
    img = Image.open(input_path).convert('L')  # 转为灰度图
    img_array = np.array(img)
    
    # 预处理
    input_tensor = torch.from_numpy(img_array.astype(np.float32)/255.0)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)

    # 生成热力图
    heatmap = model.get_heatmap(input_tensor)

    # 调整热力图大小（使用PIL）
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_uint8)
    heatmap_pil = heatmap_pil.resize(img.size, Image.BILINEAR)
    heatmap = np.array(heatmap_pil).astype(np.float32) / 255.0

    # 创建叠加可视化
    heatmap = np.clip(heatmap, 0, 1)
    
    # 应用颜色映射
    heatmap_img = plt.cm.jet(heatmap)[..., :3]  # 获取RGB通道
    heatmap_img = (heatmap_img * 255).astype(np.uint8)
    
    # 将灰度图转为RGB格式
    img_rgb = np.stack([img_array]*3, axis=-1)
    
    # 叠加图像
    superimposed_img = (heatmap_img * 0.4 + img_rgb * 0.6).clip(0, 255).astype(np.uint8)

    # 保存结果
    Image.fromarray(superimposed_img).save(output_path)
    print(f"可视化结果已保存至：{output_path}")

if __name__ == "__main__":
    # 配置参数
    config = {
        "input_image": "/root/bayes-tmp/SRGAN-master/data/verification/10/104.jpg",
        "output_heatmap": "/root/bayes-tmp/SRGAN-master/heatmap/test.png",
        "model_weights": "/root/bayes-tmp/SRGAN-master/epochs512/netG_epoch_4_99.pth",
        "scale_factor": 4
    }
    
    process_image(
        config["input_image"],
        config["output_heatmap"],
        config["model_weights"],
        config["scale_factor"]
    )