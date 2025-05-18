import torch
import numpy as np
import cv2
import os

# 高频梯度计算
def gradient(image):
    # Sobel算子的定义
    sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32).unsqueeze(0).cuda()
    sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32).unsqueeze(0).cuda()

    # 将NumPy数组转换为PyTorch张量，并调整通道顺序
    image = torch.from_numpy(image.transpose((2, 0, 1))).float().cuda()

    image = image.clamp(0, 255)  # 确保像素值在0-255范围内
    image = image.to(torch.uint8)
    image = image.to(torch.float32)
    img_tensor = image
    grads = []
    for channel in range(3):
        channel_tensor = img_tensor[channel].unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
        # 对图像应用Sobel算子进行边缘检测
        grad_x = torch.nn.functional.conv2d(channel_tensor, sobel_x, padding=1)
        grad_y = torch.nn.functional.conv2d(channel_tensor, sobel_y, padding=1)
        # 计算梯度幅值
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grads.append(grad)

    # 合并三个通道的梯度图
    grad_tensor = torch.cat(grads, dim=1).squeeze(0)  # shape: [3, H, W]
    return grad_tensor

# 读取图像
image = cv2.imread(r'../images/bicycle.JPG')

# 检查图像是否成功加载
if image is None:
    print("Error: Could not read image.")
else:
    # 计算梯度
    grad_tensor = gradient(image)

    # 将梯度张量转换回NumPy数组，并调整通道顺序
    grad_numpy = grad_tensor.cpu().numpy().transpose((1, 2, 0))
    print(grad_numpy.shape)

    # 保存梯度图像
    output_path = r'../images/bicycle_sobel.JPG' # 修改保存路径与文件名
    cv2.imwrite(output_path, grad_numpy)
    print(f"Gradient image saved to {output_path}")