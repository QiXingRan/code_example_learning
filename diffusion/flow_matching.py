import torch
import torch.nn as nn
import numpy as np
"""
类似于没有提示词的SD模型

学习一种映射关系f, 即如何从随机值生成目标集中的一个随机样本(把图修改为目标集中的样本)
"""
# 超参数
dim = 2  # 数据维度（2D点）
num_samples = 1000 # 采样点
num_steps = 50  # ODE求解步数
lr = 1e-3
epochs = 5000

# 目标分布：正弦曲线上的点（x1坐标）
x1_samples = torch.rand(num_samples, 1) * 4 * torch.pi  # 0到4π
y1_samples = torch.sin(x1_samples)  # y=sin(x)
target_data = torch.cat([x1_samples, y1_samples], dim=1)

# 噪声分布：高斯噪声（x0坐标）
noise_data = torch.randn(num_samples, dim) * 2

class VectorField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64),  # 输入维度: x (2) + t (1) = 3
            nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, x, t):
        # 直接拼接x和t（t的形状需为(batch_size, 1)）
        return self.net(torch.cat([x, t], dim=1))

model = VectorField()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    # 随机采样噪声点和目标点(eg. 在一个正弦函数中插入1000个点坐标)
    idx = torch.randperm(num_samples)

    x0 = noise_data[idx]  # 起点：噪声
    x1 = target_data[idx]  # 终点：正弦曲线

    # 时间t的形状为 (batch_size, 1)
    t = torch.rand(x0.size(0), 1)  # 例如：shape (1000, 1)

    # 线性插值生成中间点
    """
    t=0代表噪声分布 t=1代表目标分布 -> 充当类似“进度条”的作用，控制噪声到目标的过渡过程
    """
    xt = (1 - t) * x0 + t * x1
    # 模型预测向量场（直接传入t，无需squeeze）
    """
    给定 xt(中间点) t(时间)的情况下，预测向量场 -> 理解为随时间(t)流逝中的分布位置(xt)
    """
    vt_pred = model(xt, t)  # t的维度保持不变

    # 目标向量场：x1 - x0
    """ 
    flow matching主要是为了将噪声分布x0 映射为目标分布x1 -> x1-x0代表了 x0指向x1的距离向量，最后去拟合分布
    """
    vt_target = x1 - x0

    # 损失函数
    loss = torch.mean((vt_pred - vt_target) ** 2)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 使用模型
x = noise_data[0:1]  # 初始噪声点
trajectory = [x.detach().numpy()] # 初始化轨迹列表，并将初始噪声点添加到列表中
tag = torch.from_numpy(np.array([1]))
# 数值求解ODE（欧拉法）
t = 0
delta_t = 1 / num_steps # 每段时间步长
# 仅推理,无梯度计算
with torch.no_grad():
    for i in range(num_steps):
        vt = model(x, torch.tensor([[t]], dtype=torch.float32))
        t += delta_t # 增加时间步长
        x = x + vt * delta_t  # x(t+Δt) = x(t) + v(t)Δt 使用当前点的“斜率”预测下一个点位置(欧拉法)
        trajectory.append(x.detach().numpy()) # 结果

trajectory = torch.tensor(trajectory).squeeze()

print(trajectory[-1] / (torch.pi / 10 * 4))
