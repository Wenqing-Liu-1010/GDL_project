# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import Subset

# # ------------------------------------------------------------
# # 1. 设置设备和随机种子
# # ------------------------------------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"使用设备: {device}")

# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(42)

# # ------------------------------------------------------------
# # 2. 数据准备
# # ------------------------------------------------------------
# transform = transforms.Compose([
#     transforms.ToTensor(),              # HWC [0,1]
#     transforms.Normalize((0.1307,), (0.3081,))   # 标准化
# ])

# # 加载训练集和测试集
# mnist_train = datasets.MNIST(root="./data", train=True, download=True,
#                            transform=transform)
# mnist_test = datasets.MNIST(root="./data", train=False, download=True,
#                           transform=transform)

# # 使用训练集的子集
# subset_size = 2_000                     # 控制数据集大小
# subset_indices = list(range(subset_size))
# mnist_subset = Subset(mnist_train, subset_indices)

# # 展平到 [N, 784]
# x_train = torch.stack([d[0].view(-1) for d in mnist_subset]).to(device)  # [N,784]
# y_train = torch.tensor([d[1] for d in mnist_subset], device=device)      # [N]

# # 处理测试集数据
# x_test = torch.stack([d[0].view(-1) for d in mnist_test]).to(device)  # [N,784]
# y_test = torch.tensor([d[1] for d in mnist_test], device=device)      # [N]

# print(f"训练集大小: {len(mnist_subset)} 样本")
# print(f"测试集大小: {len(mnist_test)} 样本")

# # ------------------------------------------------------------
# # 3. 网络定义
# # ------------------------------------------------------------
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(784, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 10)           # 10类输出
#         )
#     def forward(self, x):
#         return self.net(x)

# # ------------------------------------------------------------
# # 4. 工具函数
# # ------------------------------------------------------------
# def flat_params(model):
#     """将模型参数展平成一维向量"""
#     return torch.cat([p.data.view(-1) for p in model.parameters()])

# def set_params(model, flat):
#     """将一维向量恢复成模型参数"""
#     idx = 0
#     for p in model.parameters():
#         n = p.numel()
#         p.data.copy_(flat[idx:idx+n].view_as(p))
#         idx += n

# def evaluate(model, params, criterion, x=None, y=None):
#     """评估给定参数下的模型性能"""
#     x = x_train if x is None else x
#     y = y_train if y is None else y
#     set_params(model, params)
#     with torch.no_grad():
#         logits = model(x)
#         loss = criterion(logits, y).item()
#         acc = (logits.argmax(1) == y).float().mean().item()
#         return loss, acc

# # ------------------------------------------------------------
# # 5. Adam训练
# # ------------------------------------------------------------
# def train_with_adam(model, lr=1e-3, epochs=300):
#     print("\nAdam训练开始...")
#     start_time = time.time()
    
#     model.to(device)
#     opt = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
    
#     for epoch in range(epochs):
#         opt.zero_grad()
#         logits = model(x_train)
#         loss = criterion(logits, y_train)
#         loss.backward()
#         opt.step()
        
#         if (epoch + 1) % 50 == 0:
#             train_loss, train_acc = evaluate(model, flat_params(model), criterion)
#             test_loss, test_acc = evaluate(model, flat_params(model), criterion, x_test, y_test)
#             print(f"Epoch {epoch+1}/{epochs}")
#             print(f"  训练集 - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
#             print(f"  测试集 - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")
    
#     # 最终评估
#     train_loss, train_acc = evaluate(model, flat_params(model), criterion)
#     test_loss, test_acc = evaluate(model, flat_params(model), criterion, x_test, y_test)
    
#     train_time = time.time() - start_time
#     return train_loss, train_acc, test_loss, test_acc, train_time

# # ------------------------------------------------------------
# # 6. PSO训练
# # ------------------------------------------------------------
# def train_with_pso(model, n_particles=300, w=0.4, c1=1.2, c2=1.2, iterations=500):
#     print("\nPSO训练开始...")
#     start_time = time.time()
    
#     model.to(device)
#     dim = sum(p.numel() for p in model.parameters())
#     criterion = nn.CrossEntropyLoss()

#     # 初始化粒子群
#     swarm = [flat_params(model).clone()+0.1*torch.randn(dim, device=device)
#              for _ in range(n_particles)]
#     vel = [torch.zeros(dim, device=device) for _ in range(n_particles)]
#     pbest = [s.clone() for s in swarm]
#     pbest_v = [float("inf")]*n_particles
#     gbest, gbest_v = None, float("inf")

#     # PSO迭代
#     for it in range(iterations):
#         for i in range(n_particles):
#             # 更新速度和位置
#             r1, r2 = torch.rand(dim, device=device), torch.rand(dim, device=device)
#             social = 0 if gbest is None else c2*r2*(gbest - swarm[i])
#             vel[i] = w*vel[i] + c1*r1*(pbest[i]-swarm[i]) + social
#             swarm[i] = swarm[i] + vel[i]

#             # 评估新位置
#             loss, _ = evaluate(model, swarm[i], criterion)
#             if loss < pbest_v[i]:
#                 pbest_v[i], pbest[i] = loss, swarm[i].clone()
#             if loss < gbest_v:
#                 gbest_v, gbest = loss, swarm[i].clone()
        
#         if (it + 1) % 10 == 0:  # 每10次迭代打印一次
#             train_loss, train_acc = evaluate(model, gbest, criterion)
#             test_loss, test_acc = evaluate(model, gbest, criterion, x_test, y_test)
#             print(f"Iteration {it+1}/{iterations}")
#             print(f"  训练集 - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
#             print(f"  测试集 - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")

#     # 使用最佳参数设置模型
#     set_params(model, gbest)
    
#     # 最终评估
#     train_loss, train_acc = evaluate(model, gbest, criterion)
#     test_loss, test_acc = evaluate(model, gbest, criterion, x_test, y_test)
    
#     train_time = time.time() - start_time
#     return train_loss, train_acc, test_loss, test_acc, train_time

# # ------------------------------------------------------------
# # 7. 主程序
# # ------------------------------------------------------------
# if __name__ == "__main__":
#     print("\n" + "="*50)
#     print("MNIST分类实验: Adam vs PSO优化器对比")
#     print("="*50)
    
#     # Adam训练
#     adam_model = SimpleNet()
#     adam_train_loss, adam_train_acc, adam_test_loss, adam_test_acc, adam_time = train_with_adam(adam_model)

#     # PSO训练
#     pso_model = SimpleNet()
#     pso_train_loss, pso_train_acc, pso_test_loss, pso_test_acc, pso_time = train_with_pso(pso_model)

#     # 结果对比
#     print("\n" + "="*50)
#     print("最终结果对比:")
#     print("="*50)
#     print(f"Adam优化器:")
#     print(f"  训练时间: {adam_time:.2f}秒")
#     print(f"  训练集 - Loss: {adam_train_loss:.4f}, Acc: {adam_train_acc*100:.2f}%")
#     print(f"  测试集 - Loss: {adam_test_loss:.4f}, Acc: {adam_test_acc*100:.2f}%")
#     print("-"*50)
#     print(f"PSO优化器:")
#     print(f"  训练时间: {pso_time:.2f}秒")
#     print(f"  训练集 - Loss: {pso_train_loss:.4f}, Acc: {pso_train_acc*100:.2f}%")
#     print(f"  测试集 - Loss: {pso_test_loss:.4f}, Acc: {pso_test_acc*100:.2f}%")
#     print("="*50)

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.decomposition import PCA

# ------------------------------------------------------------
# 1. 设置设备和随机种子
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ------------------------------------------------------------
# 2. 数据准备
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),              # HWC [0,1]
    transforms.Normalize((0.1307,), (0.3081,))   # 标准化
])

# 加载训练集和测试集
mnist_train = datasets.MNIST(root="./data", train=True, download=True,
                           transform=transform)
mnist_test = datasets.MNIST(root="./data", train=False, download=True,
                          transform=transform)

# 使用训练集的子集
subset_size = 2_000                     # 控制数据集大小
subset_indices = list(range(subset_size))
mnist_subset = Subset(mnist_train, subset_indices)

# PCA降维
n_components = 100  # 降维到100维
pca = PCA(n_components=n_components)

# 对训练数据进行拟合和转换
x_train_flat = torch.stack([d[0].view(-1) for d in mnist_subset]).cpu().numpy()
pca.fit(x_train_flat)
x_train = torch.tensor(pca.transform(x_train_flat), device=device)
y_train = torch.tensor([d[1] for d in mnist_subset], device=device)      # [N]

# 对测试数据进行转换
x_test_flat = torch.stack([d[0].view(-1) for d in mnist_test]).cpu().numpy()
x_test = torch.tensor(pca.transform(x_test_flat), device=device)
y_test = torch.tensor([d[1] for d in mnist_test], device=device)      # [N]

print(f"训练集大小: {len(mnist_subset)} 样本")
print(f"测试集大小: {len(mnist_test)} 样本")
print(f"PCA降维后的特征维度: {n_components}")

# ------------------------------------------------------------
# 3. 网络定义
# ------------------------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_components, 64),  # 输入维度改为n_components
            nn.ReLU(),
            nn.Linear(64, 32),           # 减少中间层维度
            nn.ReLU(),
            nn.Linear(32, 10)            # 输出层保持不变
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# 4. 工具函数
# ------------------------------------------------------------
def flat_params(model):
    """将模型参数展平成一维向量"""
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_params(model, flat):
    """将一维向量恢复成模型参数"""
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx+n].view_as(p))
        idx += n

def evaluate(model, params, criterion, x=None, y=None):
    """评估给定参数下的模型性能"""
    x = x_train if x is None else x
    y = y_train if y is None else y
    set_params(model, params)
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y).item()
        acc = (logits.argmax(1) == y).float().mean().item()
        return loss, acc

# ------------------------------------------------------------
# 5. Adam训练
# ------------------------------------------------------------
def train_with_adam(model, lr=1e-3, epochs=300):
    print("\nAdam训练开始...")
    start_time = time.time()
    
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        opt.step()
        
        if (epoch + 1) % 50 == 0:
            train_loss, train_acc = evaluate(model, flat_params(model), criterion)
            test_loss, test_acc = evaluate(model, flat_params(model), criterion, x_test, y_test)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  训练集 - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
            print(f"  测试集 - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")
    
    # 最终评估
    train_loss, train_acc = evaluate(model, flat_params(model), criterion)
    test_loss, test_acc = evaluate(model, flat_params(model), criterion, x_test, y_test)
    
    train_time = time.time() - start_time
    return train_loss, train_acc, test_loss, test_acc, train_time

# ------------------------------------------------------------
# 6. PSO训练
# ------------------------------------------------------------
def train_with_pso(model, n_particles=300, w=0.4, c1=1.2, c2=1.2, iterations=500):
    print("\nPSO训练开始...")
    start_time = time.time()
    
    model.to(device)
    dim = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 初始化粒子群
    swarm = [flat_params(model).clone()+0.1*torch.randn(dim, device=device)
             for _ in range(n_particles)]
    vel = [torch.zeros(dim, device=device) for _ in range(n_particles)]
    pbest = [s.clone() for s in swarm]
    pbest_v = [float("inf")]*n_particles
    gbest, gbest_v = None, float("inf")

    # PSO迭代
    for it in range(iterations):
        for i in range(n_particles):
            # 更新速度和位置
            r1, r2 = torch.rand(dim, device=device), torch.rand(dim, device=device)
            social = 0 if gbest is None else c2*r2*(gbest - swarm[i])
            vel[i] = w*vel[i] + c1*r1*(pbest[i]-swarm[i]) + social
            swarm[i] = swarm[i] + vel[i]

            # 评估新位置
            loss, _ = evaluate(model, swarm[i], criterion)
            if loss < pbest_v[i]:
                pbest_v[i], pbest[i] = loss, swarm[i].clone()
            if loss < gbest_v:
                gbest_v, gbest = loss, swarm[i].clone()
        
        if (it + 1) % 10 == 0:  # 每10次迭代打印一次
            train_loss, train_acc = evaluate(model, gbest, criterion)
            test_loss, test_acc = evaluate(model, gbest, criterion, x_test, y_test)
            print(f"Iteration {it+1}/{iterations}")
            print(f"  训练集 - Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}%")
            print(f"  测试集 - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")

    # 使用最佳参数设置模型
    set_params(model, gbest)
    
    # 最终评估
    train_loss, train_acc = evaluate(model, gbest, criterion)
    test_loss, test_acc = evaluate(model, gbest, criterion, x_test, y_test)
    
    train_time = time.time() - start_time
    return train_loss, train_acc, test_loss, test_acc, train_time

# ------------------------------------------------------------
# 7. 主程序
# ------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("MNIST分类实验: Adam vs PSO优化器对比")
    print("="*50)
    
    # Adam训练
    adam_model = SimpleNet()
    adam_train_loss, adam_train_acc, adam_test_loss, adam_test_acc, adam_time = train_with_adam(adam_model)

    # PSO训练
    pso_model = SimpleNet()
    pso_train_loss, pso_train_acc, pso_test_loss, pso_test_acc, pso_time = train_with_pso(pso_model)

    # 结果对比
    print("\n" + "="*50)
    print("最终结果对比:")
    print("="*50)
    print(f"Adam优化器:")
    print(f"  训练时间: {adam_time:.2f}秒")
    print(f"  训练集 - Loss: {adam_train_loss:.4f}, Acc: {adam_train_acc*100:.2f}%")
    print(f"  测试集 - Loss: {adam_test_loss:.4f}, Acc: {adam_test_acc*100:.2f}%")
    print("-"*50)
    print(f"PSO优化器:")
    print(f"  训练时间: {pso_time:.2f}秒")
    print(f"  训练集 - Loss: {pso_train_loss:.4f}, Acc: {pso_train_acc*100:.2f}%")
    print(f"  测试集 - Loss: {pso_test_loss:.4f}, Acc: {pso_test_acc*100:.2f}%")
    print("="*50)