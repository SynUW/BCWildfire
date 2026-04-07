import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import time
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, f1_score, precision_score, recall_score, confusion_matrix
import random
from torch.nn import functional as F
import tempfile
from torch.optim.lr_scheduler import StepLR
import tifffile
from PIL import Image
import h5py
import re
import multiprocessing as mp
from functools import partial


# import os
# os.environ["TMPDIR"] = "/mnt/raid/zhengsen/tmp"
# os.makedirs(os.environ["TMPDIR"], exist_ok=True)
# tempfile.tempdir = "/mnt/raid/zhengsen/tmp"


class Config:
    epoches = 100

# ========== 数据集路径集中管理 ========== #
DATA_DIR_2020 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/samples_2020_sampled"
DATA_DIR_2021 = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/samples_2021_sampled"

# 2020年数据路径
TRAIN_PATH_2020 = os.path.join(DATA_DIR_2020, "train.h5")
VAL_PATH_2020 = os.path.join(DATA_DIR_2020, "val.h5")
TEST_PATH_2020 = os.path.join(DATA_DIR_2020, "test.h5")

# 2021年数据路径
TEST_PATH_2021 = os.path.join(DATA_DIR_2021, "test.h5")
# ========== END ========== #


def load_sample(args):
    """加载单个样本的函数"""
    data_path, key, preload = args
    if preload:
        with h5py.File(data_path, 'r') as f:
            data = f[key][:]
        return key, data
    else:
        return key, None


class WildfireDataset(Dataset):
    def __init__(self, data_path, sample_ratio=1.0, preload=True, num_workers=None):
        """
        初始化数据集
        参数:
            data_path: H5文件路径
            sample_ratio: 采样比例（样本级）
            preload: 是否预加载数据到内存
            num_workers: 并行加载的进程数，None表示使用CPU核心数
        """
        self.data_path = data_path
        self.preload = preload
        self.num_workers = num_workers if num_workers is not None else mp.cpu_count()

        # 首先统计总数据量
        print("统计总数据量...")
        with h5py.File(data_path, 'r') as f:
            self.keys = list(f.keys())
            total_samples = len(self.keys)
            print(f"总数据量: {total_samples} 个样本")

        # 根据采样比例选择样本
        num_samples = max(1, int(total_samples * sample_ratio))
        self.keys = random.sample(self.keys, num_samples)
        print(f"采样比例: {sample_ratio}")
        print(f"采样后样本数: {len(self.keys)}")

        # 预加载选中的样本数据
        if self.preload:
            print("预加载数据到内存...")
            # 使用多进程并行加载数据
            with mp.Pool(self.num_workers) as pool:
                # 准备参数列表
                args_list = [(data_path, key, preload) for key in self.keys]
                results = list(tqdm(
                    pool.imap(load_sample, args_list),
                    total=len(self.keys),
                    desc="加载数据"
                ))
            self.data_cache = {key: data for key, data in results if data is not None}
        else:
            self.data_cache = None

        print(f"使用进程数: {self.num_workers}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        
        if self.preload:
            data = self.data_cache[key]
        else:
            with h5py.File(self.data_path, 'r') as f:
                data = f[key][:]

        # 数据格式：时间序列数据 + 标签
        x = data[:-1]  # 时间序列数据
        y = data[-1]   # 标签
        
        # 二值化标签
        y = 1 if y >= 1 else 0
        
        # 将y转换为one-hot编码
        y_onehot = np.zeros(2)  # 创建长度为2的零向量
        y_onehot[y] = 1  # 在对应位置设置为1
        
        return torch.FloatTensor(x), torch.FloatTensor(y_onehot)


# ========== Mamba相关依赖类 =============
class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms_x = norm_x * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms_x + self.eps))

# ========== ChannelWiseAttention 注意力类 =============
class ChannelWiseAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, sparsity_ratio=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.sparsity_ratio = sparsity_ratio

        self.to_qkv = nn.Linear(d_model, d_model * 3)
        self.scale = (self.head_dim) ** -0.5

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.to_qkv(x.reshape(B * L, D)).reshape(B * L, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B*L, num_heads, head_dim]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B*L, num_heads, num_heads]

        k_heads = max(1, int(self.num_heads * self.sparsity_ratio))
        topk_scores, topk_indices = torch.topk(attn_scores, k=k_heads, dim=-1)
        sparse_attn = torch.zeros_like(attn_scores).scatter_(-1, topk_indices, torch.softmax(topk_scores, dim=-1))

        out = (sparse_attn @ v).transpose(1, 2).reshape(B, L, self.d_model)
        return out

# ========== 新版 SparseDeformableMambaBlock =============
class SparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3, num_heads=8):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio
        self.num_heads = num_heads

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

        self.A = nn.Parameter(torch.zeros(d_state, d_state))
        self.B = nn.Parameter(torch.zeros(1, 1, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

        self.channel_attention = ChannelWiseAttention(self.expanded_dim, num_heads=self.num_heads, sparsity_ratio=self.sparsity_ratio)

    def forward(self, x):
        B, L, C = x.shape
        residual = x

        x_norm = self.norm(x)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

        # 注意力稀疏选择
        x_attn = self.channel_attention(x_proj)  # [B, L, expanded_dim]

        # Conv处理
        x_conv = x_attn.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :L]
        x_conv = x_conv.transpose(1, 2)

        # SSM处理
        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        outputs = []
        for t in range(L):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B) * x_t
            h = torch.matmul(h, self.A.T) + Bx
            out_t = (h * torch.sigmoid(self.C.unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x_processed = torch.stack(outputs, dim=1)
        x_processed = self.proj_out(x_processed)

        return x_processed + residual

# ========== WildfirePredictor（Mamba版） =============
class WildfirePredictor(nn.Module):
    def __init__(self, seq_len=7305, d_state=32, d_conv=4, expand=2, sparsity_ratio=0.3, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(366, 32)
        self.mamba_blocks = nn.Sequential(
            *[SparseDeformableMambaBlock(
                dim=32,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                sparsity_ratio=0.5
            ) for _ in range(num_layers)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=1)  # 保证输出在[0,1]区间
        )

    def forward(self, x):
        # x: [batch, seq_len]
        B, L = x.shape
        device = x.device  # 获取输入张量的设备
        
        # 计算需要padding的长度
        target_length = ((L + 19) // 20) * 20  # 向上取整到20的倍数
        if L % 20 != 0:
            # 需要padding
            padding_length = target_length - L
            x = torch.nn.functional.pad(x, (0, padding_length), mode='constant', value=0)
        
        # reshape to 20*366
        x = x.reshape(B, 20, -1)  # [batch, n, 20]
        x = self.input_proj(x)  # [batch, n, 32]
        
        # Mamba
        x = self.mamba_blocks(x)  # [batch, n, 32]
        x = x.permute(0, 2, 1)  # [batch, 32, n]
        x = self.pool(x).squeeze(-1)  # [batch, 32]
        out = self.fc(x)  # [batch, 2]
        return out


def create_data_loaders(train_path, val_path, test_path, sample_ratio=1.0, batch_size=64):
    """
    创建数据加载器

    参数:
        train_path: 训练集H5文件路径
        val_path: 验证集H5文件路径
        test_path: 测试集H5文件路径
        sample_ratio: 采样比例（样本级）
        batch_size: 批次大小
    """
    # 创建训练和验证数据集（预加载）
    print("Loading training and validation datasets...")
    train_dataset = WildfireDataset(train_path, sample_ratio=sample_ratio, preload=True)
    val_dataset = WildfireDataset(val_path, sample_ratio=sample_ratio, preload=True)
    # 测试集不预加载
    test_dataset = WildfireDataset(test_path, sample_ratio=sample_ratio, preload=False)

    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, max(1, len(train_dataset))),
                              shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, max(1, len(val_dataset))),
                            num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=min(batch_size, max(1, len(test_dataset))),
                             num_workers=16, pin_memory=True)

    print(f"\n数据集统计:")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("数据集样本数为0，请检查数据清洗、采样比例或数据路径！")

    return train_loader, val_loader, test_loader

def evaluate_model(model, data_loader, device, criterion):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="评估中"):
            x = x.to(device)
            y = y.to(device)
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            total_loss += loss.item()
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    total_loss /= len(data_loader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 将预测概率转换为类别
    pred_classes = np.argmax(all_preds, axis=1)  # 从概率转换为类别
    true_classes = np.argmax(all_targets, axis=1)  # 从one-hot编码转换回类别
    
    # 打印预测分布
    print("\n预测分布:")
    print(f"预测为正类的样本数: {np.sum(pred_classes == 1)}")
    print(f"预测为负类的样本数: {np.sum(pred_classes == 0)}")
    print(f"实际为正类的样本数: {np.sum(true_classes == 1)}")
    print(f"实际为负类的样本数: {np.sum(true_classes == 0)}")
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(true_classes, pred_classes).ravel()
    print("\n混淆矩阵:")
    print(f"真负例 (TN): {tn}")
    print(f"假正例 (FP): {fp}")
    print(f"假负例 (FN): {fn}")
    print(f"真正例 (TP): {tp}")
    
    # 计算评估指标
    f1 = f1_score(true_classes, pred_classes, average='binary')
    precision = precision_score(true_classes, pred_classes, average='binary')
    recall = recall_score(true_classes, pred_classes, average='binary')
    
    return total_loss, f1, precision, recall

def main():
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 创建数据加载器
    print("\n开始加载数据...")
    train_loader, val_loader, test_loader_2020 = create_data_loaders(
        train_path=TRAIN_PATH_2020,
        val_path=VAL_PATH_2020,
        test_path=TEST_PATH_2020,
        sample_ratio=1.0,  # 使用全部数据
        batch_size=128
    )

    # 打印每个数据加载器的批次数量
    print("\n数据加载器信息:")
    print(f"训练集批次数量: {len(train_loader)}")
    print(f"验证集批次数量: {len(val_loader)}")
    print(f"2020年测试集批次数量: {len(test_loader_2020)}")

    # 创建模型并移动到指定设备
    print("\n初始化模型...")
    model = WildfirePredictor().to(device)

    # 定义损失函数和优化器
    criterion_pred = torch.nn.BCELoss()  # 使用二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)

    # 训练模型
    print("\n开始训练...")
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_model_state_loss = None
    best_model_state_f1 = None
    
    for epoch in range(Config.epoches):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{Config.epoches} [Train]')
        for x, y in train_pbar:
            # 确保数据在正确的设备上
            x = x.to(device)
            y = y.to(device)
            
            # 前向传播
            y_pred = model(x)
            
            # 计算损失
            loss = criterion_pred(y_pred, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录训练信息
            train_loss += loss.item()
            train_preds.extend(y_pred.detach().cpu().numpy())
            train_targets.extend(y.cpu().numpy())
            
            # 更新进度条
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        train_loss /= len(train_loader)
        train_preds = np.array(train_preds)
        train_targets = np.array(train_targets)
        
        # 计算训练指标
        train_pred_classes = np.argmax(train_preds, axis=1)  # 从概率转换为类别
        train_true_classes = np.argmax(train_targets, axis=1)  # 从one-hot编码转换回类别
        
        train_f1 = f1_score(train_true_classes, train_pred_classes, average='binary')
        train_precision = precision_score(train_true_classes, train_pred_classes, average='binary')
        train_recall = recall_score(train_true_classes, train_pred_classes, average='binary')
        
        # 验证阶段
        val_loss, val_f1, val_precision, val_recall = evaluate_model(model, val_loader, device, criterion_pred)
        
        # 打印训练和验证结果
        print(f"\nEpoch {epoch+1}/{Config.epoches}:")
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        
        # 保存最佳模型（以val_loss为准）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_loss = model.state_dict().copy()
            print(f"保存新的最佳Loss模型！验证Loss: {best_val_loss:.4f}")
            
        # 保存最佳模型（以val_f1为准）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state_f1 = model.state_dict().copy()
            print(f"保存新的最佳F1模型！验证F1: {best_val_f1:.4f}")

    # 保存两个最佳模型
    if best_model_state_loss is not None:
        torch.save(best_model_state_loss, 'wildfire_predictor_best_loss.pth')
        print("\n已保存最佳Loss模型: wildfire_predictor_best_loss.pth")
    else:
        print("\n警告：没有找到最佳Loss模型状态！")
        
    if best_model_state_f1 is not None:
        torch.save(best_model_state_f1, 'wildfire_predictor_best_f1.pth')
        print("已保存最佳F1模型: wildfire_predictor_best_f1.pth")
    else:
        print("警告：没有找到最佳F1模型状态！")

    # 使用最佳Loss模型进行测试
    print("\n使用最佳Loss模型进行测试...")
    model.load_state_dict(best_model_state_loss)
    print("\n开始测试2020年数据...")
    test_loss_2020, test_f1_2020, test_precision_2020, test_recall_2020 = evaluate_model(model, test_loader_2020, device, criterion_pred)
    print("\n2020年测试结果:")
    print(f"Test Loss: {test_loss_2020:.4f}")
    print(f"Test F1: {test_f1_2020:.4f}")
    print(f"Test Precision: {test_precision_2020:.4f}")
    print(f"Test Recall: {test_recall_2020:.4f}")

    # 加载2021年测试数据
    print("\n加载2021年测试数据...")
    test_dataset_2021 = WildfireDataset(TEST_PATH_2021, sample_ratio=1.0, preload=False)
    test_loader_2021 = DataLoader(test_dataset_2021, batch_size=128, num_workers=16, pin_memory=True)
    print(f"2021年测试集批次数量: {len(test_loader_2021)}")

    print("\n开始测试2021年数据...")
    test_loss_2021, test_f1_2021, test_precision_2021, test_recall_2021 = evaluate_model(model, test_loader_2021, device, criterion_pred)
    print("\n2021年测试结果:")
    print(f"Test Loss: {test_loss_2021:.4f}")
    print(f"Test F1: {test_f1_2021:.4f}")
    print(f"Test Precision: {test_precision_2021:.4f}")
    print(f"Test Recall: {test_recall_2021:.4f}")

    # 使用最佳F1模型进行测试
    print("\n使用最佳F1模型进行测试...")
    model.load_state_dict(best_model_state_f1)
    print("\n开始测试2020年数据...")
    test_loss_2020_f1, test_f1_2020_f1, test_precision_2020_f1, test_recall_2020_f1 = evaluate_model(model, test_loader_2020, device, criterion_pred)
    print("\n2020年测试结果:")
    print(f"Test Loss: {test_loss_2020_f1:.4f}")
    print(f"Test F1: {test_f1_2020_f1:.4f}")
    print(f"Test Precision: {test_precision_2020_f1:.4f}")
    print(f"Test Recall: {test_recall_2020_f1:.4f}")

    print("\n开始测试2021年数据...")
    test_loss_2021_f1, test_f1_2021_f1, test_precision_2021_f1, test_recall_2021_f1 = evaluate_model(model, test_loader_2021, device, criterion_pred)
    print("\n2021年测试结果:")
    print(f"Test Loss: {test_loss_2021_f1:.4f}")
    print(f"Test F1: {test_f1_2021_f1:.4f}")
    print(f"Test Precision: {test_precision_2021_f1:.4f}")
    print(f"Test Recall: {test_recall_2021_f1:.4f}")


if __name__ == "__main__":
    main()