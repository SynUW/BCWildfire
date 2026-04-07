import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
import time
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import random
from torch.nn import functional as F


class WildfireDataset(Dataset):
    def __init__(self, data_dir, start_year=2000, end_year=2023, sequence_length=20, is_test=False):
        """
        初始化数据集
        
        参数:
            data_dir: 数据目录路径
            start_year: 起始年份
            end_year: 结束年份
            sequence_length: 输入序列长度（年数）
            is_test: 是否为测试集（使用2024年数据）
        """
        self.data_dir = data_dir
        self.start_year = start_year
        self.end_year = end_year
        self.sequence_length = sequence_length
        self.is_test = is_test
        
        # 获取所有像素文件
        self.pixel_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        
        # 预加载所有数据到内存
        print("预加载数据到内存...")
        self.data_cache = {}
        for pixel_file in tqdm(self.pixel_files):
            self.data_cache[pixel_file] = np.load(os.path.join(data_dir, pixel_file))
        
        # 计算每个像素的可用样本数
        self.samples = []
        for pixel_file in self.pixel_files:
            pixel_data = self.data_cache[pixel_file]
            # 确保数据长度足够
            if len(pixel_data) >= end_year - start_year + 1:
                if is_test:
                    # 测试集：使用2023年的数据预测2024年
                    self.samples.append((pixel_file, end_year - start_year - sequence_length))
                else:
                    # 训练集：使用2000-2022年的数据
                    n_samples = end_year - start_year - sequence_length
                    for i in range(n_samples):
                        self.samples.append((pixel_file, i))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        pixel_file, sample_idx = self.samples[idx]
        pixel_data = self.data_cache[pixel_file]
        
        # 获取输入序列（20年数据）
        start_idx = sample_idx
        end_idx = start_idx + self.sequence_length
        x = pixel_data[start_idx:end_idx]
        
        # 获取目标值（下一年的野火次数）
        y = pixel_data[end_idx]
        
        return torch.FloatTensor(x), torch.FloatTensor([y])


class WildfirePredictor(nn.Module):
    def __init__(self, input_size=20):
        """
        初始化模型
        
        参数:
            input_size: 输入序列长度（年数）
        """
        super(WildfirePredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        # 添加注意力机制
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.unsqueeze(-1)  # 添加特征维度
        # x shape: (batch_size, sequence_length, 1)
        
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        
        # 计算注意力权重
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # 应用注意力
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 通过全连接层
        out = self.fc(context)
        return out


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        # 计算MSE损失
        mse_loss = F.mse_loss(pred, target, reduction='none')
        
        # 计算focal loss
        pt = torch.exp(-mse_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * mse_loss
        
        return focal_loss.mean()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    训练模型
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备
    """
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(y.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_r2 = r2_score(train_targets, train_preds)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_r2 = r2_score(val_targets, val_preds)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train R²: {train_r2:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
    
    return best_model_state


def evaluate_model(model, test_loader, device):
    """
    评估模型
    
    参数:
        model: 模型
        test_loader: 测试数据加载器
        device: 训练设备
    """
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc='Evaluating'):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(y.cpu().numpy())
    
    # 计算评估指标
    r2 = r2_score(test_targets, test_preds)
    mse = mean_squared_error(test_targets, test_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_targets, test_preds)
    
    print('\nTest Results:')
    print(f'R² Score: {r2:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    
    return r2, mse, rmse, mae


def create_data_loaders(data_dir, train_ratio=0.8, sample_ratio=1.0, batch_size=64):
    """
    创建数据加载器
    
    参数:
        data_dir: 数据目录路径
        train_ratio: 训练集比例
        sample_ratio: 采样比例（使用多少比例的数据）
        batch_size: 批次大小
    """
    # 创建完整数据集
    print("Loading datasets...")
    train_val_dataset = WildfireDataset(data_dir, end_year=2022)
    test_dataset = WildfireDataset(data_dir, end_year=2023, is_test=True)
    
    # 计算采样数量
    total_samples = len(train_val_dataset)
    sample_size = int(total_samples * sample_ratio)
    
    # 随机采样
    indices = list(range(total_samples))
    random.shuffle(indices)
    sampled_indices = indices[:sample_size]
    
    # 创建采样后的数据集
    sampled_dataset = Subset(train_val_dataset, sampled_indices)
    
    # 划分训练集和验证集
    train_size = int(train_ratio * len(sampled_dataset))
    val_size = len(sampled_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        sampled_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=16, pin_memory=True)
    
    print(f"\n数据集统计:")
    print(f"总样本数: {total_samples}")
    print(f"采样后样本数: {sample_size}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def main():
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 数据目录
    data_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/single_pixel_data_without_zero"
    
    # 创建数据加载器（使用0.01%的数据）
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_dir,
        train_ratio=0.8,  # 训练集占采样数据的80%
        sample_ratio=0.0000001,  # 使用0.01%的总数据
        batch_size=64
    )
    
    # 创建模型
    model = WildfirePredictor().to(device)
    
    # 定义损失函数和优化器
    criterion = FocalLoss(alpha=0.25, gamma=2)  # 使用Focal Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 训练模型
    print("\nStarting training...")
    best_model_state = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=50, device=device
    )
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 评估模型
    print("\nEvaluating model on 2024 data...")
    evaluate_model(model, test_loader, device)
    
    # 保存模型
    torch.save(model.state_dict(), 'wildfire_predictor.pth')
    print("\nModel saved as 'wildfire_predictor.pth'")


if __name__ == "__main__":
    main()