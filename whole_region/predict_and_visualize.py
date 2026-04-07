import os
import numpy as np
import torch
import h5py
from tqdm import tqdm
import re
import argparse
from train_wildfire import WildfirePredictor, WildfireDataset
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

# Use default matplotlib font
plt.rcParams['axes.unicode_minus'] = False  # For proper display of minus signs

def load_model(model_path, device):
    """加载模型"""
    model = WildfirePredictor().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_dataset(model, data_path, device, batch_size=128):
    """对整个数据集进行预测"""
    # 创建数据集
    dataset = WildfireDataset(data_path, sample_ratio=1.0, preload=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True
    )
    
    # 存储预测结果和真实值
    all_preds = []
    all_targets = []
    all_positions = []
    all_inputs = []  # 存储输入数据
    
    # 进行预测
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="预测中"):
            x = x.to(device)
            y = y.to(device)
            
            # 获取预测结果
            y_pred = model(x)
            
            # 存储结果
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_inputs.extend(x.cpu().numpy())  # 存储输入数据
            
            # 存储位置信息
            pos_x = x[:, -2].cpu().numpy()
            pos_y = x[:, -1].cpu().numpy()
            all_positions.extend(list(zip(pos_x, pos_y)))
    
    return np.array(all_preds), np.array(all_targets), np.array(all_positions), np.array(all_inputs)

def calculate_metrics(preds, targets):
    """计算评价指标"""
    # 将预测值二值化
    preds_binary = (preds > 0.5).astype(int)
    targets_binary = (targets >= 1).astype(int)
    
    # 计算各项指标
    f1 = f1_score(targets_binary, preds_binary)
    precision = precision_score(targets_binary, preds_binary, zero_division=0)
    recall = recall_score(targets_binary, preds_binary, zero_division=0)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def create_prediction_map(preds, positions, shape=(2602, 5525)):
    """创建预测图"""
    pred_map = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)  # 创建掩码
    
    for (x, y), pred in zip(positions, preds):
        x, y = int(x), int(y)
        if 0 <= x < shape[0] and 0 <= y < shape[1]:  # 确保坐标在有效范围内
            pred_map[x, y] = pred[0]
            mask[x, y] = True
    
    # 将没有值的地方设置为NaN
    pred_map[~mask] = np.nan
    return pred_map

def create_ground_truth_map(targets, positions, shape=(2602, 5525)):
    """创建真实值图"""
    gt_map = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)  # 创建掩码
    
    for (x, y), target in zip(positions, targets):
        x, y = int(x), int(y)
        if 0 <= x < shape[0] and 0 <= y < shape[1]:  # 确保坐标在有效范围内
            gt_map[x, y] = target[0]
            mask[x, y] = True
    
    # 将没有值的地方设置为NaN
    gt_map[~mask] = np.nan
    return gt_map

def create_input_summary_map(inputs, positions, shape=(2602, 5525)):
    """创建输入数据汇总图"""
    # 计算每个位置所有输入特征的总和
    input_sum = np.sum(inputs[:, :-2], axis=1)  # 排除最后两个位置特征
    summary_map = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)
    
    for (x, y), value in zip(positions, input_sum):
        x, y = int(x), int(y)
        if 0 <= x < shape[0] and 0 <= y < shape[1]:
            summary_map[x, y] = value
            mask[x, y] = True
    
    summary_map[~mask] = np.nan
    return summary_map

def create_sampling_map(positions, shape=(2602, 5525)):
    """创建采样位置图"""
    sampling_map = np.zeros(shape, dtype=bool)
    
    for x, y in positions:
        x, y = int(x), int(y)
        if 0 <= x < shape[0] and 0 <= y < shape[1]:
            sampling_map[x, y] = True
    
    return sampling_map

def create_confusion_matrix_map(preds, targets, positions, shape=(2602, 5525)):
    """创建混淆矩阵可视化图
    0: TN (真阴性，预测为0，实际为0)
    1: FP (假阳性，预测为1，实际为0)
    2: FN (假阴性，预测为0，实际为1)
    3: TP (真阳性，预测为1，实际为1)
    """
    confusion_map = np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)
    
    # 将预测值和真实值二值化
    preds_binary = (preds > 0.5).astype(int)
    targets_binary = (targets >= 1).astype(int)
    
    for (x, y), pred, target in zip(positions, preds_binary, targets_binary):
        x, y = int(x), int(y)
        if 0 <= x < shape[0] and 0 <= y < shape[1]:
            # 计算混淆矩阵类别
            if pred == 0 and target == 0:  # TN
                confusion_map[x, y] = 0
            elif pred == 1 and target == 0:  # FP
                confusion_map[x, y] = 1
            elif pred == 0 and target == 1:  # FN
                confusion_map[x, y] = 2
            else:  # TP
                confusion_map[x, y] = 3
            mask[x, y] = True
    
    confusion_map[~mask] = np.nan
    return confusion_map

def visualize_maps(pred_map, gt_map, input_summary_map, sampling_map, confusion_map, output_dir, year):
    """Visualize prediction and ground truth maps"""
    # Get image dimensions
    height, width = pred_map.shape
    
    # Create custom colormap
    colors = [(1, 1, 1), (1, 0, 0)]  # from white to red
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    # Create figure for prediction and ground truth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot prediction map
    im1 = ax1.imshow(pred_map, cmap=cmap, vmin=0, vmax=1)
    ax1.set_title('Prediction Map')
    
    # Plot ground truth map
    im2 = ax2.imshow(gt_map, cmap=cmap, vmin=0, vmax=1)
    ax2.set_title('Ground Truth Map')
    
    # Add axis labels
    for ax in [ax1, ax2]:
        ax.set_xlabel(f'Width ({width})')
        ax.set_ylabel(f'Height ({height})')
    
    # Save comparison figure
    plt.savefig(os.path.join(output_dir, f'prediction_map_{year}.png'), dpi=3000, bbox_inches='tight')
    plt.close()
    
    # Save prediction map as separate PNG
    plt.figure(figsize=(10, 10))
    plt.imshow(pred_map, cmap=cmap, vmin=0, vmax=1)
    plt.title('Prediction Map')
    plt.xlabel(f'Width ({width})')
    plt.ylabel(f'Height ({height})')
    plt.savefig(os.path.join(output_dir, f'prediction_{year}.png'), dpi=3000, bbox_inches='tight')
    plt.close()
    
    # Save ground truth map as separate PNG
    plt.figure(figsize=(10, 10))
    plt.imshow(gt_map, cmap=cmap, vmin=0, vmax=1)
    plt.title('Ground Truth Map')
    plt.xlabel(f'Width ({width})')
    plt.ylabel(f'Height ({height})')
    plt.savefig(os.path.join(output_dir, f'ground_truth_{year}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save input summary map
    plt.figure(figsize=(10, 10))
    plt.imshow(input_summary_map, cmap='viridis')
    plt.title('Input Features Summary')
    plt.xlabel(f'Width ({width})')
    plt.ylabel(f'Height ({height})')
    plt.savefig(os.path.join(output_dir, f'input_summary_{year}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save sampling locations map
    plt.figure(figsize=(10, 10))
    plt.imshow(sampling_map, cmap='binary')
    plt.title('Sampling Locations')
    plt.xlabel(f'Width ({width})')
    plt.ylabel(f'Height ({height})')
    plt.savefig(os.path.join(output_dir, f'sampling_locations_{year}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confusion matrix map
    plt.figure(figsize=(10, 10))
    # 创建自定义颜色映射：TN(白色), FP(蓝色), FN(绿色), TP(红色)
    confusion_colors = ['white', 'blue', 'green', 'red']
    confusion_cmap = mpl.colors.ListedColormap(confusion_colors)
    plt.imshow(confusion_map, cmap=confusion_cmap, vmin=0, vmax=3)
    plt.title('Confusion Matrix Map\nWhite: TN, Blue: FP, Green: FN, Red: TP')
    plt.xlabel(f'Width ({width})')
    plt.ylabel(f'Height ({height})')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{year}.png'), dpi=3000, bbox_inches='tight')
    plt.close()
    
    # Print mask statistics
    pred_mask = ~np.isnan(pred_map)
    gt_mask = ~np.isnan(gt_map)
    total_pixels = height * width
    print(f"\nMask Statistics (Year {year}):")
    print(f"Prediction Map Valid Pixels: {np.sum(pred_mask)}")
    print(f"Ground Truth Map Valid Pixels: {np.sum(gt_mask)}")
    print(f"Prediction Map Coverage: {np.sum(pred_mask)/total_pixels*100:.2f}%")
    print(f"Ground Truth Map Coverage: {np.sum(gt_mask)/total_pixels*100:.2f}%")
    
    # Print confusion matrix statistics
    valid_mask = ~np.isnan(confusion_map)
    tn = np.sum(confusion_map[valid_mask] == 0)
    fp = np.sum(confusion_map[valid_mask] == 1)
    fn = np.sum(confusion_map[valid_mask] == 2)
    tp = np.sum(confusion_map[valid_mask] == 3)
    total = tn + fp + fn + tp
    
    print(f"\nConfusion Matrix Statistics (Year {year}):")
    print(f"True Negatives (TN): {tn} ({tn/total*100:.2f}%)")
    print(f"False Positives (FP): {fp} ({fp/total*100:.2f}%)")
    print(f"False Negatives (FN): {fn} ({fn/total*100:.2f}%)")
    print(f"True Positives (TP): {tp} ({tp/total*100:.2f}%)")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用最佳模型进行预测并可视化')
    parser.add_argument('--model_path', type=str, 
                        default='wildfire_predictor.pth',
                        help='模型文件路径')
    parser.add_argument('--data_dir', type=str, 
                        default="/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/resampled_by_year",
                        help='数据文件目录')
    parser.add_argument('--output_dir', type=str,
                        default='prediction_results',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    print("加载模型...")
    model = load_model(args.model_path, device)
    
    # 获取所有H5文件
    h5_files = [f for f in os.listdir(args.data_dir) if f.endswith('.h5')]
    h5_files.sort()
    
    # 处理每个文件
    for h5_file in h5_files:
        year = h5_file.split('_')[1].split('.')[0]
        print(f"\n处理年份: {year}")
        
        # 预测
        data_path = os.path.join(args.data_dir, h5_file)
        preds, targets, positions, inputs = predict_dataset(model, data_path, device)
        
        # 计算评价指标
        metrics = calculate_metrics(preds, targets)
        print(f"评价指标:")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        # 创建预测图和真实图
        pred_map = create_prediction_map(preds, positions)
        gt_map = create_ground_truth_map(targets, positions)
        input_summary_map = create_input_summary_map(inputs, positions)
        sampling_map = create_sampling_map(positions)
        confusion_map = create_confusion_matrix_map(preds, targets, positions)
        
        # 可视化
        visualize_maps(pred_map, gt_map, input_summary_map, sampling_map, confusion_map, args.output_dir, year)
        
        # 保存评价指标
        with open(os.path.join(args.output_dir, f'metrics_{year}.txt'), 'w') as f:
            f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
    
    print("\n处理完成！")

if __name__ == "__main__":
    main() 