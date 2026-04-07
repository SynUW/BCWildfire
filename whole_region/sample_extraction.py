import os
import numpy as np
from osgeo import gdal
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from scipy.spatial import cKDTree
import time
from multiprocessing import Pool, cpu_count

def find_max_in_neighborhood(data, center_row, center_col, window_size=10, max_attempts=3, valid_coords=None):
    """
    在中心点周围的窗口中寻找最大值点，如果周围有背景值则尝试其他窗口大小
    
    参数:
        data: 输入数据数组
        center_row, center_col: 中心点坐标
        window_size: 初始窗口大小（半径）
        max_attempts: 最大尝试次数
        valid_coords: 有效的坐标集合，如果提供，只在这些坐标中寻找最大值
    
    返回:
        (max_row, max_col): 最大值点的坐标
    """
    for attempt in range(max_attempts):
        current_window = window_size - attempt * 2  # 每次尝试减小窗口大小
        if current_window < 3:  # 如果窗口太小，返回原始点
            return center_row, center_col
            
        # 计算窗口边界
        row_start = max(0, center_row - current_window)
        row_end = min(data.shape[0], center_row + current_window + 1)
        col_start = max(0, center_col - current_window)
        col_end = min(data.shape[1], center_col + current_window + 1)
        
        # 提取窗口区域
        window = data[row_start:row_end, col_start:col_end].astype(np.float32)
        
        # 如果提供了有效坐标集合，创建掩码
        if valid_coords is not None:
            valid_mask = np.zeros_like(window, dtype=bool)
            for r, c in valid_coords:
                if row_start <= r < row_end and col_start <= c < col_end:
                    valid_mask[r - row_start, c - col_start] = True
            # 将无效区域的值设为最小值
            window[~valid_mask] = np.finfo(np.float32).min
        
        # 找到最大值的位置
        max_idx = np.unravel_index(np.argmax(window), window.shape)
        
        # 转换回原始坐标
        max_row = row_start + max_idx[0]
        max_col = col_start + max_idx[1]
        
        # 检查最大值点周围是否存在背景值
        if not check_neighborhood_background(data, max_row, max_col):
            return max_row, max_col
    
    # 如果所有尝试都失败，返回原始点
    return center_row, center_col

def check_neighborhood_background(data, row, col, window_size=1):
    """
    检查指定点周围邻域内是否存在背景值（255）
    
    参数:
        data: 输入数据数组
        row, col: 中心点坐标
        window_size: 窗口大小（半径），默认为1，对应3*3的窗口
    
    返回:
        bool: 如果邻域内存在背景值返回True，否则返回False
    """
    # 计算窗口边界
    row_start = max(0, row - window_size)
    row_end = min(data.shape[0], row + window_size + 1)
    col_start = max(0, col - window_size)
    col_end = min(data.shape[1], col + window_size + 1)
    
    # 提取窗口区域
    window = data[row_start:row_end, col_start:col_end]
    
    # 检查是否存在背景值
    return np.any(window == 255)

def process_max_point(args):
    """处理单个点的最大值查找，用于并行计算"""
    data, row, col, valid_coords_set = args
    max_row, max_col = find_max_in_neighborhood(data, row, col, valid_coords=valid_coords_set)
    return (max_row, max_col)

def parallel_find_max_points(data, coords_array, selected_indices, valid_coords_set, n_jobs=None):
    """
    并行查找最大值点
    
    参数:
        data: 输入数据数组
        coords_array: 坐标数组
        selected_indices: 选中的点的索引
        valid_coords_set: 有效坐标集合
        n_jobs: 并行进程数，默认为CPU核心数
    
    返回:
        最大值点坐标列表
    """
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # 准备参数
    args_list = [(data, coords_array[idx][0], coords_array[idx][1], valid_coords_set) 
                 for idx in selected_indices]
    
    # 使用进程池并行处理
    with Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(process_max_point, args_list), 
                          total=len(args_list),
                          desc="并行寻找最大值点"))
    
    return results

def spatial_balanced_sampling(coords, n_samples, min_grid_size=5, max_grid_size=20, min_distance=10, data=None, exclude_coords=None):
    """
    使用KD树进行空间均衡采样，选择最分散的点
    
    参数:
        coords: 坐标列表 [(row1, col1), (row2, col2), ...]
        n_samples: 需要采样的数量
        min_grid_size: 最小网格大小
        max_grid_size: 最大网格大小
        min_distance: 样本点之间的最小距离
        data: 输入数据数组，用于在选中点周围寻找最大值
        exclude_coords: 需要排除的坐标列表
    
    返回:
        采样后的坐标列表
    """
    if len(coords) <= n_samples:
        return coords
    
    print(f"开始采样，目标样本数: {n_samples}, 可用样本数: {len(coords)}")
    
    # 将坐标转换为numpy数组
    coords_array = np.array(coords)
    
    # 如果有需要排除的坐标，创建排除集合
    exclude_set = set()
    if exclude_coords is not None:
        exclude_set = set((row, col) for row, col in exclude_coords)
        print(f"需要排除的坐标数: {len(exclude_set)}")
    
    # 过滤掉需要排除的坐标
    valid_indices = []
    for i, (row, col) in enumerate(coords):
        if (row, col) not in exclude_set:
            valid_indices.append(i)
    
    if len(valid_indices) < n_samples:
        print(f"警告：有效样本点数量不足，需要 {n_samples} 个，实际只有 {len(valid_indices)} 个")
        return [coords[i] for i in valid_indices]
    
    valid_coords = coords_array[valid_indices]
    
    # 使用KD树进行空间采样
    selected_indices = []
    remaining_indices = list(range(len(valid_coords)))
    
    # 首先选择空间中心点
    center = np.mean(valid_coords, axis=0)
    distances_to_center = np.linalg.norm(valid_coords - center, axis=1)
    center_idx = np.argmin(distances_to_center)
    selected_indices.append(center_idx)
    remaining_indices.remove(center_idx)
    
    # 构建KD树
    tree = cKDTree(valid_coords[selected_indices])
    
    # 迭代选择最远的点
    while len(selected_indices) < n_samples and remaining_indices:
        # 计算剩余点到已选点的最小距离
        distances, _ = tree.query(valid_coords[remaining_indices], k=1)
        
        # 选择距离最远的点
        max_dist_idx = np.argmax(distances)
        selected_idx = remaining_indices[max_dist_idx]
        
        # 检查最小距离约束
        if distances[max_dist_idx] >= min_distance:
            selected_indices.append(selected_idx)
            remaining_indices.pop(max_dist_idx)
            
            # 更新KD树
            tree = cKDTree(valid_coords[selected_indices])
        else:
            # 如果无法满足最小距离约束，移除该点
            remaining_indices.pop(max_dist_idx)
    
    print(f"KD树采样后选择的样本数: {len(selected_indices)}")
    
    # 在选中点周围寻找最大值
    if data is not None:
        # 创建有效坐标集合
        valid_coords_set = set((row, col) for row, col in coords)
        # 使用并行处理查找最大值点
        final_coords = parallel_find_max_points(data, coords_array, selected_indices, valid_coords_set)
        print(f"寻找最大值后的样本数: {len(final_coords)}")
        return final_coords
    
    return [coords[valid_indices[i]] for i in selected_indices]

def grid_based_sampling(coords, n_samples, min_grid_size=21, max_grid_size=21, min_distance=10, data=None, exclude_coords=None):
    """
    使用网格进行空间采样，主要用于负样本
    
    参数:
        coords: 坐标列表 [(row1, col1), (row2, col2), ...]
        n_samples: 需要采样的数量
        min_grid_size: 最小网格大小
        max_grid_size: 最大网格大小
        min_distance: 样本点之间的最小距离
        data: 输入数据数组，用于在选中点周围寻找最大值
        exclude_coords: 需要排除的坐标列表
    
    返回:
        采样后的坐标列表
    """
    if len(coords) <= n_samples:
        return coords
    
    print(f"开始网格采样，目标样本数: {n_samples}, 可用样本数: {len(coords)}")
    
    # 将坐标转换为numpy数组
    coords_array = np.array(coords)
    
    # 如果有需要排除的坐标，创建排除集合
    exclude_set = set()
    if exclude_coords is not None:
        exclude_set = set((row, col) for row, col in exclude_coords)
        print(f"需要排除的坐标数: {len(exclude_set)}")
    
    # 计算空间范围
    min_row, min_col = coords_array.min(axis=0)
    max_row, max_col = coords_array.max(axis=0)
    
    # 自适应确定网格大小
    grid_size = min_grid_size
    while grid_size <= max_grid_size:
        # 创建网格
        row_bins = np.linspace(min_row, max_row, grid_size + 1)
        col_bins = np.linspace(min_col, max_col, grid_size + 1)
        
        # 计算每个点所属的网格
        row_indices = np.digitize(coords_array[:, 0], row_bins) - 1
        col_indices = np.digitize(coords_array[:, 1], col_bins) - 1
        grid_indices = row_indices * grid_size + col_indices
        
        # 统计每个网格中的点数
        grid_counts = np.bincount(grid_indices, minlength=grid_size*grid_size)
        non_empty_grids = np.sum(grid_counts > 0)
        
        # 如果非空网格数量合适，退出循环
        if non_empty_grids >= n_samples // 2:
            break
        
        grid_size += 1
    
    print(f"使用网格大小: {grid_size}x{grid_size}")
    print(f"非空网格数量: {non_empty_grids}")
    
    # 计算每个非空网格应该选择的样本数
    samples_per_grid = n_samples // non_empty_grids
    remaining_samples = n_samples % non_empty_grids
    
    # 从每个网格中采样
    selected_indices = []
    for grid_idx in range(grid_size * grid_size):
        grid_points = np.where(grid_indices == grid_idx)[0]
        if len(grid_points) > 0:
            # 计算当前网格应该选择的样本数
            n_grid_samples = samples_per_grid + (1 if remaining_samples > 0 else 0)
            remaining_samples -= 1
            
            # 随机打乱网格内的点
            np.random.shuffle(grid_points)
            
            # 选择点，直到达到目标数量或遍历完所有点
            valid_points = []
            for idx in grid_points:
                point = coords_array[idx]
                point_tuple = (point[0], point[1])
                
                # 检查是否在排除集合中
                if point_tuple in exclude_set:
                    continue
                
                if len(valid_points) == 0:
                    valid_points.append(idx)
                else:
                    # 检查与已选点的距离
                    min_dist = np.min(np.linalg.norm(coords_array[valid_points] - point, axis=1))
                    if min_dist >= min_distance:
                        valid_points.append(idx)
                
                # 如果已经选择了足够的点，就停止
                if len(valid_points) >= n_grid_samples:
                    break
            
            selected_indices.extend(valid_points)
    
    print(f"网格采样后选择的样本数: {len(selected_indices)}")
    
    # 在选中点周围寻找最大值
    if data is not None:
        final_coords = []
        # 创建有效坐标集合
        valid_coords_set = set((row, col) for row, col in coords)
        for idx in selected_indices:
            row, col = coords_array[idx]
            max_row, max_col = find_max_in_neighborhood(data, row, col, valid_coords=valid_coords_set)
            final_coords.append((max_row, max_col))
        print(f"寻找最大值后的样本数: {len(final_coords)}")
        return final_coords
    
    return [coords[i] for i in selected_indices]

def random_sampling(coords, n_samples, min_distance=10, data=None, exclude_coords=None):
    """
    随机采样，主要用于负样本
    
    参数:
        coords: 坐标列表 [(row1, col1), (row2, col2), ...]
        n_samples: 需要采样的数量
        min_distance: 样本点之间的最小距离
        data: 输入数据数组，用于在选中点周围寻找最大值
        exclude_coords: 需要排除的坐标列表
    
    返回:
        采样后的坐标列表
    """
    if len(coords) <= n_samples:
        return coords
    
    print(f"开始随机采样，目标样本数: {n_samples}, 可用样本数: {len(coords)}")
    
    # 将坐标转换为numpy数组
    coords_array = np.array(coords)
    
    # 如果有需要排除的坐标，创建排除集合
    exclude_set = set()
    if exclude_coords is not None:
        exclude_set = set((row, col) for row, col in exclude_coords)
        print(f"需要排除的坐标数: {len(exclude_set)}")
    
    # 随机打乱所有点的顺序
    indices = np.random.permutation(len(coords))
    selected_indices = []
    
    # 选择点，确保满足最小距离约束
    for idx in indices:
        point = coords_array[idx]
        point_tuple = (point[0], point[1])
        
        # 检查是否在排除集合中
        if point_tuple in exclude_set:
            continue
        
        if len(selected_indices) == 0:
            selected_indices.append(idx)
        else:
            # 检查与已选点的距离
            min_dist = np.min(np.linalg.norm(coords_array[selected_indices] - point, axis=1))
            if min_dist >= min_distance:
                selected_indices.append(idx)
        
        # 如果已经选择了足够的点，就停止
        if len(selected_indices) >= n_samples:
            break
    
    print(f"随机采样后选择的样本数: {len(selected_indices)}")
    
    # 直接返回随机采样的结果，不进行最大值点查找
    return [coords[i] for i in selected_indices]

def print_time_cost(start_time, step_name):
    """打印步骤耗时"""
    end_time = time.time()
    cost = end_time - start_time
    print(f"{step_name} 耗时: {cost:.2f} 秒")
    return end_time

def extract_samples(input_tif, output_dir, n_samples=200, seed=42, min_distance=10):
    """
    从sum图像中均匀抽取样本点
    
    参数:
        input_tif: 输入TIF文件路径
        output_dir: 输出目录路径
        n_samples: 每个集合中正负样本的数量
        seed: 随机种子
        min_distance: 样本点之间的最小距离
    """
    total_start_time = time.time()
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取输入文件
    print(f"正在读取文件: {input_tif}")
    start_time = time.time()
    ds = gdal.Open(input_tif)
    if ds is None:
        print("无法打开输入文件")
        return
    
    # 获取地理信息
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    
    # 读取数据
    data = ds.GetRasterBand(1).ReadAsArray()
    start_time = print_time_cost(start_time, "读取文件")
    
    # 获取NoData值
    no_data_value = 255
    
    # 创建掩膜，排除NoData值
    valid_mask = (data != no_data_value)
    
    # 获取正样本和负样本的位置
    start_time = time.time()
    positive_pixels = np.where((data >= 1) & valid_mask)
    negative_pixels = np.where((data < 1) & valid_mask)
    
    # 将坐标转换为列表
    positive_coords = list(zip(positive_pixels[0], positive_pixels[1]))
    negative_coords = list(zip(negative_pixels[0], negative_pixels[1]))
    
    print(f"找到 {len(positive_coords)} 个正样本点")
    print(f"找到 {len(negative_coords)} 个负样本点")
    start_time = print_time_cost(start_time, "获取样本点位置")
    
    # 过滤掉3*3邻域内有背景值的点
    filtered_positive_coords = []
    filtered_negative_coords = []
    
    print("过滤正样本点...")
    start_time = time.time()
    for row, col in tqdm(positive_coords):
        if not check_neighborhood_background(data, row, col):
            filtered_positive_coords.append((row, col))
    start_time = print_time_cost(start_time, "过滤正样本点")
    
    print("过滤负样本点...")
    start_time = time.time()
    for row, col in tqdm(negative_coords):
        if not check_neighborhood_background(data, row, col):
            filtered_negative_coords.append((row, col))
    start_time = print_time_cost(start_time, "过滤负样本点")
    
    print(f"过滤后剩余 {len(filtered_positive_coords)} 个正样本点")
    print(f"过滤后剩余 {len(filtered_negative_coords)} 个负样本点")
    
    if len(filtered_positive_coords) < n_samples * 2 or len(filtered_negative_coords) < n_samples * 2:
        print("错误：过滤后的样本点数量不足以满足训练集和验证集的需求")
        print(f"需要至少 {n_samples * 2} 个正样本和负样本")
        print(f"实际有 {len(filtered_positive_coords)} 个正样本和 {len(filtered_negative_coords)} 个负样本")
        return
    
    # 使用空间均衡采样选择训练集
    print("\n选择训练集样本...")
    start_time = time.time()
    # 正样本使用KD树采样
    train_pos = spatial_balanced_sampling(filtered_positive_coords, n_samples, min_distance=min_distance, data=data)
    if len(train_pos) < n_samples:
        print(f"警告：无法选择足够的训练集正样本，只选择了 {len(train_pos)} 个")
        return
    start_time = print_time_cost(start_time, "选择训练集正样本")
    
    # 负样本使用随机采样
    start_time = time.time()
    train_neg = random_sampling(filtered_negative_coords, n_samples, min_distance=min_distance, data=data)
    if len(train_neg) < n_samples:
        print(f"警告：无法选择足够的训练集负样本，只选择了 {len(train_neg)} 个")
        return
    start_time = print_time_cost(start_time, "选择训练集负样本")
    
    # 从原始样本集中删除训练集使用的样本
    print("删除训练集使用的样本...")
    start_time = time.time()
    train_set = set(train_pos + train_neg)
    filtered_positive_coords = [p for p in filtered_positive_coords if p not in train_set]
    filtered_negative_coords = [p for p in filtered_negative_coords if p not in train_set]
    start_time = print_time_cost(start_time, "删除训练集样本")
    
    print(f"删除训练集样本后剩余 {len(filtered_positive_coords)} 个正样本点")
    print(f"删除训练集样本后剩余 {len(filtered_negative_coords)} 个负样本点")
    
    train_coords = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)
    
    # 从剩余样本中抽取验证集
    print("\n选择验证集样本...")
    start_time = time.time()
    # 正样本使用KD树采样
    val_pos = spatial_balanced_sampling(filtered_positive_coords, n_samples, min_distance=min_distance, data=data)
    if len(val_pos) < n_samples:
        print(f"警告：无法选择足够的验证集正样本，只选择了 {len(val_pos)} 个")
        return
    start_time = print_time_cost(start_time, "选择验证集正样本")
    
    # 负样本使用随机采样
    start_time = time.time()
    val_neg = random_sampling(filtered_negative_coords, n_samples, min_distance=min_distance, data=data)
    if len(val_neg) < n_samples:
        print(f"警告：无法选择足够的验证集负样本，只选择了 {len(val_neg)} 个")
        return
    start_time = print_time_cost(start_time, "选择验证集负样本")
    
    # 从剩余样本集中删除验证集使用的样本
    print("删除验证集使用的样本...")
    start_time = time.time()
    val_set = set(val_pos + val_neg)
    filtered_positive_coords = [p for p in filtered_positive_coords if p not in val_set]
    filtered_negative_coords = [p for p in filtered_negative_coords if p not in val_set]
    start_time = print_time_cost(start_time, "删除验证集样本")
    
    print(f"删除验证集样本后剩余 {len(filtered_positive_coords)} 个正样本点")
    print(f"删除验证集样本后剩余 {len(filtered_negative_coords)} 个负样本点")
    
    val_coords = val_pos + val_neg
    val_labels = [1] * len(val_pos) + [0] * len(val_neg)
    
    # 从剩余样本中抽取测试集
    print("\n选择测试集样本...")
    start_time = time.time()
    # 使用所有剩余的正样本作为测试集的正样本
    test_pos = filtered_positive_coords
    print(f"测试集正样本数量: {len(test_pos)}")
    
    # 从剩余负样本中选择与正样本相同数量的样本
    if len(filtered_negative_coords) >= len(test_pos):
        # 负样本使用随机采样
        test_neg = random_sampling(filtered_negative_coords, len(test_pos), min_distance=min_distance, data=data)
    else:
        print("警告：剩余负样本数量不足以匹配测试集正样本数量")
        test_neg = filtered_negative_coords
    start_time = print_time_cost(start_time, "选择测试集样本")
    
    test_coords = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)
    
    # 验证数据集之间没有重叠
    start_time = time.time()
    train_set = set(train_coords)
    val_set = set(val_coords)
    test_set = set(test_coords)
    
    print("\n验证数据集重叠情况:")
    print(f"训练集和验证集重叠: {len(train_set & val_set)}")
    print(f"训练集和测试集重叠: {len(train_set & test_set)}")
    print(f"验证集和测试集重叠: {len(val_set & test_set)}")
    start_time = print_time_cost(start_time, "验证数据集重叠")
    
    # 如果存在重叠，重新采样
    if len(train_set & val_set) > 0 or len(train_set & test_set) > 0 or len(val_set & test_set) > 0:
        print("错误：数据集之间存在重叠，需要重新采样")
        return
    
    # 打印每个数据集的样本分布
    print("\n数据集样本分布:")
    print(f"训练集 - 正样本: {len(train_pos)}, 负样本: {len(train_neg)}")
    print(f"验证集 - 正样本: {len(val_pos)}, 负样本: {len(val_neg)}")
    print(f"测试集 - 正样本: {len(test_pos)}, 负样本: {len(test_neg)}")
    
    # 保存样本点
    def save_samples(coords, labels, filename):
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write("row,col,label\n")
            for (row, col), label in zip(coords, labels):
                f.write(f"{row},{col},{label}\n")
    
    # 保存训练集
    start_time = time.time()
    save_samples(train_coords, train_labels, "train_samples.csv")
    print(f"训练集样本数: {len(train_coords)}")
    start_time = print_time_cost(start_time, "保存训练集")
    
    # 保存验证集
    start_time = time.time()
    save_samples(val_coords, val_labels, "val_samples.csv")
    print(f"验证集样本数: {len(val_coords)}")
    start_time = print_time_cost(start_time, "保存验证集")
    
    # 保存测试集
    start_time = time.time()
    save_samples(test_coords, test_labels, "test_samples.csv")
    print(f"测试集样本数: {len(test_coords)}")
    start_time = print_time_cost(start_time, "保存测试集")
    
    # 保存样本分布图
    def create_sample_map(coords, labels, filename):
        sample_map = np.zeros_like(data)
        for (row, col), label in zip(coords, labels):
            if label == 1:  # 正样本保持原始值
                sample_map[row, col] = data[row, col]
            else:  # 负样本统一设为99
                sample_map[row, col] = 99
        
        driver = gdal.GetDriverByName('GTiff')
        out_ds = driver.Create(
            os.path.join(output_dir, filename),
            data.shape[1],
            data.shape[0],
            1,
            gdal.GDT_Byte,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        
        # 复制地理信息
        out_ds.SetGeoTransform(geo_transform)
        out_ds.SetProjection(projection)
        
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(sample_map)
        out_band.SetNoDataValue(0)
        
        out_ds = None
    
    # 创建并保存样本分布图
    start_time = time.time()
    create_sample_map(train_coords, train_labels, "train_samples.tif")
    create_sample_map(val_coords, val_labels, "val_samples.tif")
    create_sample_map(test_coords, test_labels, "test_samples.tif")
    start_time = print_time_cost(start_time, "保存样本分布图")
    
    total_time = print_time_cost(total_start_time, "总耗时")
    print("\n完成! 所有样本已保存到:", output_dir)

if __name__ == "__main__":
    # 设置输入输出路径
    input_tif = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/FIRMS_2024_sum_cleaned.tif"
    output_dir = "/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/FIRMS/with issues/hackathon_data/samples_2024"
    
    # 执行样本抽取
    extract_samples(input_tif, output_dir, n_samples=200, min_distance=5)