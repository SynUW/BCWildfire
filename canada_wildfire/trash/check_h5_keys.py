import h5py
import os


def count_pos_neg_samples_in_h5(h5_file):
    """统计h5文件中正负样本数量（数据集名包含positive/negative）"""
    pos_count = 0
    neg_count = 0
    try:
        with h5py.File(h5_file, 'r') as f:
            for key in f.keys():
                if 'positive' in key:
                    pos_count += 1
                elif 'negative' in key:
                    neg_count += 1
    except Exception as e:
        print(f"处理文件 {h5_file} 时出错: {e}")
    return pos_count, neg_count


def check_all_h5_in_dir(directory):
    """遍历目录下所有h5文件，统计正负样本数量及比例"""
    h5_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    if not h5_files:
        print(f"在 {directory} 中未找到h5文件")
        return
    for h5_file in h5_files:
        path = os.path.join(directory, h5_file)
        pos, neg = count_pos_neg_samples_in_h5(path)
        total = pos + neg
        if total > 0:
            ratio = pos / total
            print(f"文件: {h5_file} | 正样本: {pos} | 负样本: {neg} | 正负样本比例: {ratio:.2f}")
        else:
            print(f"文件: {h5_file} | 正样本: {pos} | 负样本: {neg} | 无样本")


if __name__ == "__main__":
    work_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_merged_normalized/test/files'
    check_all_h5_in_dir(os.path.dirname(os.path.abspath(work_dir)))
