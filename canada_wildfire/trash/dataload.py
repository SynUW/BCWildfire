import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MultiH5PastFutureDataset(Dataset):
    def __init__(self, h5_dir):
        self.h5_dir = h5_dir
        self.h5_files = sorted([os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith('.h5')])
        self.sample_index = []  # (h5_idx, past_key, future_key)
        self._build_index()

    def _build_index(self):
        for h5_idx, h5_path in tqdm(enumerate(self.h5_files), total=len(self.h5_files), desc='Building index'):
            with h5py.File(h5_path, 'r') as f:
                past_keys = [k for k in f.keys() if '_past_' in k]
                for past_key in past_keys:
                    future_key = past_key.replace('_past_', '_future_')
                    if future_key in f:
                        self.sample_index.append((h5_idx, past_key, future_key))

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        h5_idx, past_key, future_key = self.sample_index[idx]
        h5_path = self.h5_files[h5_idx]
        with h5py.File(h5_path, 'r') as f:
            past = torch.from_numpy(f[past_key][:]).float()
            future = torch.from_numpy(f[future_key][:]).float()
        return past, future

# 用法示例
if __name__ == '__main__':
    h5_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/samples_merged'
    dataset = MultiH5PastFutureDataset(h5_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    for past, future in dataloader:
        print('past:', past.shape, 'future:', future.shape)
        break