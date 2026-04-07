import os
import glob
import numpy as np
from osgeo import gdal
from datetime import datetime

class OptimizedWindowSampler:
    def __init__(self, window_size=10, stride=5, min_fire_pixels=1, negative_ratio=2):
        self.window_size = window_size
        self.stride = stride
        self.min_fire_pixels = min_fire_pixels
        self.negative_ratio = negative_ratio
    def _is_valid_window(self, window, shape):
        x1, y1, x2, y2 = window
        height, width = shape
        return (x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height and x2 > x1 and y2 > y1)
    def _calculate_overlap_ratio(self, window1, window2):
        x1, y1, x2, y2 = window1
        ex1, ey1, ex2, ey2 = window2
        overlap_x1 = max(x1, ex1)
        overlap_y1 = max(y1, ey1)
        overlap_x2 = min(x2, ex2)
        overlap_y2 = min(y2, ey2)
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return 0.0
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        window_area = (x2 - x1) * (y2 - y1)
        return overlap_area / window_area if window_area > 0 else 0.0
    def _sample_positive_windows_efficient(self, fire_data):
        fire_y, fire_x = np.where(fire_data == 1)
        if len(fire_x) == 0:
            return []
        fire_points = list(zip(fire_x, fire_y))
        selected_windows = []
        grid_size = self.window_size // 2
        grid = {}
        for x, y in fire_points:
            grid_key = (x // grid_size, y // grid_size)
            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append((x, y))
        for cell_points in grid.values():
            if not cell_points:
                continue
            center_x = sum(p[0] for p in cell_points) / len(cell_points)
            center_y = sum(p[1] for p in cell_points) / len(cell_points)
            best_point = min(cell_points, key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
            x, y = best_point
            window = (x - self.window_size//2, y - self.window_size//2, x + self.window_size//2, y + self.window_size//2)
            if self._is_valid_window(window, fire_data.shape):
                valid = True
                for existing_window in selected_windows:
                    if self._calculate_overlap_ratio(window, existing_window) > 0.5:
                        valid = False
                        break
                if valid:
                    selected_windows.append(window)
        return selected_windows
    def _sample_negative_windows_efficient(self, fire_data, positive_windows, target_count):
        height, width = fire_data.shape
        half_width = width // 2
        step = max(1, self.stride)
        y_coords = np.arange(self.window_size//2, height - self.window_size//2, step)
        x_coords = np.arange(half_width + self.window_size//2, width - self.window_size//2, step)
        if len(y_coords) == 0 or len(x_coords) == 0:
            return []
        Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
        candidate_points = list(zip(Y.flatten(), X.flatten()))
        valid_candidates = []
        for y, x in candidate_points:
            window = (x - self.window_size//2, y - self.window_size//2, x + self.window_size//2, y + self.window_size//2)
            if self._is_valid_window(window, fire_data.shape):
                x1, y1, x2, y2 = window
                window_data = fire_data[y1:y2, x1:x2]
                if not np.any(window_data == 1):
                    valid_candidates.append(window)
        final_windows = []
        for window in valid_candidates:
            valid = True
            for pos_window in positive_windows:
                if self._calculate_overlap_ratio(window, pos_window) > 0.1:
                    valid = False
                    break
            if valid:
                final_windows.append(window)
        if len(final_windows) < target_count:
            return final_windows
        np.random.seed(42)
        indices = np.random.choice(len(final_windows), target_count, replace=False)
        selected_windows = [final_windows[i] for i in indices]
        np.random.seed(None)
        return selected_windows
    def count_windows_for_file(self, fire_file):
        ds = gdal.Open(fire_file)
        if ds is None:
            return 0
        fire_data = ds.ReadAsArray()
        ds = None
        if fire_data is None:
            return 0
        if len(fire_data.shape) == 3:
            fire_data = fire_data[0]
        if not np.any(fire_data == 1):
            return 0
        pos_windows = self._sample_positive_windows_efficient(fire_data)
        neg_windows = self._sample_negative_windows_efficient(fire_data, pos_windows, len(pos_windows)*self.negative_ratio)
        return len(pos_windows) + len(neg_windows)

def main():
    firms_dir = '/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/all_data/Firms_Detection_resampled'
    all_files = glob.glob(os.path.join(firms_dir, '*.tif'))
    year_files = {}
    for file in all_files:
        basename = os.path.basename(file)
        parts = basename.split('_')
        if len(parts) >= 3:
            year = parts[-3]
            if year not in year_files:
                year_files[year] = []
            year_files[year].append(file)
    sorted_years = sorted([y for y in year_files.keys() if y.isdigit()], reverse=True)
    sorted_years = [y for y in sorted_years if 2000 <= int(y) <= 2024]
    sampler = OptimizedWindowSampler(window_size=10, stride=5, min_fire_pixels=1, negative_ratio=2)
    for year in sorted_years:
        total = 0
        for file in year_files[year]:
            total += sampler.count_windows_for_file(file)
        print(f"{year}年: 窗口数量: {total}")

if __name__ == '__main__':
    main() 