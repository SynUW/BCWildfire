# -*- coding: utf-8 -*-
"""
GEE 多账号直下 — 优先重试 + 逐轮加倍切分 + 稳健完成判定 + 结尾复核
- 不创建/导入 Asset；不 clip/reproject
- 固定 R×C 网格；单波段直下；必要时 N×N 切分 + 本地 merge；多波段最终堆栈
- 未下齐 → 高优先级重试队列 retry_q（优先于新日期）；按尝试轮次加倍切分密度
- 完成判定：文件“TIFF 头 或 ≥256B”即可（可切换为严格魔数）
- 结束后快速复核实际完成天数，避免进度条统计偏差
"""

import os, ee, math, time, json, logging, tempfile, requests, warnings, geopandas as gpd, pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.geometry.base import BaseGeometry
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context, Manager
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import threading
import shutil
import glob
import atexit

# ---- 静默掉 area 的 CRS 警告（我们仅用于“筛碎片”）----
warnings.filterwarnings("ignore", message="There is no STAC entry for")
warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS. Results from 'area' are likely incorrect.")

# ================== 你的配置 ==================
# save_dir = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/gee_patches'
save_dir = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/bc_raw_data_new_fast/drivers'
os.makedirs(save_dir, exist_ok=True)
# shapefile_path = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/shpfiles/bc_boundary/bc_boundary_without_sea.shp'
shapefile_path = r'/mnt/raid/zhengsen/wildfire_dataset/self_built_materials/shpfiles/Canada_boundary/Canada_Export_grids_1_2.shp'
SERVICE_ACCOUNT_KEYS = [
    "./api_key/b250_down3.json",
    './api_key/b250_down4.json',
    './api_key/b250_down5.json',
    './api_key/b250_down6.json',
    './api_key/b250_down7.json',   
   './api_key/b250down1-ef11f747afec.json',
    './api_key/wildfiredown2-30c54f4b9109.json',
    "./api_key/wildfire20-4a19244f8185.json",
    "./api_key/wildfiredown2-8362a8c7e344.json",
    "./api_key/b250down1-ef11f747afec.json",
]

# ===== 并发与阈值（更稳）=====
TILE_WORKERS   = 6   # 每个进程：瓦片并发（限低更抗限流）
BAND_WORKERS   = 3   # 每个瓦片：波段并发
HTTP_TIMEOUT   = 20  #  HTTP请求超时，20秒
RETRY_TIMES    = 4
MAX_REQUEST_MB = 40.0  # 单请求安全阈值（减小以提高成功率）
DOWNLOAD_WORKERS = 3  # 多线程下载并发数

# ===== 日期范围 & 断点 =====
DATE_START = "2000-01-01"
DATE_END   = "2025-12-31"
RESUME_DATE = None           # 如 "2012-07-01"
MAX_DATE_RETRIES = 6         # 每天最多尝试次数（含首次）

# ===== 网格参数 =====
NUM_TILES = 4
TILE_ROWS = None
TILE_COLS = None
MIN_TILE_AREA_DEG2 = 0.02

# ===== 稳定性 =====
MIN_SUBTILE_AREA_DEG2  = 1e-6
SIMPLIFY_TOL = 0.001
USE_BBOX_FALLBACK = True

# ===== 完成判定（稳健&快）=====
STRICT_TIFF_MAGIC = False     # True → 严格读魔数；False → “TIFF 头 或 ≥MIN_VALID_BYTES”
MIN_VALID_BYTES   = 256       # 允许全掩膜小文件被判定为有效

# ===== 日志 =====
logging.basicConfig(level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - [%(processName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('GEE_Direct_MultiSA')

# ===== 临时文件清理管理器 =====
class TempFileManager:
    def __init__(self, cleanup_interval=3600):  # 默认1小时清理一次
        self.temp_dirs = set()
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        self.lock = threading.Lock()
        
        # 设置临时文件目录到有更多空间的位置
        self.temp_base_dir = self._get_best_temp_dir()
        
        # 注册退出时清理
        atexit.register(self.cleanup_all)
    
    def _get_best_temp_dir(self):
        """选择最佳临时目录（优先选择有更多空间的目录）"""
        candidates = [
            '/tmp',  # 系统临时目录
            '/var/tmp',  # 系统临时目录
            '/mnt/raid/tmp',  # 如果有RAID存储
            '/home/zhengsen/tmp',  # 用户目录下的tmp
            tempfile.gettempdir()  # 默认临时目录
        ]
        
        best_dir = tempfile.gettempdir()
        max_free_space = 0
        
        for candidate in candidates:
            try:
                if os.path.exists(candidate) and os.access(candidate, os.W_OK):
                    # 检查可用空间
                    statvfs = os.statvfs(candidate)
                    free_space = statvfs.f_frsize * statvfs.f_bavail
                    if free_space > max_free_space:
                        max_free_space = free_space
                        best_dir = candidate
            except (OSError, IOError):
                continue
        
        logger.info(f"选择临时目录: {best_dir} (可用空间: {max_free_space / (1024**3):.2f} GB)")
        return best_dir
    
    def get_temp_dir(self, prefix="tmp"):
        """获取临时目录，使用最佳位置"""
        # 检查磁盘空间
        self._check_disk_space()
        return tempfile.mkdtemp(prefix=prefix, dir=self.temp_base_dir)
    
    def _check_disk_space(self):
        """检查磁盘空间，如果不足则清理"""
        try:
            statvfs = os.statvfs(self.temp_base_dir)
            free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            
            if free_space_gb < 5.0:  # 如果可用空间小于5GB
                logger.warning(f"磁盘空间不足: {free_space_gb:.2f} GB，开始紧急清理...")
                self._emergency_cleanup()
                
                # 再次检查
                statvfs = os.statvfs(self.temp_base_dir)
                free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
                logger.info(f"紧急清理后可用空间: {free_space_gb:.2f} GB")
                
        except (OSError, IOError) as e:
            logger.warning(f"检查磁盘空间失败: {e}")
    
    def _emergency_cleanup(self):
        """紧急清理：清理所有可清理的临时文件"""
        with self.lock:
            # 清理所有注册的临时目录（除了正在使用的）
            for temp_dir in list(self.temp_dirs):
                if not self._is_dir_in_use(temp_dir):
                    try:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir, ignore_errors=True)
                            self.temp_dirs.discard(temp_dir)
                            logger.info(f"紧急清理目录: {temp_dir}")
                    except Exception as e:
                        logger.warning(f"紧急清理失败 {temp_dir}: {e}")
            
            # 清理所有匹配模式的临时文件
            patterns = [
                os.path.join(self.temp_base_dir, "download_*"),
                os.path.join(self.temp_base_dir, "band_*"),
                os.path.join(self.temp_base_dir, "tmp*")
            ]
            
            for pattern in patterns:
                try:
                    for temp_path in glob.glob(pattern):
                        if os.path.isdir(temp_path):
                            try:
                                shutil.rmtree(temp_path, ignore_errors=True)
                                logger.info(f"紧急清理目录: {temp_path}")
                            except Exception:
                                pass
                        elif os.path.isfile(temp_path):
                            try:
                                os.remove(temp_path)
                                logger.info(f"紧急清理文件: {temp_path}")
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"紧急清理模式 {pattern} 失败: {e}")
    
    def register_temp_dir(self, temp_dir):
        """注册临时目录"""
        with self.lock:
            self.temp_dirs.add(temp_dir)
    
    def unregister_temp_dir(self, temp_dir):
        """注销临时目录"""
        with self.lock:
            self.temp_dirs.discard(temp_dir)
    
    def cleanup_old_temp_files(self):
        """清理超过1小时的临时文件（只清理未注册的临时文件）"""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        with self.lock:
            self.last_cleanup = current_time
            # 获取当前注册的临时目录集合
            registered_dirs = set(self.temp_dirs)
            
        # 清理系统临时目录中的旧文件（排除正在使用的）
        temp_base = self.temp_base_dir
        patterns = [
            os.path.join(temp_base, "download_*"),
            os.path.join(temp_base, "band_*"),
            os.path.join(temp_base, "tmp*")
        ]
        
        cleaned_count = 0
        for pattern in patterns:
            try:
                for temp_path in glob.glob(pattern):
                    # 跳过正在使用的临时目录
                    if temp_path in registered_dirs:
                        continue
                        
                    if os.path.isdir(temp_path):
                        # 检查目录创建时间
                        try:
                            dir_time = os.path.getctime(temp_path)
                            if current_time - dir_time > self.cleanup_interval:
                                shutil.rmtree(temp_path, ignore_errors=True)
                                cleaned_count += 1
                        except (OSError, IOError):
                            pass
                    elif os.path.isfile(temp_path):
                        # 检查文件修改时间
                        try:
                            file_time = os.path.getmtime(temp_path)
                            if current_time - file_time > self.cleanup_interval:
                                os.remove(temp_path)
                                cleaned_count += 1
                        except (OSError, IOError):
                            pass
            except Exception as e:
                logger.warning(f"清理临时文件时出错: {e}")
        
        if cleaned_count > 0:
            logger.info(f"清理了 {cleaned_count} 个过期临时文件/目录（已排除正在使用的目录）")
    
    def cleanup_all(self):
        """清理所有注册的临时目录"""
        with self.lock:
            for temp_dir in list(self.temp_dirs):
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        logger.info(f"清理临时目录: {temp_dir}")
                except Exception as e:
                    logger.warning(f"清理临时目录失败 {temp_dir}: {e}")
            self.temp_dirs.clear()
    
    def safe_cleanup_dir(self, temp_dir):
        """安全清理单个临时目录"""
        try:
            if os.path.exists(temp_dir):
                # 检查目录是否还在使用中（通过检查是否有进程在使用文件）
                if self._is_dir_in_use(temp_dir):
                    logger.warning(f"临时目录 {temp_dir} 正在使用中，跳过清理")
                    return False
                
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.unregister_temp_dir(temp_dir)
                return True
        except Exception as e:
            logger.warning(f"清理临时目录失败 {temp_dir}: {e}")
        return False
    
    def _is_dir_in_use(self, temp_dir):
        """检查目录是否正在被使用"""
        try:
            if not os.path.exists(temp_dir):
                return False
                
            # 检查目录中是否有最近修改的文件（2分钟内）
            current_time = time.time()
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # 如果文件在2分钟内被修改，认为正在使用
                        if current_time - os.path.getmtime(file_path) < 120:  # 2分钟
                            return True
                        # 如果文件正在被写入（文件大小在变化），认为正在使用
                        if os.path.isfile(file_path):
                            stat = os.stat(file_path)
                            if current_time - stat.st_atime < 60:  # 1分钟内被访问
                                return True
                    except (OSError, IOError):
                        continue
            return False
        except Exception:
            # 如果检查失败，保守起见认为正在使用
            return True

# 全局临时文件管理器（更频繁清理：每10分钟）
temp_manager = TempFileManager(cleanup_interval=600)

# ===== 性能监控 =====
class PerformanceMonitor:
    def __init__(self):
        self.download_times = []
        self.error_counts = {}
        self.success_rates = {}
        self.total_downloads = 0
        self.successful_downloads = 0
        self.start_time = time.time()
    
    def log_download_time(self, duration):
        self.download_times.append(duration)
    
    def log_error(self, error_type):
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def log_success(self):
        self.successful_downloads += 1
        self.total_downloads += 1
    
    def log_failure(self):
        self.total_downloads += 1
    
    def get_stats(self):
        if not self.download_times:
            return "无下载数据"
        
        avg_time = sum(self.download_times) / len(self.download_times)
        success_rate = (self.successful_downloads / max(1, self.total_downloads)) * 100
        elapsed = time.time() - self.start_time
        
        return f"平均下载时间: {avg_time:.2f}s, 成功率: {success_rate:.1f}%, 总耗时: {elapsed:.1f}s"

# 全局性能监控器
perf_monitor = PerformanceMonitor()

# ================== 数据源（含起始日） ==================
data_sources = {
    'MOD09A1': {
        'collection': 'MODIS/061/MOD09A1',
        'bands': ['sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b07','QA', 'StateQA'],
        'native_res': 500,
        'start': '2000-02-18',
    },
    'MYD09A1': {
        'collection': 'MODIS/061/MYD09A1',
        'bands': ['sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b07','QA', 'StateQA'],
        'native_res': 500,
        'start': '2002-07-04',
    },
    'MOD11A2': {
        'collection': 'MODIS/061/MOD11A2',
        'bands': ['LST_Day_1km', 'LST_Night_1km', 'Emis_31', 'Emis_32', 'QC_Day', 'QC_Night', 'Clear_sky_days'],
        'native_res': 1000,
        'start': '2000-02-18',
    },
    'MYD11A2': {
        'collection': 'MODIS/061/MYD11A2',
        'bands': ['LST_Day_1km', 'LST_Night_1km', 'Emis_31', 'Emis_32', 'QC_Day', 'QC_Night', 'Clear_sky_days'],
        'native_res': 1000,
        'start': '2002-07-04',
    },
    
    'MOD13Q1': {
        'collection': 'MODIS/061/MOD13Q1',
        'bands': ['NDVI', 'EVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'DetailedQA', 'SummaryQA'],
        'native_res': 250,
        'start': '2000-01-01',
    },
    'MYD13Q1': {
        'collection': 'MODIS/061/MYD13Q1',
        'bands': ['NDVI', 'EVI', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b03', 'sur_refl_b07', 'DetailedQA', 'SummaryQA'],
        'native_res': 250,
        'start': '2002-01-01',
    },
    # 'MOD16A2': {
    #     'collection': 'MODIS/061/MOD16A2',
    #     'bands': ['ET', 'LT', 'PET', 'PLE', 'ET_QC'],
    #     'native_res': 500,
    #     'start': '2021-01-01',
    # },
    # 'MYD16A2': {
    #     'collection': 'MODIS/061/MYD16A2',
    #     'bands': ['ET', 'LT', 'PET', 'PLE', 'ET_QC'],
    #     'native_res': 500,
    #     'start': '2021-01-01',
    # },
    'MCD12Q1': {
        'collection': 'MODIS/061/MCD12Q1',
        'bands': ['LC_Type1', 'QC'],
        'native_res': 500,
        'start': '2000-01-01',
    },
    'MYD09GA_b1237': {
        'collection': 'MODIS/061/MYD09GA',
        'bands': ['sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b07','QC_500m'],
        'native_res': 500,
        'start': '2002-07-04',
    },
    'MOD09GA_b1237': {
        'collection': 'MODIS/061/MOD09GA',
        'bands': ['sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b07','QC_500m'],
        'native_res': 500,
        'start': '2000-02-24',
    },
    'MOD09GA_QA_state1km': {
        'collection': 'MODIS/061/MOD09GA',
        'bands': ['state_1km'],
        'native_res': 1000,
        'start': '2000-02-24',
    },
    'MYD09GA_QA_state1km': {
        'collection': 'MODIS/061/MYD09GA',
        'bands': ['state_1km'],
        'native_res': 1000,
        'start': '2002-07-04',
    },
    'MOD14A1': {
        'collection': 'MODIS/061/MOD14A1',
        'bands': ['FireMask', 'MaxFRP', 'QA'],
        'native_res': 1000,
        'start': '2000-02-24',
    },
    'MYD14A1': {
        'collection': 'MODIS/061/MYD14A1',
        'bands': ['FireMask', 'MaxFRP', 'QA'],
        'native_res': 1000,
        'start': '2002-07-04',
    },
    'MCD15A3H':{
        'collection': 'MODIS/061/MCD15A3H',
        'bands': ['Lai', 'Fpar', 'FparLai_QC', 'FparExtra_QC'],
        'native_res': 500,
        'start': '2002-07-04',
    },
    'MOD11A1': {
        'collection': 'MODIS/061/MOD11A1',
        'bands': ['LST_Day_1km', 'LST_Night_1km', 'Emis_31', 'Emis_32', 'QC_Day', 'QC_Night'],
        'native_res': 1000,
        'start': '2000-03-05',
    },
    'MYD11A1': {
        'collection': 'MODIS/061/MYD11A1',
        'bands': ['LST_Day_1km', 'LST_Night_1km', 'Emis_31', 'Emis_32', 'QC_Day', 'QC_Night'],
        'native_res': 1000,
        'start': '2002-07-04',
    },
    'MOD09CMG':{
        'collection': 'MODIS/061/MOD09CMG',
        'bands': [
            'Coarse_Resolution_Brightness_Temperature_Band_20','Coarse_Resolution_Brightness_Temperature_Band_21',
            'Coarse_Resolution_Brightness_Temperature_Band_31','Coarse_Resolution_Brightness_Temperature_Band_32',
            'Coarse_Resolution_QA','Coarse_Resolution_State_QA','Coarse_Resolution_Internal_CM'
        ],
        'native_res': 5600,
        'start': '2000-02-24',
    },
    'MYD09CMG':{
        'collection': 'MODIS/061/MYD09CMG',
        'bands': [
            'Coarse_Resolution_Brightness_Temperature_Band_20','Coarse_Resolution_Brightness_Temperature_Band_21',
            'Coarse_Resolution_Brightness_Temperature_Band_31','Coarse_Resolution_Brightness_Temperature_Band_32',
            'Coarse_Resolution_QA','Coarse_Resolution_State_QA','Coarse_Resolution_Internal_CM'
        ],
        'native_res': 5600,
        'start': '2002-07-04',
    },
    'ERA5':{
        'collection': 'ECMWF/ERA5_LAND/DAILY_AGGR',
        'bands': ['temperature_2m','u_component_of_wind_10m','v_component_of_wind_10m','snow_cover',
                  'total_precipitation_sum','surface_latent_heat_flux_sum','dewpoint_temperature_2m',
                  'surface_pressure','volumetric_soil_water_layer_1','volumetric_soil_water_layer_2',
                  'volumetric_soil_water_layer_3','volumetric_soil_water_layer_4', 'temperature_2m_max',
                  'skin_temperature_max','potential_evaporation_sum','total_evaporation_sum',
                  'skin_reservoir_content', 'surface_net_solar_radiation_sum'],
        'native_res': 11132,
        'start': '1981-01-01',
    },
    
}

# ================== 几何与网格 ==================
def load_bc_region(shp_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs('EPSG:4326') if gdf.crs else gdf.set_crs('EPSG:4326')
    geom = unary_union(gdf.geometry)
    return gpd.GeoDataFrame(geometry=[geom], crs='EPSG:4326')

def choose_layout(num_tiles: int):
    if TILE_ROWS and TILE_COLS: return TILE_ROWS, TILE_COLS
    best = None
    for r in range(1, num_tiles + 1):
        c = (num_tiles + r - 1) // r
        score = (abs(r*c - num_tiles), abs(r - c))
        if best is None or score < best[0]: best = (score, r, c)
    _, rows, cols = best
    return rows, cols

def make_fixed_grid(bounds, rows, cols):
    minx, miny, maxx, maxy = bounds
    xs = [minx + (maxx - minx) * i / cols for i in range(cols + 1)]
    ys = [miny + (maxy - miny) * j / rows for j in range(rows + 1)]
    cells = [box(xs[i], ys[j], xs[i+1], ys[j+1]) for i in range(cols) for j in range(rows)]
    return gpd.GeoDataFrame(geometry=cells, crs='EPSG:4326')

def intersect_tiles(region_gdf: gpd.GeoDataFrame, rows: int, cols: int):
    bounds = region_gdf.total_bounds
    grid = make_fixed_grid(bounds, rows, cols)
    grid = grid[grid.intersects(region_gdf.geometry.iloc[0])]
    tiles = gpd.overlay(grid, region_gdf, how='intersection')
    tiles = tiles[tiles.area >= MIN_TILE_AREA_DEG2]   # 仅筛碎片（度²）
    tiles.reset_index(drop=True, inplace=True)
    return tiles

def simplify_or_bbox(poly: BaseGeometry, use_bbox=False) -> BaseGeometry:
    if use_bbox: return box(*poly.bounds)
    simp = poly.simplify(SIMPLIFY_TOL, preserve_topology=True)
    return simp if not simp.is_empty else box(*poly.bounds)

# ================== GEE & HTTP ==================
def initialize_gee_with_key(json_key_path: str):
    with open(json_key_path, "r") as f:
        sa = json.load(f)
    email = sa.get("client_email")
    creds = ee.ServiceAccountCredentials(email, json_key_path)
    ee.Initialize(creds)

_session = None
_session_lock = threading.Lock()

def create_optimized_session():
    """创建优化的HTTP会话，支持重试和连接池"""
    session = requests.Session()
    
    # 配置重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    # 配置适配器
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20,
        pool_block=False
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 设置请求头
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'image/tiff,image/*,*/*',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
    })
    
    return session

def http_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = create_optimized_session()
    return _session

def file_ok(fp: str) -> bool:
    """稳健且快速：存在 且 (TIFF魔数 或 ≥MIN_VALID_BYTES)。"""
    if not os.path.exists(fp): return False
    try:
        if STRICT_TIFF_MAGIC:
            with open(fp, "rb") as f:
                head = f.read(4)
            return head in (b"II*\x00", b"MM\x00*")
        # 宽松：魔数 或 ≥阈值
        with open(fp, "rb") as f:
            head = f.read(4)
        if head in (b"II*\x00", b"MM\x00*"): return True
        return os.path.getsize(fp) >= MIN_VALID_BYTES
    except Exception:
        return False

def download_chunk(session, url, start_byte, end_byte, chunk_id, tmp_dir):
    """下载文件的一个分块"""
    headers = {'Range': f'bytes={start_byte}-{end_byte}'}
    chunk_file = os.path.join(tmp_dir, f"chunk_{chunk_id}.part")
    
    try:
        with session.get(url, headers=headers, stream=True, timeout=HTTP_TIMEOUT) as r:
            r.raise_for_status()
            with open(chunk_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return chunk_file
    except Exception as e:
        logger.error(f"下载分块 {chunk_id} 失败: {e}")
        return None

def merge_chunks(chunk_files, output_path):
    """合并下载的分块"""
    try:
        with open(output_path, 'wb') as outfile:
            for chunk_file in sorted(chunk_files):
                if os.path.exists(chunk_file):
                    with open(chunk_file, 'rb') as infile:
                        outfile.write(infile.read())
                    os.remove(chunk_file)  # 清理临时文件
        return True
    except Exception as e:
        logger.error(f"合并分块失败: {e}")
        return False

def get_file_size(session, url):
    """获取文件大小"""
    try:
        response = session.head(url, timeout=HTTP_TIMEOUT)
        return int(response.headers.get('content-length', 0))
    except Exception:
        return 0

def http_download_tiff_multithreaded(url: str, out_path: str, timeout: int = HTTP_TIMEOUT):
    """多线程下载TIFF文件"""
    if file_ok(out_path): 
        return True
    
    session = http_session()
    tmp_dir = temp_manager.get_temp_dir(prefix="download_")
    temp_manager.register_temp_dir(tmp_dir)  # 注册临时目录
    
    try:
        # 定期清理旧临时文件
        temp_manager.cleanup_old_temp_files()
        
        # 获取文件大小
        file_size = get_file_size(session, url)
        if file_size == 0:
            # 如果无法获取大小，回退到单线程下载
            return http_download_tiff_single(url, out_path, timeout)
        
        # 计算分块大小和数量
        chunk_size = max(1024 * 1024, file_size // DOWNLOAD_WORKERS)  # 至少1MB
        chunks = []
        for i in range(0, file_size, chunk_size):
            start_byte = i
            end_byte = min(i + chunk_size - 1, file_size - 1)
            chunks.append((start_byte, end_byte, len(chunks)))
        
        # 多线程下载分块
        chunk_files = []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
            futures = []
            for start_byte, end_byte, chunk_id in chunks:
                future = executor.submit(download_chunk, session, url, start_byte, end_byte, chunk_id, tmp_dir)
                futures.append(future)
            
            for future in as_completed(futures):
                chunk_file = future.result()
                if chunk_file:
                    chunk_files.append(chunk_file)
        
        # 合并分块
        if len(chunk_files) == len(chunks):
            if merge_chunks(chunk_files, out_path):
                return file_ok(out_path)
        
        # 如果多线程下载失败，回退到单线程
        return http_download_tiff_single(url, out_path, timeout)
        
    except Exception as e:
        logger.error(f"多线程下载失败: {e}")
        return http_download_tiff_single(url, out_path, timeout)
    finally:
        # 使用管理器安全清理临时目录
        temp_manager.safe_cleanup_dir(tmp_dir)

def http_download_tiff_single(url: str, out_path: str, timeout: int = HTTP_TIMEOUT):
    """单线程下载（作为多线程下载的备用方案）"""
    if file_ok(out_path): 
        return True
    
    s = http_session()
    tmp = out_path + ".part"
    if os.path.exists(tmp):
        try: 
            os.remove(tmp)
        except Exception: 
            pass
    
    try:
        for i in range(RETRY_TIMES):
            try:
                with s.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk: 
                                f.write(chunk)
                os.replace(tmp, out_path)
                return file_ok(out_path)
            except Exception:
                time.sleep(1.0 + 1.5 * i)
                if os.path.exists(tmp):
                    try: 
                        os.remove(tmp)
                    except Exception: 
                        pass
        return False
    finally:
        # 确保临时文件被清理
        if os.path.exists(tmp):
            try: 
                os.remove(tmp)
            except Exception: 
                pass

def http_download_tiff(url: str, out_path: str, timeout: int = HTTP_TIMEOUT):
    """智能选择下载方式，包含性能监控"""
    start_time = time.time()
    
    try:
        session = http_session()
        file_size = get_file_size(session, url)
        
        if file_size > 0 and file_size < 5 * 1024 * 1024:  # 小于5MB使用单线程
            success = http_download_tiff_single(url, out_path, timeout)
        else:
            success = http_download_tiff_multithreaded(url, out_path, timeout)
        
        # 记录性能数据
        duration = time.time() - start_time
        perf_monitor.log_download_time(duration)
        
        if success:
            perf_monitor.log_success()
        else:
            perf_monitor.log_failure()
            perf_monitor.log_error("download_failed")
        
        # 下载完成后立即进行清理
        temp_manager.cleanup_old_temp_files()
        
        return success
        
    except Exception as e:
        perf_monitor.log_failure()
        perf_monitor.log_error(str(type(e).__name__))
        return http_download_tiff_single(url, out_path, timeout)

# ================== 体量估算 & 切分 ==================
def estimate_request_mb(bbox_tuple, scale_m, bands, dtype_bytes):
    minx, miny, maxx, maxy = bbox_tuple
    width_deg, height_deg = max(0.0, maxx-minx), max(0.0, maxy-miny)
    if width_deg<=0 or height_deg<=0 or not scale_m or scale_m<=0: return 0.0
    lat_mid = (miny+maxy)/2.0
    width_m  = width_deg  * 111000.0 * max(0.1, math.cos(math.radians(lat_mid)))
    height_m = height_deg * 111000.0
    nx, ny = max(1, int(round(width_m/float(scale_m)))), max(1, int(round(height_m/float(scale_m))))
    px = nx*ny
    return px * max(1,bands) * max(1,dtype_bytes) / (1024.0*1024.0)

def decide_grid_splits(bbox_tuple, scale_m, dtype_bytes, limit_mb=MAX_REQUEST_MB, inflate=1):
    """按体量估算基础 N，再按重试轮次 inflate 倍增（1,2,3,...）"""
    mb = estimate_request_mb(bbox_tuple, scale_m, bands=1, dtype_bytes=dtype_bytes)
    base = 1 if mb <= limit_mb else int(math.ceil(math.sqrt(mb/limit_mb * 1.05)))
    return max(1, base * max(1, inflate))

def split_bbox_grid(bbox_tuple, n):
    minx, miny, maxx, maxy = bbox_tuple
    dx, dy = (maxx-minx)/float(n), (maxy-miny)/float(n)
    return [(minx+j*dx, miny+i*dy, minx+(j+1)*dx, miny+(i+1)*dy) for i in range(n) for j in range(n)]

# ================== 查询影像（区域 mosaic） ==================
def query_image(config, date_str, region_geom: ee.Geometry):
    d0 = datetime.strptime(date_str, "%Y-%m-%d")
    start_date, end_date = date_str, (d0 + timedelta(days=1)).strftime("%Y-%m-%d")
    col = ee.ImageCollection(config['collection']).filterDate(start_date, end_date).filterBounds(region_geom)
    if 'bands' in config and config['bands']:
        col = col.select(config['bands'])
    if col.size().getInfo() == 0:
        return None
    return ee.Image(col.mosaic())

# ================== 单波段直下（按轮次加倍切分） ==================
# def get_band_url(img, band, region_polygon, scale=None):
#     region = ee.Geometry(region_polygon.__geo_interface__)
#     params = {"filePerBand": False, "format": "GEO_TIFF", "region": region}
#     if scale is not None: params["scale"] = scale
#     return img.select([band]).getDownloadURL(params)

def get_band_url(img, band, region_polygon, scale=None):
    region = ee.Geometry(region_polygon.__geo_interface__)
    params = {"filePerBand": False, "format": "GEO_TIFF", "region": region}

    params["crs"] = "EPSG:4326"
    if scale is not None:
        params["scale"] = scale  # 度单位的像元尺寸由 EE 计算；一般给原生米级 scale 也可，EE会按crs换算

    return img.select([band]).reproject(crs="EPSG:4326", scale=scale).getDownloadURL(params)

def download_band_onepiece_or_grid(img, band, poly, scale, dtype_bytes, inflate_factor):
    bbox_tuple = tuple(poly.bounds)
    tmp_root = temp_manager.get_temp_dir(prefix=f"band_{band}_")
    temp_manager.register_temp_dir(tmp_root)  # 注册临时目录
    
    try:
        n = decide_grid_splits(bbox_tuple, scale, dtype_bytes, limit_mb=MAX_REQUEST_MB, inflate=inflate_factor)
        if n == 1:
            try:
                simp = simplify_or_bbox(poly, use_bbox=False)
                url = get_band_url(img, band, simp, scale=scale)
                out = os.path.join(tmp_root, "part.tif")
                if http_download_tiff(url, out): return out
                if USE_BBOX_FALLBACK:
                    url2 = get_band_url(img, band, box(*poly.bounds), scale=scale)
                    out2 = os.path.join(tmp_root, "part_bbox.tif")
                    if http_download_tiff(url2, out2): return out2
            except Exception:
                pass
            return None
        else:
            pieces = []
            for sub_bbox in split_bbox_grid(bbox_tuple, n):
                inter = poly.intersection(box(*sub_bbox))
                if inter.is_empty or inter.area < MIN_SUBTILE_AREA_DEG2: continue
                try:
                    simp = simplify_or_bbox(inter, use_bbox=False)
                    url = get_band_url(img, band, simp, scale=scale)
                    out = os.path.join(tmp_root, f"p_{len(pieces)}.tif")
                    if http_download_tiff(url, out):
                        pieces.append(out); continue
                    if USE_BBOX_FALLBACK:
                        url2 = get_band_url(img, band, box(*simp.bounds), scale=scale)
                        out2 = os.path.join(tmp_root, f"p_{len(pieces)}_bbox.tif")
                        if http_download_tiff(url2, out2): pieces.append(out2)
                except Exception:
                    continue
            if not pieces: return None
            try:
                import rasterio
                from rasterio.merge import merge as rio_merge
                srcs = [rasterio.open(p) for p in pieces]
                mosaic, out_transform = rio_merge(srcs)
                profile = srcs[0].profile.copy()
                profile.update({"height": mosaic.shape[1],"width": mosaic.shape[2],"transform": out_transform,"tiled": False,"compress": "LZW","count": 1})
                merged_tif = os.path.join(tmp_root, "merged.tif")
                with rasterio.open(merged_tif, "w", **profile) as dst:
                    dst.write(mosaic[0:1,:,:])
                for s in srcs: s.close()
                return merged_tif
            except Exception:
                return None
    except Exception as e:
        logger.error(f"下载波段 {band} 失败: {e}")
        return None
    # 注意：这里不立即清理，因为返回的文件路径还在使用中
    # 清理将在调用方完成使用后进行

def stack_bands_to_multitiff(band_paths, out_path):
    import rasterio
    opens = [rasterio.open(p) for p in band_paths]
    ref = opens[0]
    profile = ref.profile.copy()
    profile.update({"count": len(opens), "compress": "LZW"})
    tmp = out_path + ".part"
    if os.path.exists(tmp):
        try: os.remove(tmp)
        except Exception: pass
    with rasterio.open(tmp, "w", **profile) as dst:
        for i, ds in enumerate(opens, 1):
            dst.write(ds.read(1, masked=False), i)
    for ds in opens: ds.close()
    os.replace(tmp, out_path)
    return True

# ================== 单日期：仅补缺，返回“本轮是否完成” ==================
def process_one_day(date_str, tiles_gdf, region_ee, attempt_idx):
    """attempt_idx 从 0 开始。切分密度 inflate = max(1, attempt_idx+1)"""
    inflate = max(1, attempt_idx + 1)

    active_products = [(vn, cfg) for vn, cfg in data_sources.items()
                       if date_str >= cfg.get('start', DATE_START)]

    # 快速：逐 tile 判断是否都已存在
    tile_count = len(tiles_gdf)
    all_done = True
    for var_name, cfg in active_products:
        var_dir = os.path.join(save_dir, var_name)
        for i in range(tile_count):
            fp = os.path.join(var_dir, f"{var_name}_{date_str.replace('-', '_')}_tile{i:02d}.tif")
            if not file_ok(fp):
                all_done = False
                break
        if not all_done:
            break
    if all_done:
        return True

    # 补缺
    day_done = True
    for var_name, cfg in active_products:
        var_dir = os.path.join(save_dir, var_name)
        os.makedirs(var_dir, exist_ok=True)
        prefix  = f"{var_name}_{date_str.replace('-', '_')}_tile"

        missing_jobs = []
        for i, poly in enumerate(tiles_gdf.geometry):
            fp = os.path.join(var_dir, f"{prefix}{i:02d}.tif")
            if not file_ok(fp):
                missing_jobs.append((i, poly))

        if not missing_jobs:
            continue

        # 仅缺失时才取影像（区域 mosaic）
        img = query_image(cfg, date_str, region_ee)
        if img is None:
            continue  # 当日确无影像

        scale = cfg.get('native_res')
        dtype_bytes = int(cfg.get('dtype_bytes', 2))
        bands = cfg['bands']

        def work_one_tile(idx, poly):
            fp_base = os.path.join(var_dir, f"{prefix}{idx:02d}.tif")
            if file_ok(fp_base):
                return True
            band_to_path = {}
            temp_dirs_to_cleanup = []  # 记录需要清理的临时目录
            
            def dl_one_band(band):
                return band, download_band_onepiece_or_grid(img, band, poly, scale, dtype_bytes, inflate_factor=inflate)
            
            ok = True
            with ThreadPoolExecutor(max_workers=BAND_WORKERS) as ex_band:
                futs = [ex_band.submit(dl_one_band, b) for b in bands]
                for f in as_completed(futs):
                    try:
                        b, p = f.result()
                        if p: 
                            band_to_path[b] = p
                            # 记录临时目录路径用于后续清理
                            temp_dir = os.path.dirname(p)
                            if temp_dir not in temp_dirs_to_cleanup:
                                temp_dirs_to_cleanup.append(temp_dir)
                        else: 
                            ok = False
                    except Exception:
                        ok = False
            
            if (not ok) or (len(band_to_path) != len(bands)):
                # 清理临时文件
                for temp_dir in temp_dirs_to_cleanup:
                    temp_manager.safe_cleanup_dir(temp_dir)
                return False
            
            try:
                ordered = [band_to_path[b] for b in bands]
                stack_bands_to_multitiff(ordered, fp_base)
                
                # 成功合并后立即清理临时文件
                for temp_dir in temp_dirs_to_cleanup:
                    temp_manager.safe_cleanup_dir(temp_dir)
                
                return True
            except Exception as e:
                logger.error(f"合并波段失败: {e}")
                # 异常时也要清理临时文件
                for temp_dir in temp_dirs_to_cleanup:
                    temp_manager.safe_cleanup_dir(temp_dir)
                return False

        tile_ok = True
        with ThreadPoolExecutor(max_workers=TILE_WORKERS) as ex_tile:
            futs = [ex_tile.submit(work_one_tile, i, poly) for i, poly in missing_jobs]
            for f in as_completed(futs):
                try:
                    if not f.result():
                        tile_ok = False
                except Exception:
                    tile_ok = False

        if not tile_ok:
            day_done = False

    # 每次处理完一个日期后，进行主动清理
    temp_manager.cleanup_old_temp_files()
    
    return day_done

# ================== Worker 进程（优先重试） ==================
def worker_proc(json_key_path, tiles_wkb, dates_q, retry_q, progress_q, region_geojson):
    # 初始化 GEE
    try:
        with open(json_key_path, "r") as f:
            sa = json.load(f)
        creds = ee.ServiceAccountCredentials(sa["client_email"], json_key_path)
        ee.Initialize(creds)
    except Exception as e:
        logger.error(f"GEE 初始化失败: {json_key_path} - {e}")
        return

    # 反序列化 tiles & region
    from shapely import wkb
    geoms = [wkb.loads(b) for b in tiles_wkb]
    tiles_gdf = gpd.GeoDataFrame(geometry=geoms, crs='EPSG:4326')
    region_ee = ee.Geometry(region_geojson)

    while True:
        # 先取重试队列；没有再取新日期
        item = None
        try:
            item = retry_q.get_nowait()
        except Exception:
            pass
        if item is None:
            try:
                item = dates_q.get_nowait()
            except Exception:
                break  # 两个队列都空 → 退出

        date_str, attempt = item
        try:
            complete = process_one_day(date_str, tiles_gdf, region_ee, attempt_idx=attempt)
        except Exception as e:
            logger.error(f"[{os.path.basename(json_key_path)}] 处理 {date_str} 异常: {e}")
            complete = False

        # 完成则 done；未完成但未达重试上限 → 放回 retry_q（高优先级），尝试次数+1
        if complete or attempt+1 >= MAX_DATE_RETRIES:
            try:
                progress_q.put({"done": 1, "success": int(bool(complete))})
            except Exception:
                pass
        else:
            try:
                retry_q.put((date_str, attempt+1))
            except Exception:
                pass

# ================== 收尾：快速复核真实完成天数 ==================
def recount_completed_days(dates, tiles_gdf):
    """以 file_ok() 快速复核：当日所有‘生效产品 × 所有瓦片’均存在即 +1"""
    ok = 0
    tile_count = len(tiles_gdf)
    for d in dates:
        good = True
        for var_name, cfg in data_sources.items():
            if d < cfg.get('start', DATE_START):
                continue
            var_dir = os.path.join(save_dir, var_name)
            for i in range(tile_count):
                fp = os.path.join(var_dir, f"{var_name}_{d.replace('-', '_')}_tile{i:02d}.tif")
                if not file_ok(fp):
                    good = False
                    break
            if not good: break
        if good: ok += 1
    return ok

# ================== 主入口 ==================
def main():
    # 启动时检查磁盘空间
    print("=== 磁盘空间检查 ===")
    temp_manager._check_disk_space()
    
    # 预计算网格 & 区域
    bc_gdf = load_bc_region(shapefile_path)
    rows, cols = choose_layout(NUM_TILES)
    tiles_gdf = intersect_tiles(bc_gdf, rows, cols)
    tiles_wkb = [g.wkb for g in tiles_gdf.geometry]
    region_geojson = unary_union(bc_gdf.geometry).__geo_interface__

    # 日期列表（支持断点起）
    dates_all = pd.date_range(start=DATE_START, end=DATE_END).strftime('%Y-%m-%d').tolist()
    dates = [d for d in dates_all if (RESUME_DATE is None or d >= RESUME_DATE)]
    total_steps = len(dates)
    if total_steps == 0:
        print("无可处理日期。"); return

    ctx = get_context("spawn")
    with Manager() as m:
        dates_q    = m.Queue()
        retry_q    = m.Queue()   # 高优先级重试队列
        progress_q = m.Queue()
        for d in dates:
            dates_q.put((d, 0))  # (日期, 已尝试次数)

        # 启动进程
        procs = []
        for key_path in SERVICE_ACCOUNT_KEYS:
            if not os.path.exists(key_path):
                print(f"[WARN] 找不到 JSON: {key_path}，跳过该账号"); continue
            p = ctx.Process(target=worker_proc,
                            args=(key_path, tiles_wkb, dates_q, retry_q, progress_q, region_geojson),
                            daemon=False)
            p.start(); procs.append(p)
        if not procs:
            print("未启动任何进程：请检查 SERVICE_ACCOUNT_KEYS。"); return

        # 单一进度条：完成的天数（成功或达到重试上限都计入 done）
        done_days, ok_days = 0, 0
        pbar = tqdm(total=total_steps, desc="Days completed", ncols=100)
        try:
            while done_days < total_steps:
                try:
                    msg = progress_q.get(timeout=1.0)
                    if msg.get("done"):
                        done_days += 1
                        ok_days += int(msg.get("success", 0))
                        pbar.update(1)
                        pbar.set_postfix_str(f"ok={ok_days}, retry<= {MAX_DATE_RETRIES}")
                except Exception:
                    if not any(p.is_alive() for p in procs):
                        break
        finally:
            pbar.close()
        for p in procs: p.join()

    # 结尾复核：以磁盘上的真实文件为准再数一遍
    actual_ok = recount_completed_days(dates, tiles_gdf)
    print(f"进度条统计：成功天数 = {ok_days} / {total_steps}；复核统计：成功天数 = {actual_ok} / {total_steps}")
    
    # 最终清理所有临时文件
    print("\n=== 清理临时文件 ===")
    temp_manager.cleanup_all()
    
    # 输出性能统计
    print(f"\n=== 性能统计 ===")
    print(f"下载性能: {perf_monitor.get_stats()}")
    if perf_monitor.error_counts:
        print(f"错误统计: {perf_monitor.error_counts}")
    print("=" * 50)

if __name__ == "__main__":
    main()
