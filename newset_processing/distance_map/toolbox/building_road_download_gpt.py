import requests
import geopandas as gpd
import pandas as pd
import logging
from shapely.geometry import Polygon, LineString
import time
import os
import math

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 坐标系 (WGS84)
DEFAULT_CRS = "EPSG:4326"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# BC省分块下载配置
BC_BBOX = (48.2, -139.0, 60.0, -114.1)  # 整个BC省边界
CHUNK_SIZE = 2.0  # 每个分块的大小(度)


def split_bbox(bbox, chunk_size):
    """将大边界框分割为小方块"""
    south, west, north, east = bbox
    chunks = []

    lat_steps = math.ceil((north - south) / chunk_size)
    lon_steps = math.ceil((east - west) / chunk_size)

    for i in range(lat_steps):
        for j in range(lon_steps):
            chunk_south = south + i * chunk_size
            chunk_west = west + j * chunk_size
            chunk_north = min(chunk_south + chunk_size, north)
            chunk_east = min(chunk_west + chunk_size, east)

            # 确保最小有效区域
            if chunk_north - chunk_south > 0.1 and chunk_east - chunk_west > 0.1:
                chunks.append((chunk_south, chunk_west, chunk_north, chunk_east))

    return chunks


def build_query(bbox, query_type):
    """构建优化的查询语句"""
    south, west, north, east = bbox

    if query_type == "buildings":
        return f"""
        [out:json][timeout:1800];
        (
            way["building"]({south},{west},{north},{east});
            relation["building"]({south},{west},{north},{east});
        );
        (._;>;);
        out body geom;
        """
    else:  # roads
        return f"""
        [out:json][timeout:1800];
        (
            way["highway"]["highway"!~"footway|cycleway|path"]({south},{west},{north},{east});
        );
        (._;>;);
        out body geom;
        """


def download_chunk(bbox, query_type, max_retries=3):
    """下载单个分块数据"""
    for attempt in range(max_retries):
        try:
            query = build_query(bbox, query_type)
            response = requests.post(OVERPASS_URL, data={'data': query}, timeout=1800)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"分块下载失败 (尝试 {attempt + 1}): {str(e)}")
            time.sleep(30 * (attempt + 1))
    return None


def process_data(data, data_type):
    """处理原始OSM数据"""
    features = []
    for element in data.get('elements', []):
        if element['type'] == 'way' and 'geometry' in element:
            coords = [(n['lon'], n['lat']) for n in element['geometry']]
            if data_type == "buildings" and len(coords) >= 4:
                geom = Polygon(coords) if coords[0] == coords[-1] else LineString(coords)
            else:
                geom = LineString(coords)

            features.append({
                'type': 'Feature',
                'geometry': geom,
                'properties': element.get('tags', {})
            })

    if not features:
        return None

    gdf = gpd.GeoDataFrame.from_features(features, crs=DEFAULT_CRS)
    gdf['data_type'] = data_type
    return gdf


def download_region(region_name, bbox, data_type):
    """下载整个区域数据（自动分块）"""
    chunks = split_bbox(bbox, CHUNK_SIZE)
    all_gdfs = []

    for i, chunk in enumerate(chunks):
        logger.info(f"正在下载 {region_name} {data_type} 分块 {i + 1}/{len(chunks)}...")

        data = download_chunk(chunk, data_type)
        if data and data['elements']:
            gdf = process_data(data, data_type)
            if gdf is not None:
                all_gdfs.append(gdf)

        time.sleep(10)  # 分块间延迟

    if all_gdfs:
        return gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    return None


def main():
    # 确保输出目录存在
    os.makedirs("../osm_data", exist_ok=True)

    # 下载BC省数据（分块）
    for data_type in ["buildings", "roads"]:
        logger.info(f"\n=== 开始下载BC省 {data_type} ===")
        gdf = download_region("British_Columbia", BC_BBOX, data_type)

        if gdf is not None:
            output_file = f"osm_data/British_Columbia_{data_type}.geojson"
            gdf.to_file(output_file, driver='GeoJSON')
            logger.info(f"成功保存 {len(gdf)} 条记录到 {output_file}")
            logger.info(f"几何类型统计:\n{gdf.geometry.type.value_counts()}")
        else:
            logger.warning(f"未获取到BC省 {data_type} 数据")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序错误: {str(e)}", exc_info=True)