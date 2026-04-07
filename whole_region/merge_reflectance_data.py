def process_single_sensor(input_path, output_path):
    """处理单个传感器数据，转换为WGS84"""
    # 检查输出文件是否存在
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        print(f'Skipping single-sensor processing: {os.path.basename(output_path)} already exists')
        return

    # [Rest of the function remains the same]