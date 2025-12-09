import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csvfile = "/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/Crime_Data_from_2020_to_Present_20250929.csv"

data = pd.read_csv(csvfile)

row_name = data.columns.tolist()
print(row_name)

# Print each column name and its data type
for col in row_name:
    print(f"{col}: {data[col].dtype}")

# 分析经纬度数据
lat_col = 'LAT'  # 纬度列名
lon_col = 'LON'  # 经度列名

# 检查这两列是否存在(根据实际列名可能需要调整)
# 常见的列名可能是 'LAT', 'Latitude', 'lat' 等
lat_lon_cols = [(col1, col2) for col1 in row_name for col2 in row_name 
                if 'lat' in col1.lower() and 'lon' in col2.lower()]

if lat_lon_cols:
    lat_col, lon_col = lat_lon_cols[0]
    
    print(f"\n=== 经纬度分析 ===")
    print(f"纬度列名: {lat_col}")
    print(f"经度列名: {lon_col}")
    
    # 总记录数
    total_records = len(data)
    print(f"\n总记录数: {total_records}")
    
    # 非空经纬度记录数
    valid_coords = data[[lat_col, lon_col]].dropna()
    
    # 去除 (0.0, 0.0) 的无效数据
    valid_coords = valid_coords[~((valid_coords[lat_col] == 0.0) & (valid_coords[lon_col] == 0.0))]
    
    valid_count = len(valid_coords)
    print(f"有效经纬度记录数: {valid_count}")
    print(f"缺失或无效经纬度记录数: {total_records - valid_count}")
    
    # 获取并打印最大最小值
    print(f"\n=== 经纬度范围 ===")
    print(f"纬度最小值: {valid_coords[lat_col].min():.4f}")
    print(f"纬度最大值: {valid_coords[lat_col].max():.4f}")
    print(f"经度最小值: {valid_coords[lon_col].min():.4f}")
    print(f"经度最大值: {valid_coords[lon_col].max():.4f}")
    
    # 唯一的经纬度组合
    unique_coords = valid_coords.drop_duplicates()
    unique_count = len(unique_coords)
    print(f"\n唯一的经纬度组合数: {unique_count}")
    
    # 重复的经纬度
    duplicate_count = valid_count - unique_count
    print(f"重复的经纬度记录数: {duplicate_count}")
    
    # 重复率
    if valid_count > 0:
        duplicate_rate = (duplicate_count / valid_count) * 100
        print(f"重复率: {duplicate_rate:.2f}%")
    
    # 显示最常见的经纬度(排除0.0, 0.0)
    print(f"\n最常见的5个经纬度位置:")
    coord_counts = valid_coords.groupby([lat_col, lon_col]).size().sort_values(ascending=False)
    print(coord_counts.head(5))
else:
    print("\n未找到经纬度列,请检查列名")
    print("可用的列名:", row_name)

# 绘制经纬度热力图
if lat_lon_cols:
    print("\n=== 绘制经纬度热力图 ===")
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 使用hexbin创建六边形热力图
    hexbin = plt.hexbin(valid_coords[lon_col], valid_coords[lat_col], 
                        gridsize=50,  # 网格大小,可以调整
                        cmap='YlOrRd',  # 颜色映射:黄色到橙色到红色
                        mincnt=1)  # 最小计数
    
    plt.xlabel('经度 (Longitude)', fontsize=12)
    plt.ylabel('纬度 (Latitude)', fontsize=12)
    plt.title('洛杉矶犯罪事件地理分布热力图\n(颜色越深=犯罪事件越多)', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cb = plt.colorbar(hexbin, label='犯罪事件数量')
    cb.set_label('犯罪事件数量', fontsize=11)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/crime_heatmap.png', 
                dpi=300, bbox_inches='tight')
    print("热力图已保存到: crime_heatmap.png")
    
    plt.show()
    
    # 可选:绘制散点密度图
    plt.figure(figsize=(12, 10))
    
    # 计算2D直方图用于颜色映射
    h, xedges, yedges = np.histogram2d(valid_coords[lon_col], valid_coords[lat_col], bins=100)
    
    # 为每个点找到对应的密度值
    x_idx = np.digitize(valid_coords[lon_col], xedges) - 1
    y_idx = np.digitize(valid_coords[lat_col], yedges) - 1
    x_idx = np.clip(x_idx, 0, h.shape[0]-1)
    y_idx = np.clip(y_idx, 0, h.shape[1]-1)
    colors = h[x_idx, y_idx]
    
    scatter = plt.scatter(valid_coords[lon_col], valid_coords[lat_col], 
                         c=colors, cmap='YlOrRd', s=1, alpha=0.5)
    
    plt.xlabel('经度 (Longitude)', fontsize=12)
    plt.ylabel('纬度 (Latitude)', fontsize=12)
    plt.title('洛杉矶犯罪事件地理分布散点图\n(颜色越深=密度越高)', fontsize=14, fontweight='bold')
    
    cb2 = plt.colorbar(scatter, label='区域密度')
    cb2.set_label('区域密度', fontsize=11)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('/Users/mannormal/Desktop/课程/y4t1/stat 4011/part2/crime_scatter.png', 
                dpi=300, bbox_inches='tight')
    print("散点图已保存到: crime_scatter.png")
    
    plt.show()