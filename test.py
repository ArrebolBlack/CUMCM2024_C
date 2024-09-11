import pandas as pd
import numpy as np

# 读取数据
file_path = r'E:\C_2024\your_excel_file.xlsx'
crop_data = pd.read_excel(file_path, sheet_name='sheet_name_here')

# 创建年份列表
years = list(range(2024, 2031))

# 初始化一个地块x作物x年份的三维矩阵，包含亩产量、种植成本和销售单价
plots = crop_data['地块名称'].unique()  # 获取唯一的地块名称
crops = crop_data['作物名称'].unique()  # 获取唯一的作物名称

# 创建空字典，用于存储每块地的矩阵
matrix = {plot: {crop: {year: {"亩产量": None, "种植成本": None, "销售单价": None}
                        for year in years}
                 for crop in crops}
          for plot in plots}

# 填充数据矩阵
for _, row in crop_data.iterrows():
    plot = row['地块名称']
    crop = row['作物名称']
    for year in years:
        matrix[plot][crop][year]['亩产量'] = row['亩产量/斤']
        matrix[plot][crop][year]['种植成本'] = row['种植成本/(元/亩)']
        matrix[plot][crop][year]['销售单价'] = row['销售单价/(元/斤)']

# 示例：打印某个地块的某作物在某年份的数据
print(matrix['A1']['黄豆'][2024])
