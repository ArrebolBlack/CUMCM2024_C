import pandas as pd
import numpy as np

# 读取数据
file_path_f1 = r'E:\C_2024\f_1.xlsx'
farmland_data = pd.read_excel(file_path_f1, sheet_name='乡村的现有耕地')
crop_data = pd.read_excel(file_path_f1, sheet_name='乡村种植的农作物')
file_path_f2 = r'E:\C_2024\f_2.xlsx'
file_path_nzw = r'E:\C_2024\农作物数据.xlsx'

# 创建年份列表
# years = list(range(2024, 2031))
years = np.array(range(2024, 2031))


# 初始化一个地块x作物x年份的三维矩阵，包含亩产量、种植成本和销售单价

# 获取唯一的地块名称
plots = farmland_data['地块名称'].unique()
# 初始化一个新的列表，按照先第一季度再第二季度的顺序
plots_with_season = []
# 先处理所有地块的第一季度
for plot in plots:
    plots_with_season.append(f"{plot}_1季")
# 再处理只有部分地块有第二季度的情况
second_season_plots = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
                       "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15", "E16",
                       "F1", "F2", "F3", "F4"]
for plot in second_season_plots:
    plots_with_season.append(f"{plot}_2季")
plots_with_season = np.array(plots_with_season)
# print(plots_with_season)

# 获取唯一的作物名称
# 去除 NaN 和包含非作物类的说明文字
crops = crop_data['作物名称'].dropna()  # 删除 NaN
crops = crops[~crops.str.contains('\(|（')]  # 删除包含说明的行
crops = crops.unique()

# 初始化 NumPy 数组 (地块 x 作物 x 年份)
yield_matrix = np.zeros((len(plots_with_season), len(crops), len(years)))
print(yield_matrix.shape)
# 如果需要查找特定地块、作物和年份的数据
# 使用 np.where 查找索引
plot_idx = np.where(plots_with_season == 'A1_1季') # 查找 'A1' 在 plots 中的索引
crop_idx = np.where(crops == '黄豆') # 查找 '黄豆' 在 crops 中的索引
year_idx = np.where(years == 2024) # 查找 2024 在 years 中的索引


# 访问对应位置的值
# print(yield_matrix[plot_idx, crop_idx, year_idx])

# # 填充示例：假设你有亩产量数据，可以直接填充
# yield_matrix[plot_idx, crop_idx, year_idx] = 400
# print(f"地块-季度: A1_1季, 作物: 黄豆, 年份: 2024, 亩产量: {yield_matrix[plot_idx, crop_idx, year_idx]}")

yield_matrix = np.zeros((len(plots_with_season), len(crops), len(years)))
price_matrix = np.zeros((len(plots_with_season), len(crops), len(years)))
cost_matrix = np.zeros((len(plots_with_season), len(crops), len(years)))

file_path_yield = r'E:\C_2024\mu_chan_liang.csv'
file_path_price = r'E:\C_2024\dan_jia(,).csv'
file_path_cost = r'E:\C_2024\cheng_ben.csv'
yield_data = pd.read_csv(file_path_yield)
price_data = pd.read_csv(file_path_price)
cost_data = pd.read_csv(file_path_cost)

# 确保数据干净，无 NaN，可以先处理 NaN
yield_data_clean = yield_data.fillna(0)
price_data_clean = price_data.fillna(0)
cost_data_clean = cost_data.fillna(0)

yield_matrix_2023 = np.zeros((len(plots_with_season), len(crops)))
price_matrix_2023 = np.zeros((len(plots_with_season), len(crops)))
cost_matrix_2023 = np.zeros((len(plots_with_season), len(crops)))

# 为 2024 到 2030 的每一年填充 yield_matrix, price_matrix, cost_matrix
for i, year in enumerate(years):
    yield_matrix[:, :, i] = yield_matrix_2023  # 按年份的索引填充数据
    price_matrix[:, :, i] = price_matrix_2023
    cost_matrix[:, :, i] = cost_matrix_2023

# 示例：查看 2024 年的 yield_matrix 数据
print("Yield matrix for 2024:")
print(yield_matrix[:, :, 0])  # 0 对应 2024 年

# # 检查 2024 年（i = 0 对应 2024 年）和 2023 年的数据是否相同
# is_correct_2024 = np.array_equal(yield_matrix[:, :, 0], yield_matrix_2023)
# print(f"2024 年的数据是否正确复制: {is_correct_2024}")
#
# # 检查 2030 年（i = 6 对应 2030 年）和 2023 年的数据是否相同
# is_correct_2030 = np.array_equal(yield_matrix[:, :, 6], yield_matrix_2023)
# print(f"2030 年的数据是否正确复制: {is_correct_2030}")
