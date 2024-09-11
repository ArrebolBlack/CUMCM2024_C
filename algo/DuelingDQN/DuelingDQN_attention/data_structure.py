import pandas as pd
import numpy as np
from tensorboard.compat.tensorflow_stub.tensor_shape import vector
from tensorflow.python.ops.gen_linalg_ops import matrix_inverse


class data():

    def __init__(self):
        # 读取数据
        file_path_f1 = r'/home/shengjiaao/Newpython/C_2024/files/f_1.xlsx'
        farmland_data = pd.read_excel(file_path_f1, sheet_name='乡村的现有耕地')
        crop_data = pd.read_excel(file_path_f1, sheet_name='乡村种植的农作物')
        file_path_f2 = r'/home/shengjiaao/Newpython/C_2024/files/f_2.xlsx'
        file_path_nzw = r'/home/shengjiaao/Newpython/C_2024/files/农作物数据.xlsx'


        # 提取第三列（地块面积/亩）为列表
        self.plots_area = farmland_data['地块面积/亩'].tolist()
        self.plots_with_season_area = self.plots_area + self.plots_area[26:]
        self.plots_with_season_area = np.array(self.plots_with_season_area)

        # 创建年份列表 ndarray
        self.years = np.array(range(2024, 2031))

        # 获取唯一的地块名称
        plots = farmland_data['地块名称'].unique()

        # 初始化一个新的列表，按照先第一季度再第二季度的顺序
        plots_with_season = []
        # 先处理所有地块的第一季度
        for plot in plots:
            plots_with_season.append(f"{plot}_1季")
        # 再处理只有部分地块有第二季度的情况
        second_season_plots = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
                               "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14",
                               "E15", "E16",
                               "F1", "F2", "F3", "F4"]
        for plot in second_season_plots:
            plots_with_season.append(f"{plot}_2季")
        self.plots_with_season = np.array(plots_with_season)

        # 获取唯一的作物名称
        crops = crop_data['作物名称'].dropna()  # 删除 NaN
        crops = crops[~crops.str.contains('\(|（')]  # 删除包含说明的行
        self.crops = crops.unique()

        self.yield_matrix = self.make_matrix()
        self.price_matrix = self.make_matrix()
        self.cost_matrix = self.make_matrix()
        self.sale_matrix = self.make_matrix()

        file_path_yield = r'/home/shengjiaao/Newpython/C_2024/files/mu_chan_liang.csv'
        file_path_price = r'/home/shengjiaao/Newpython/C_2024/files/danjia(aver).csv'
        file_path_cost = r'/home/shengjiaao/Newpython/C_2024/files/cheng_ben.csv'
        file_path_result = r'/home/shengjiaao/Newpython/C_2024/files/result_2023(1).csv'

        yield_data = pd.read_csv(file_path_yield, header=None)
        price_data = pd.read_csv(file_path_price, header=None)
        cost_data = pd.read_csv(file_path_cost, header=None)
        result_data = pd.read_csv(file_path_result, header=None)

        # print(price_data.head())
        # print(price_data.tail())
        # print(price_data.to_numpy().shape)

        # 确保数据干净，无 NaN，可以先处理 NaN
        yield_data_clean = yield_data.fillna(0)
        price_data_clean = price_data.fillna(0)
        cost_data_clean = cost_data.fillna(0)
        result_data_clean = result_data.fillna(0)

        self.yield_matrix_2023 = yield_data_clean.to_numpy()
        self.price_matrix_2023 = price_data_clean.to_numpy()
        # print(self.price_matrix_2023.shape)
        self.cost_matrix_2023 = cost_data_clean.to_numpy()
        self.result_2023 = result_data_clean.to_numpy()
        self.sale_2023 = np.sum(self.result_2023 * self.yield_matrix_2023, axis=0)
        # print(self.sale_2023.shape)
        # 计算得到 种植比例
        self.result_2023_normalized = self.result_2023 / self.plots_with_season_area[:, np.newaxis]

        # 为 2024 到 2030 的每一年填充 yield_matrix, price_matrix, cost_matrix
        # 可在此处修改每年的更新逻辑
        for i, year in enumerate(self.years):
            self.yield_matrix[:, :, i] = self.yield_matrix_2023  # 按年份的索引填充数据
            self.price_matrix[:, :, i] = self.price_matrix_2023
            self.cost_matrix[:, :, i] = self.cost_matrix_2023
            self.sale_matrix[:, :, i] = self.sale_2023

        #
        self.input_dim = 1082


    def make_matrix(self):
        return np.zeros((len(self.plots_with_season), len(self.crops), len(self.years)))

    def get_item(self, matrix, plot_with_season_name, crop_name, year_name):
        # 如果需要查找特定地块、作物和年份的数据
        # 使用 np.where 查找索引
        plot_idx = np.where(self.plots_with_season == plot_with_season_name)  # 查找 'A1' 在 plots 中的索引
        crop_idx = np.where(self.crops == crop_name)  # 查找 '黄豆' 在 crops 中的索引
        year_idx = np.where(self.years == year_name)  # 查找 2024 在 years 中的索引
        print(matrix[plot_idx, crop_idx, year_idx])
        return matrix[plot_idx, crop_idx, year_idx]


    # TODO:
    def transition_3d(self, matrix_3d):

        new_matrix = np.zeros((self.input_dim, matrix_3d.shape[2]))

        for i in range(matrix_3d.shape[2]):
            vector = self.matrix2vector(matrix_3d[:, :, i])
            new_matrix[:, i] = vector

        return new_matrix

    def matrix2vector(self, matrix):


        # 粮食ABC.
        liangshi = matrix[0:26, 0:15].flatten()

        # 第一季DEF（包含水稻）.
        bunaihanshucai_shuijiaodi = matrix[26:54, 15:34].flatten()

        # 第二季D.
        naihan_shucai = matrix[54:62, 34:37].flatten()

        # 第二季E
        shiyongjun = matrix[62:78, 37:41].flatten()

        # 第二季F（智慧大棚）
        bunaihanshucai_dapeng = matrix[78:82, 16:34].flatten()



        vector = np.concatenate([liangshi, bunaihanshucai_shuijiaodi, naihan_shucai, shiyongjun, bunaihanshucai_dapeng])
        return vector

    def reverse_transition_3d(self, new_matrix, original_shape):
        # 初始化反向操作生成的3D矩阵
        matrix_3d = np.zeros(original_shape)

        for i in range(new_matrix.shape[1]):
            # 对每个向量进行逆操作
            matrix_3d[:, :, i] = self.vector2matrix(new_matrix[:, i])

        return matrix_3d

    def vector2matrix(self, vector):
        # 逆向提取每个区域的值，并重新放入矩阵
        matrix = np.zeros((82, 41))  # 根据原矩阵的形状初始化

        # 获取每个区域的大小
        liangshi_size = 26 * 15
        bunaihanshucai_shuijiaodi_size = 28 * 19
        naihan_shucai_size = 8 * 3
        shiyongjun_size = 16 * 4
        bunaihanshucai_dapeng_size = 4 * 18

        # 从向量中提取各个部分
        liangshi = vector[:liangshi_size].reshape(26, 15)
        bunaihanshucai_shuijiaodi = vector[liangshi_size:liangshi_size + bunaihanshucai_shuijiaodi_size].reshape(28, 19)
        naihan_shucai = vector[liangshi_size + bunaihanshucai_shuijiaodi_size:
                               liangshi_size + bunaihanshucai_shuijiaodi_size + naihan_shucai_size].reshape(8, 3)
        shiyongjun = vector[liangshi_size + bunaihanshucai_shuijiaodi_size + naihan_shucai_size:
                            liangshi_size + bunaihanshucai_shuijiaodi_size + naihan_shucai_size + shiyongjun_size].reshape(
            16, 4)
        bunaihanshucai_dapeng = vector[-bunaihanshucai_dapeng_size:].reshape(4, 18)

        # 还原到矩阵的相应位置
        matrix[0:26, 0:15] = liangshi
        matrix[26:54, 15:34] = bunaihanshucai_shuijiaodi
        matrix[54:62, 34:37] = naihan_shucai
        matrix[62:78, 37:41] = shiyongjun
        matrix[78:82, 16:34] = bunaihanshucai_dapeng

        return matrix


data_1 = data()

# if __name__ == "__main__":
#     #