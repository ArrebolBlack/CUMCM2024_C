import gym
from gym import spaces
import numpy as np
import pandas as pd
from Object_function_1_2 import objective_function
from data_structure import data

class CropPlantingEnv(gym.Env):

    def __init__(self):
        self.num_plots = 82 # 地块数量 （含季度）
        self.num_crops = 41 # 作物种类
        self.num_years = 7 # 总共预测7年

        self.data = data()

        plot_area = self.data.plots_area
        plot_area += plot_area[26:]

        self.plot_areas = np.array(plot_area)
        self.yield_matrix = self.data.yield_matrix
        self.price_matrix = self.data.price_matrix
        self.cost_matrix = self.data.cost_matrix

        self.sale_matrix = np.sum(self.data.sale_matrix, axis=0)

        self.yield_matrix_2023 = self.data.yield_matrix_2023
        self.price_matrix_2023 = self.data.price_matrix_2023
        self.cost_matrix_2023 = self.data.cost_matrix_2023
        self.result_matrix_2023 = self.data.result_2023
        self.sale_matrix_2023 = self.data.sale_2023


        self.input_dim = 1082
        # obs: (82,41,8)
        self.obs_yield_matrix = np.concatenate((self.yield_matrix_2023[:, :, np.newaxis], self.yield_matrix), 2)
        self.obs_price_matrix = np.concatenate((self.price_matrix_2023[:, :, np.newaxis], self.price_matrix), 2)
        self.obs_cost_matrix = np.concatenate((self.cost_matrix_2023[:, :, np.newaxis], self.cost_matrix), 2)

        # 将长度为 328 的 sale_matrix 展平
        flattened_sale = self.sale_matrix.flatten()

        # 使用 np.pad() 将其补零到 1082
        padded_sale = np.pad(flattened_sale, (0, self.input_dim - len(flattened_sale)), 'constant')
        self.obs_sale = padded_sale.reshape(self.input_dim, 1)
        # 打印每个数组的形状以调试
        print(self.data.transition_3d(self.obs_yield_matrix).shape)
        print(self.data.transition_3d(self.obs_price_matrix).shape)
        print(self.data.transition_3d(self.obs_cost_matrix).shape)
        print(self.data.transition_3d(self.result_matrix_2023[:, :, np.newaxis]).shape)
        print(self.obs_sale.shape)

        # 拼接数组
        self.observation = np.concatenate(
            [self.data.transition_3d(self.obs_yield_matrix),
             self.data.transition_3d(self.obs_price_matrix),
             self.data.transition_3d(self.obs_cost_matrix),
             self.data.transition_3d(self.result_matrix_2023[:, :, np.newaxis]),
             self.obs_sale],  # 使用补零后的 sales_matrix
            axis=1
        )

        print("self.observation_shape===:", self.observation.shape)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.input_dim, self.num_years), dtype=np.float32)

        self.observation_space = spaces.Box(low=0.0, high=10000.0, shape=(self.input_dim, 26), dtype=np.float32)



        self.info = None


    def step(self, action):


        assert action.shape == (self.data.input_dim, self.num_years), f"Action shape {action.shape} does not match the expected shape {(self.data.input_dim, self.num_years)}"

        action = self.data.reverse_transition_3d(action, (82, 41, 7))

        reward = objective_function(action, self.yield_matrix, self.price_matrix, self.cost_matrix, self.sale_matrix, self.plot_areas)


        done = True
        info = self.info
        return self.observation, reward, done, info


    def reset(self):
        return self.observation

