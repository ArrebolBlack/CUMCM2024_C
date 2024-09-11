import gym
from gym import spaces
import numpy as np
import pandas as pd
from Object_function import objective_function
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

        # 归一化，输出的是每种作物比例
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_plots, self.num_crops, self.num_years), dtype=np.float32)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_plots, self.num_crops, 4), dtype=np.float32)
        # Initialize
        self.planting_matrix = self.data.make_matrix()

        self.observation =  np.stack(
            [self.yield_matrix_2023, self.price_matrix_2023, self.cost_matrix_2023, self.result_matrix_2023], axis=2)
        self.info = None


    def step(self, action):

        # assert action.shape == (self.num_plots, self.num_crops, self.num_years), \
        #     f"Action shape {action.shape} does not match the expected shape {(self.num_plots, self.num_crops, self.num_years)}"

        assert action.shape == (self.data.input_dim, self.num_years), f"Action shape {action.shape} does not match the expected shape {(self.data.input_dim, self.num_years)}"
        action = self.data.reverse_transition_3d(action)


        reward = objective_function(action, self.yield_matrix, self.price_matrix, self.cost_matrix, self.sale_matrix, self.plot_areas)

        observation = self.get_observation(self.observation)

        done = True
        info = self.info
        return observation, reward, done, info


    def reset(self):
        self.planting_matrix = self.data.make_matrix()
        self.yield_matrix = self.data.yield_matrix
        self.price_matrix = self.data.price_matrix
        self.cost_matrix = self.data.cost_matrix


        observation = self.get_observation(self.observation)
        # 返回初始obs
        return observation

    def get_observation(self, observation_matrix):
        """
        :param observation_matrix:
        :return: observation_vector:
        """
        observation = self.data.transition_3d(observation_matrix)

        return observation