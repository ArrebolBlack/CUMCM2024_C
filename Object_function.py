import numpy as np

from data_structure import data

data = data()

import time
def objective_function(action, yield_matrix, price_matrix, cost_matrix, sale_matrix, plot_areas):
    """
    计算总收益作为目标函数的值。

    参数:
    action: np.ndarray，动作空间的输入，即种植比例矩阵 (num_plots, num_crops, num_years) 即planting_matrix
    yield_matrix: np.ndarray，亩产量矩阵 (num_plots, num_crops, num_years)
    price_matrix: np.ndarray，销售价格矩阵 (num_plots, num_crops, num_years)
    cost_matrix: np.ndarray，种植成本矩阵 (num_plots, num_crops, num_years)
    sale_matrix: np.ndarray，预计销售额矩阵(num_crops, num_years)
    plot_areas: np.ndarray， 每块地实际面积(num_plots,)
    返回:
    total_profit: float，总收益
    """
    # t0 = time.time()

    # shuidao
    # for i in range(8):
    #     # 检查 action[26+i:26+i+1, 15:16] 是否存在值大于 0.1 的元素   水稻地种植比例大于0.1，则全种水稻
    #     if np.any(action[26 + i:26 + i + 1, 15:16] > 0.1):
    #         action[34 + i:34 + i + 1, :] = 0  # 将对应行设置为 0
    #
    #         action[26 + i:26 + i + 1, 15:16] = 1
    #     else:
    #         action[26 + i:26 + i + 1, 15:16] = 0  # 否则将该单元格设置为 0
    action[26:34, 15:16] = 0

    # # 确保动作矩阵的每个地块每年种植比例和为1
    # for plot in range(action.shape[0]):
    #     for year in range(action.shape[2]):
    #         sum_allocation = np.sum(action[plot, :, year])
    #         if sum_allocation > 0:
    #             action[plot, :, year] /= sum_allocation

    # 计算真实种植面积(亩数)
    real_planting_areas = plot_areas[:, np.newaxis, np.newaxis] * action

    # 计算总收益
    # 总产量 (82, 41, 7) 还没叠加
    P_total = yield_matrix * real_planting_areas
    # 计算  普通产量P1  和  昂贵智慧大棚产量P2
    P2 = np.zeros_like(P_total)
    P2[77:81, 16:34] = P_total[77:81, 16:34]
    P1 = P_total - P2

    # P_total = np.sum(P_total, axis=0)# (41, 7)
    P2 = np.sum(P2, axis=0) #(41, 7)
    P1 = np.sum(P1, axis=0) #(41, 7)

    # 先卖贵的， S1 shape (41, 7)
    S2 = np.minimum(sale_matrix, P2)
    S1 = np.minimum((sale_matrix - S2), P1)


    # 对price分块， 然后求在crop轴上 的非零值的平均。
    price_matrix_1 = price_matrix[0:77]
    price_matrix_2 = price_matrix[77:81]

    price_matrix_1 = np.nanmean(np.where(price_matrix_1 != 0, price_matrix_1, np.nan), axis=0)
    price_matrix_2 = np.nanmean(np.where(price_matrix_2 != 0, price_matrix_2, np.nan), axis=0)
    price_matrix_1 = np.nan_to_num(price_matrix_1, nan=0.0)
    price_matrix_2 = np.nan_to_num(price_matrix_2, nan=0.0)


    total_revenue = np.sum((S2 * price_matrix_2 + S1 * price_matrix_1))

    # 计算总成本 亩数 x 成本 元/亩
    total_cost = np.sum(cost_matrix * real_planting_areas)
    # print(total_revenue, total_cost)

    total_profit = total_revenue - total_cost


    # reward = total_profit - decay
    # # normalize result::
    # # Z-score normalize
    normalized_total_profit = z_score_normalization(total_profit, 16927795.85910371, 178851.17323063727)
    normalized_total_profit = z_score_normalization(normalized_total_profit, 1694.0602617498055, 21.85462273898446)
    weight_main = 1

    # t1 = time.time()
    # print("total_profit_time_cost:", t1 - t0)

    #
    weight_1 = 5
    decay_1 = check_upper_bound(action)  # Avg = 11200 >= 0
    # decay_1_normalized = z_score_normalization(decay_1, 11193.185174639553, 44.183047156748025)
    decay_1_normalized = min_max_normalization(decay_1, 0, 12000)

    # t2 = time.time()
    # print("decay_1_time_cost:", t2 - t1)
    # # 计算惩罚项

    # if not suitable_plant_decay(action):
    #     decay_2 = -5

    decay_3 = 0
    if not continuous_cropping_decay_in_range(action):
        decay_3 = -5

    # t3 = time.time()
    # print("decay_3_time_cost:", t3 - t2)
    decay_4 = 0
    if not plant_beans_decay_in_range_of_years(action):
        decay_4 = -5
    # t4 = time.time()
    # print("decay_4_time_cost:", t4 - t3)

    # weight_5 = 1
    # decay_5 = planting_dispersion_decay(action)
    # decay_5_normalized = min_max_normalization(decay_5, 0, 2600)
    weight_5 = 10
    decay_5 = cv(action)
    decay_5_normalized = min_max_normalization(decay_5, 0, 336)

    weight_6 = 5
    decay_6 = area_deviation(action)
    decay_6_normalized = min_max_normalization(decay_6, 0, 2600)
    # t5 = time.time()
    # print("decay_6_time_cost:", t5 - t4)

    terms = [
        weight_main * normalized_total_profit,
        - weight_1 * decay_1_normalized,
        #  decay_2,
         decay_3,
         decay_4,
        - weight_5 * decay_5_normalized,
        - weight_6 * decay_6_normalized,
    ]

    # 求和选中的项
    normalized_reward = sum(terms)

    # Reward Clipping
    normalized_reward_clipped = reward_clipping(normalized_reward, -100, 250)
    # return normalized_reward_clipped + 100
    return total_profit


def reward_clipping(reward, min_reward=-10, max_reward=10):
    return max(min(reward, max_reward), min_reward)

def z_score_normalization(reward, mean_value, std_value):
    return (reward - mean_value) / std_value

def min_max_normalization(reward, min_value, max_value):
    return (reward - min_value) / (max_value - min_value)

def suitable_plant_decay(df: np.ndarray):
    if (df.shape != (82, 41, 7)):
        return False
    slices = np.split(df, 7, axis=2)

    nan = data.cost_matrix[:, :, 0]

    for _, slice_i in enumerate(slices):
        slice_i = np.squeeze(slice_i)
        if not np.all((nan == 0) | (slice_i == 0)):
            return False
    # 适宜种植
    return True


def range_bound(action):
    # 1. 计算超出 [0, 1] 范围的软约束惩罚
    lower_bound_penalty = np.sum(np.maximum(0 - action, 0) ** 2)  # 对小于 0 的值进行惩罚
    upper_bound_penalty = np.sum(np.maximum(action - 1, 0) ** 2)  # 对大于 1 的值进行惩罚

    # 总惩罚：可以通过系数调整惩罚的强度
    boundary_penalty = lower_bound_penalty + upper_bound_penalty
    penalty_coefficient = 1  # 惩罚项的系数，可根据需要调整
    return boundary_penalty * penalty_coefficient

def check_upper_bound(action):
    # 计算每一行的和
    row_sums = np.sum(action, axis=1)

    # 计算每一行的偏差，偏差是和1的差值
    deviation = np.abs(row_sums - np.ones_like(row_sums))

    # 返回偏差的总和，或平均偏差
    total_deviation = np.sum(deviation)
    # print(total_deviation)
    return total_deviation  # 偏差值越接近0，说明约束条件越接近满足

def continuous_cropping_decay_in_range(action):
    # 1、先拼接action和23年result
    total_matrix = np.concatenate((data.result_2023_normalized[..., np.newaxis], action), axis=2)
    for i in range(0, total_matrix.shape[2]):
        current_matrix = total_matrix[:, :, i:i+1]
        sum_current_matrix = np.sum(current_matrix, axis=2)
        # 检查 sum current matrix中，是否有大于一的值，如果有，返回False
        if np.any(sum_current_matrix > 1):
            return False
    return True



def plant_beans_decay_in_range_of_years(action):
    total_matrix = np.concatenate((data.result_2023_normalized[..., np.newaxis], action), axis=2)
    for i in range(0, total_matrix.shape[2]-1):
        current_matrix = total_matrix[:, :, i:i+2]
        sum_current_matrix = np.sum(current_matrix, axis=2)
        selected_rows_sum = np.sum(sum_current_matrix[:, 0:5], axis=1) + np.sum(sum_current_matrix[:, 16:19], axis=1)
        if np.any(selected_rows_sum > 1):
            return False
    return True


from joblib import Parallel, delayed

def calculate_distance_sum(vector):
    """
    计算向量中所有1的元素之间的距离和

    :param vector: 01向量
    :return: 所有1的元素之间的距离和
    """
    indices = np.where(vector == 1)[0]
    if len(indices) < 2:
        return 0
    distances = np.diff(indices)
    return np.sum(distances)


def process_action_matrix(action_matrix):
    """
    处理三维的action矩阵，对每个列（每个年份的每个作物）进行并行计算

    :param action_matrix: 形状为 (地块, 作物, 年份) 的三维action矩阵
    :return: 计算结果矩阵，形状为 (作物, 年份)，对应每个作物每个年份的1之间的距离和
    """
    n_blocks, n_crops, n_years = action_matrix.shape

    # 并行计算，直接返回每个作物每个年份的距离和
    results = Parallel(n_jobs=-1)(
        delayed(calculate_distance_sum)(action_matrix[:, crop, year])
        for crop in range(n_crops)
        for year in range(n_years)
    )

    # 将结果重塑为 (作物, 年份) 的矩阵
    result_matrix = np.array(results).reshape((n_crops, n_years))

    return result_matrix
def planting_dispersion_decay(action):
    # 将非零值转换为1
    action_binary = np.where(action != 0, 1, 0)
    # 计算所有作物所有年份的1之间的距离和的总和
    total_dispersion = np.sum(process_action_matrix(action_binary))

    return total_dispersion
def area_deviation(action):
    lower_bound = 0.1
    # 计算满足条件 0 < action < lower_bound 的元素个数
    count = np.sum((action > 0) & (action < lower_bound))
    return count


def cv(action):
    cv_total = 0
    num_years = action.shape[2]
    num_plots = action.shape[0]

    for year in range(num_years):
        for plot in range(num_plots):
            tem_vec = action[plot, :, year]
            mean = np.mean(tem_vec)
            std = np.std(tem_vec)
            if mean != 0:  # 防止均值为0导致除零错误
                cv_total += std / mean
            else:
                cv_total += 0  # 均值为0时CV定义为0或根据实际情况调整

    # return cv_total
    return min_max_normalization(cv_total, 0, 336)

if __name__ == "__main__":
    from env import CropPlantingEnv
    env = CropPlantingEnv()
    # 假设 data.result_2023_normalized 是一个形状为 (82, 41) 的二维矩阵
    result_2023_normalized = data.result_2023_normalized

    # 在最后一个维度添加一个新维度，使其成为 (82, 41, 1)
    expanded_matrix = np.expand_dims(result_2023_normalized, axis=-1)

    # 沿着新维度重复7次，形成一个形状为 (82, 41, 7) 的三维矩阵
    action = np.tile(expanded_matrix, (1, 1, 7))
    reward = objective_function(action, env.yield_matrix, env.price_matrix, env.cost_matrix, env.sale_matrix,
                                env.plot_areas)
    print(reward)

    real_planting_areas = env.plot_areas[:, np.newaxis, np.newaxis] * action
    W = np.sum(real_planting_areas * env.yield_matrix * env.price_matrix)
    C = np.sum(real_planting_areas * env.cost_matrix)
    print(W)
    print(C)