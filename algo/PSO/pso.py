import numpy as np
from pyswarm import pso
from data_structure import data
from Object_function import objective_function
from env import CropPlantingEnv
import matplotlib.pyplot as plt

# 初始化数据和环境
data = data()
env = CropPlantingEnv()

# 定义PSO目标函数，并记录收敛过程
convergence_history = []


def pso_objective_function(x):
    # 将1D向量重塑为3D矩阵（82, 41, 7）
    matrix_shape = (82, 41, 7)
    action = x.reshape(matrix_shape)

    # 计算奖励值作为目标函数的返回值
    reward = objective_function(action, env.yield_matrix, env.price_matrix, env.cost_matrix, env.sale_matrix,
                                env.plot_areas)
    # 记录当前最优值
    current_best = -reward
    convergence_history.append(current_best)

    # 返回负的reward，因为pso是最小化问题
    return -reward


# 定义三维矩阵的形状和展平后的维度
matrix_shape = (82, 41, 7)
dim = np.prod(matrix_shape)

# 定义优化变量的上下界
lower_bounds = [-1] * dim
upper_bounds = [1] * dim

# 进行PSO优化
best_position, best_value = pso(pso_objective_function, lower_bounds, upper_bounds,
                                swarmsize=200,  # 增加粒子群的数量
                                maxiter=1000,  # 增加迭代次数
                                omega=0.5,  # 惯性权重
                                phip=1.5,  # 认知系数
                                phig=1.5,  # 社会系数
                                minstep=1e-8,  # 最小步长，用于判断收敛
                                minfunc=1e-8,  # 最小目标函数值，用于判断收敛
                                debug=True)  # 启用调试信息输出

# 将优化后的最佳位置重塑为原来的3D矩阵形状
best_matrix = best_position.reshape(matrix_shape)

# 保存最佳动作为.npy文件
np.save("best_action.npy", best_matrix)

# 输出结果
print("Best Matrix:\n", best_matrix)
print("Best Value:", -best_value)  # 还原成正值

# 可视化收敛过程
plt.plot(convergence_history)
plt.title('PSO Convergence')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness Value')
plt.grid(True)
plt.show()
