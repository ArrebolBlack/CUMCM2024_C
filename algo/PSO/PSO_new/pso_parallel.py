import numpy as np
from pyswarm import pso
from data_structure import data
from Object_function import objective_function
from env_new import CropPlantingEnv
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm  # 新增导入
import time  # 用于记录迭代时长
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告

# 初始化数据和环境
data = data()
env = CropPlantingEnv()

# 定义PSO目标函数
def pso_objective_function(x):
    matrix_shape = (1082, 7)
    action = x.reshape(matrix_shape)
    _, reward, _, _ = env.step(action)
    return -reward

# 并行化适应度评估
def parallel_objective_function(positions):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        fitness_values = pool.map(pso_objective_function, positions)
    return fitness_values

# 动态调整惯性权重的PSO实现
def parallel_pso(func, lb, ub, swarmsize=100, maxiter=100, omega=0.9, omega_end=0.4, phip=1.5, phig=1.5, minstep=1e-8, minfunc=1e-8, velocity_clamp=None):
    # 初始化
    lb = np.array(lb)
    ub = np.array(ub)
    dimensions = len(lb)
    x = np.random.uniform(low=lb, high=ub, size=(swarmsize, dimensions))
    v = np.zeros_like(x)
    p = np.copy(x)
    pbest = np.full(swarmsize, np.inf)
    gbest = None
    gbest_fitness = np.inf

    # 初始化收敛记录
    convergence_history = []
    iteration_times = []

    for i in tqdm(range(maxiter), desc="PSO Iterations"):
        start_time = time.time()

        fitness = parallel_objective_function(x)

        # 更新个体和全局最佳
        for j in range(swarmsize):
            if fitness[j] < pbest[j]:
                pbest[j] = fitness[j]
                p[j] = x[j]
            if fitness[j] < gbest_fitness:
                gbest_fitness = fitness[j]
                gbest = x[j]

        # 记录当前迭代的全局最佳适应度
        convergence_history.append(gbest_fitness)

        # 计算迭代时长
        iteration_time = time.time() - start_time
        iteration_times.append(iteration_time)

        # 如果满足最小目标函数值和步长的停止条件
        if np.abs(gbest_fitness - minfunc) < minstep:
            break

        # 动态调整惯性权重，随着迭代次数逐渐减小
        omega_dynamic = omega - ((omega - omega_end) * i / maxiter)

        # 更新速度和位置
        rp = np.random.uniform(size=(swarmsize, dimensions))
        rg = np.random.uniform(size=(swarmsize, dimensions))
        v = omega_dynamic * v + phip * rp * (p - x) + phig * rg * (gbest - x)

        # 速度限制 (Velocity Clamping)
        if velocity_clamp is not None:
            v = np.clip(v, -velocity_clamp, velocity_clamp)

        x = np.clip(x + v, lb, ub)

        if i % 10 == 0:
            print(f"Iteration {i + 1}/{maxiter}, Global Best Fitness: {gbest_fitness}")

    # 返回全局最佳位置和对应的适应度值，以及收敛历史记录
    return gbest, gbest_fitness, convergence_history, iteration_times

# 定义新动作空间的形状和展平后的维度
matrix_shape = (env.data.input_dim, env.num_years)
dim = np.prod(matrix_shape)

# 定义优化变量的上下界
lower_bounds = [0] * dim
upper_bounds = [1] * dim

# 进行PSO优化
best_position, best_value, convergence_history, iteration_times = parallel_pso(pso_objective_function, lower_bounds, upper_bounds,
                                                              swarmsize=500,  # 增加粒子群的数量
                                                              maxiter=1000,  # 增加迭代次数
                                                              omega=0.9,  # 动态调整初始惯性权重
                                                              omega_end=0.4,  # 迭代结束时的惯性权重
                                                              phip=1.5,  # 认知系数
                                                              phig=1.5,  # 社会系数
                                                              minstep=1e-8,  # 最小步长
                                                              minfunc=1e-8,  # 最小目标函数值
                                                              velocity_clamp=0.2)  # 速度限制，防止发散

# 将优化后的最佳位置重塑为原来的矩阵形状
best_matrix = best_position.reshape(matrix_shape)

# 保存最佳动作为.npy文件
np.save("best_action_pso_parallel_.npy", best_matrix)

# 输出结果
print("Best Matrix:\n", best_matrix)
print("Best Value:", -best_value)  # 还原成正值

# 可视化收敛过程
plt.figure(figsize=(10, 6))
plt.plot(convergence_history)
plt.title('PSO Convergence')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness Value')
plt.grid(True)
plt.show()

# 可视化训练时长
plt.figure(figsize=(10, 6))
plt.plot(iteration_times)
plt.title('Training Time per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.show()
