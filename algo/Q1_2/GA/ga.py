import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from Object_function import objective_function
from env import CropPlantingEnv
import sys
sys.path.append('/home/shengjiaao/Newpython/C_2024')
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告

# 初始化环境
env = CropPlantingEnv()

# 遗传算法相关参数
population_size = 1000
generations = 3000
mutation_rate = 0.5


# 初始化种群
def init_population(size):
    return np.random.rand(size, env.num_plots, env.num_crops, env.num_years)


# 适应度函数，计算总利润
def fitness(individual):
    profit = objective_function(
        individual,  # 将 action 作为位置参数
        env.yield_matrix,
        env.price_matrix,
        env.cost_matrix,
        env.sale_matrix,
        env.plot_areas
    )
    return profit


# 修改并行计算适应度，提前初始化进程池
def parallel_fitness(population, pool):
    fitness_scores = pool.map(fitness, population)
    return np.array(fitness_scores)


# 选择操作
# def selection(population, fitness_scores):
#     prob = fitness_scores / np.sum(fitness_scores)
#     return population[np.random.choice(len(population), size=len(population), p=prob)]

def selection(population, fitness_scores):
    # 平移适应度值，确保所有值非负
    fitness_scores_shifted = fitness_scores - np.min(fitness_scores) + 1e-8  # 避免出现全零的概率
    prob = fitness_scores_shifted / np.sum(fitness_scores_shifted)
    return population[np.random.choice(len(population), size=len(population), p=prob)]



# 交叉操作
def crossover(parent1, parent2):
    point = np.random.randint(1, env.num_plots * env.num_crops * env.num_years - 1)
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1.flat[point:], child2.flat[point:] = parent2.flat[point:], parent1.flat[point:]
    return child1, child2


def mutate(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        # 选择多个随机位置进行变异
        num_mutations = np.random.randint(1, 5)  # 变异1到5个位置
        for _ in range(num_mutations):
            i = np.random.randint(env.num_plots)
            j = np.random.randint(env.num_crops)
            k = np.random.randint(env.num_years)
            individual[i, j, k] = np.random.rand()
    return individual


from tqdm import tqdm

def genetic_algorithm():
    population = init_population(population_size)
    best_solution = None
    best_fitness = float('-inf')
    fitness_history = []

    with Pool(processes=cpu_count()) as pool:
        for generation in tqdm(range(generations), desc="GA Progress"):
            current_mutation_rate = mutation_rate * (1 - generation / generations)
            fitness_scores = parallel_fitness(population, pool)
            population = selection(population, fitness_scores)
            new_population = []

            for i in range(0, len(population), 2):
                parent1 = population[i]
                parent2 = population[min(i + 1, len(population) - 1)]
                child1, child2 = crossover(parent1, parent2)
                new_population.append(mutate(child1, current_mutation_rate))
                new_population.append(mutate(child2, current_mutation_rate))

            population = np.array(new_population)

            gen_best = population[np.argmax(fitness_scores)]
            gen_best_fitness = np.max(fitness_scores)
            fitness_history.append(gen_best_fitness)

            if gen_best_fitness > best_fitness:
                best_solution = gen_best
                best_fitness = gen_best_fitness
                save_best_solution(best_solution, generation)

            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    return best_solution, fitness_history

import os
save_dir = '/home/shengjiaao/Newpython/C_2024/algo/GA'  # 保存最优解的目录
# 保存最优解
def save_best_solution(solution, generation):
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'best_solution_ga.npy')
    np.save(filename, solution)
    print(f"Best solution saved at generation {generation} as {filename}")


# 运行遗传算法
best_solution, fitness_history = genetic_algorithm()

# 总利润随代数变化趋势
plt.figure(figsize=(10, 6))
plt.plot(fitness_history, label='Total Profit')
plt.xlabel('Generations')
plt.ylabel('Profit')
plt.title('Total Profit over Generations')
plt.legend()
plt.grid(True)
plt.show()


# # 可视化最佳种植方案
# def plot_solution(solution):
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(solution[:, :, 0], annot=True, fmt=".2f", cmap='Blues')
#     plt.title("Optimal Crop Distribution Across Lands (Year 1)")
#     plt.xlabel("Crops")
#     plt.ylabel("Land Blocks")
#     plt.show()
#
#
# # 可视化第一年的最佳种植方案
# plot_solution(best_solution[:, :, 0])
