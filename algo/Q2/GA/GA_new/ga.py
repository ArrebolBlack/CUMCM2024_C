import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
# from Object_function import objective_function
from env_new import CropPlantingEnv
import os
import warnings

warnings.filterwarnings("ignore")  # 忽略所有警告

# 初始化环境
env = CropPlantingEnv()

# 遗传算法相关参数
population_size = 1000
generations = 1000
mutation_rate = 0.5
elite_size = int(0.05 * population_size)  # 精英保留比例
tournament_size = 5  # 锦标赛选择中，每次随机挑选的个体数


# 初始化种群
def init_population(size):
    return np.random.rand(size, env.data.input_dim, env.num_years)


# 适应度函数，计算总利润
def fitness(individual):
    _, profit, _, _ = env.step(individual)

    return profit


# 修改并行计算适应度，提前初始化进程池
def parallel_fitness(population, pool):
    fitness_scores = pool.map(fitness, population)
    return np.array(fitness_scores)


# 锦标赛选择机制
def tournament_selection(population, fitness_scores):
    selected = []
    for _ in range(len(population)):
        # 从种群中随机挑选 tournament_size 个个体
        indices = np.random.randint(0, len(population), tournament_size)
        # 挑选其中适应度最高的个体
        winner = np.argmax(fitness_scores[indices])
        selected.append(population[indices[winner]])
    return np.array(selected)


# 多点交叉操作
def crossover(parent1, parent2):
    points = np.random.randint(1, env.data.input_dim * env.num_years - 1, size=2)
    points.sort()  # 确保交叉点顺序
    child1, child2 = parent1.copy(), parent2.copy()
    child1.flat[points[0]:points[1]], child2.flat[points[0]:points[1]] = parent2.flat[
                                                                         points[0]:points[1]], parent1.flat[
                                                                                               points[0]:points[1]]
    return child1, child2


def mutate(individual, mutation_rate):
    if np.random.rand() < mutation_rate:
        num_mutations = np.random.randint(1, 5)  # 变异1到5个位置
        for _ in range(num_mutations):
            i = np.random.randint(env.data.input_dim)  # 随机选择 input_dim 的索引
            j = np.random.randint(env.num_years)       # 随机选择 num_years 的索引
            # 对个体的指定基因进行变异
            individual[i, j] = np.clip(individual[i, j] + np.random.randn() * 0.1, 0, 1)
    return individual


# 遗传算法主循环
def genetic_algorithm():
    population = init_population(population_size)
    best_solution = None
    best_fitness = float('-inf')
    fitness_history = []

    with Pool(processes=cpu_count()) as pool:
        for generation in tqdm(range(generations), desc="GA Progress"):
            current_mutation_rate = mutation_rate * (1 - generation / generations)
            fitness_scores = parallel_fitness(population, pool)

            # 精英保留
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            elites = population[elite_indices]

            # 锦标赛选择
            population = tournament_selection(population, fitness_scores)
            new_population = []

            for i in range(0, len(population) - elite_size, 2):
                parent1 = population[i]
                parent2 = population[min(i + 1, len(population) - 1)]
                child1, child2 = crossover(parent1, parent2)
                new_population.append(mutate(child1, current_mutation_rate))
                new_population.append(mutate(child2, current_mutation_rate))

            # 加入精英
            new_population.extend(elites)
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


# 保存最优解
def save_best_solution(solution, generation):
    save_dir = '/home/ziwu/Newpython/C_2024/algo/Q2/GA/GA_new'
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'best_ga_2.npy')
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
