import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from Object_function import objective_function
from env import CropPlantingEnv

# 初始化环境
env = CropPlantingEnv()

# 遗传算法相关参数
population_size = 50
generations = 100
mutation_rate = 0.01


# 初始化种群
def init_population(size):
    return np.random.rand(size, env.num_plots, env.num_crops, env.num_years)


# 适应度函数，计算总利润
def fitness(individual):
    profit = objective_function(
        action=individual,
        yield_matrix=env.yield_matrix,
        price_matrix=env.price_matrix,
        cost_matrix=env.cost_matrix,
        sale_matrix=env.sale_matrix,
        plot_areas=env.plot_areas
    )
    return profit


# 选择操作
def selection(population):
    fitness_scores = np.array([fitness(individual) for individual in population])
    prob = fitness_scores / np.sum(fitness_scores)
    return population[np.random.choice(len(population), size=len(population), p=prob)]


# 交叉操作
def crossover(parent1, parent2):
    point = np.random.randint(1, env.num_plots * env.num_crops - 1)
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1.flat[point:], child2.flat[point:] = parent2.flat[point:], parent1.flat[point:]
    return child1, child2


# 变异操作
def mutate(individual):
    if np.random.rand() < mutation_rate:
        i = np.random.randint(env.num_plots)
        j = np.random.randint(env.num_crops)
        k = np.random.randint(env.num_years)
        individual[i, j, k] = np.random.rand()
    return individual


# 主遗传算法过程
def genetic_algorithm():
    population = init_population(population_size)
    best_solution = None
    best_fitness = float('-inf')
    fitness_history = []

    for generation in range(generations):
        population = selection(population)
        new_population = []

        # 交叉产生新个体
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[min(i + 1, len(population) - 1)]
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = np.array(new_population)

        # 记录最佳个体
        gen_best = max(population, key=fitness)
        gen_best_fitness = fitness(gen_best)
        fitness_history.append(gen_best_fitness)

        if gen_best_fitness > best_fitness:
            best_solution = gen_best
            best_fitness = gen_best_fitness

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

    return best_solution, fitness_history


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


# 可视化最佳种植方案
def plot_solution(solution):
    plt.figure(figsize=(12, 8))
    sns.heatmap(solution[:, :, 0], annot=True, fmt=".2f", cmap='Blues')
    plt.title("Optimal Crop Distribution Across Lands (Year 1)")
    plt.xlabel("Crops")
    plt.ylabel("Land Blocks")
    plt.show()


# 可视化第一年的最佳种植方案
plot_solution(best_solution)
