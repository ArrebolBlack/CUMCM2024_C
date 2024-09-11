# 2024年国赛 CUMCM C题代码

## 项目简介
本项目为 2024 年全国大学生数学建模竞赛 (CUMCM) C 题的代码。该题目可以视为一个在所有种植策略的空间中进行搜索的问题，或者是一个序列生成问题。我们的思路是首先构造一个充分的评估函数，计算种植策略的总利润（得分），然后定义一个强化学习环境，在动作空间中进行搜索。

## 核心思路
1. 构造足够充分的评估函数，评估种植策略的总利润。
2. 定义强化学习环境，在动作空间中进行搜索，优化种植策略。

## 核心代码结构
- **data_structure.py**：处理数据，定义通用数据形式，提供接口。将所有信息转化成 `result` 格式（plots, crops, years），以便后续所有计算以张量形式运行。
- **Object_function.py**：定义目标函数及约束条件，构成评估函数，也就是 `loss function`。
- **env.py**：定义强化学习环境，实现单步决策。
- **algo**：包含各种算法在强化学习环境中的实现。

## 工具
- **eval_obj_no_parallel.py** 系列：用于多次采样统计一个函数的分布，可以用于检查目标函数及约束条件的分布，并进行归一化处理。
- 为了加快计算速度，提供了各个算法和工具函数的并行、多线程实现，请注意区分

## 优化与压缩搜索空间
初始定义的搜索空间为 `result` 的空间，大小为 `(82, 41, 7)`。经过优化，删除了无意义或不可能出现的点，压缩成 `(1082, 7)`。压缩空间后的环境文件为 `env_new.py`。在 `data_structure.py` 中构建了来回转换的函数。

## 对题目的解答
### 第一题：
1. 对于 **第一问**，使用 `data_structure.py` + `Object_function.py` + `env_new.py`。
2. 对于 **第二问**，修改目标函数主体部分的计算方式，使用 `data_structure.py` + `Object_function_1.2.py` + `env_new.py`。

### 第二题：
更换未来七年的数据模拟，原本是将 2023 年的情况复制，现在按照题目条件进行随机模拟，使用 `data_structure_2.py` + `Object_function.py` + `env_new.py`。

## 简单步骤说明
在简化的情况下：

1. 第一题的第一问：`data_structure.py` + `Object_function.py` + `env_new.py`
2. 第一题的第二问：`data_structure.py` + `Object_function_1.2.py` + `env_new.py`
3. 第二题：`data_structure_2.py` + `Object_function.py` + `env_new.py`

## 简单想法
先将环境模拟得足够真实，剩下的交给算法进行优化，通过算法在搜索空间中不断寻找最佳的种植策略。

## 算法介绍
本项目中使用了以下算法：

### 简单搜索算法
- **Random_Agent**：随机探索（猴子策略）。
- **epsilon-greedy 策略**：在一定概率下随机选择动作。
- **UCB (上限置信界)**：由于这是一个单步决策问题，因此视作多臂老虎机问题，使用 UCB 算法进行探索。

### 深度强化学习算法
- **DQN**：经典的深度 Q 网络。
- **DuelingDQN**：基于 DQN 的改进版本，引入了决斗网络架构，分离了状态价值和动作优势函数。
- **DuelingDQN + Attention**：结合 Transformer 的嵌入层，捕捉输入和输出之间的关系，提升 Q 值预测的准确性。
- **PPO + Transformer**：视作序列生成问题，输入 2023-2030 年的价格、成本、销售量、产量等信息，输出 2024-2030 年的种植策略。使用 Transformer 捕捉输入与输出的关系，PPO 框架进行优化。

### 启发式算法
- **GA (遗传算法)**：通过自然选择、交叉和变异寻找最优解。
- **PSO (粒子群算法)**：模拟粒子群的协作，寻找最优解。


## 项目结构

- `algo` 算法实现目录
  - `DQN` DQN 算法
  - `PPO+Transformer` PPO + Transformer 实现
  - ... 其他算法
- `data_structure.py` 数据处理及通用接口定义
- `Object_function.py` 目标函数和约束条件
- `env.py` 强化学习环境定义
- `eval_obj_no_parallel.py` 目标函数的多次采样统计
- `README.md` 项目说明文档
- ...



## 欢迎交流！欢迎star
有问题or交个朋友：qq邮箱 207804110@qq.com



