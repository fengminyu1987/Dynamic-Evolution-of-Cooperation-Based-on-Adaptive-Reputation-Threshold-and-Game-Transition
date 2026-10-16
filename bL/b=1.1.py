import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from matplotlib.colors import ListedColormap

# ---------------------------
# 参数设置
# ---------------------------
N = 50 * 50  # 网络中个体数目 (50x50方格网络)
b_L = 1.1  # 低价值博弈基础回报

# 模型参数
c = 1  # 合作成本
delta = 0.01  # 声誉变化幅度
m = 0.6  # 适应度权重
p_game = 0.9  # 高价值博弈选择概率
beta = 1.0  # 费米规则强度
T = 3000  # 总迭代轮数

# 需要记录策略分布的时间步
record_steps = [0, 10, 100, 500, 1000, 1500, 2000, 2500, 3000]


# ---------------------------
# 创建方格网络
# ---------------------------
def create_grid_network(n, m):
    G = nx.grid_2d_graph(n, m)  # 创建n×m的方格网络
    # 将节点标签从(i,j)转换为单一整数索引
    mapping = {(i, j): i * m + j for i in range(n) for j in range(m)}
    G = nx.relabel_nodes(G, mapping)
    return G


# ---------------------------
# 自适应阈值更新函数
# ---------------------------
def update_theta(reputations):
    return np.mean(list(reputations.values()))


# ---------------------------
# 博弈函数
# ---------------------------
def play_game(i, j, strategies, reputations, theta, p, c, b_L):
    ri, rj = reputations[i], reputations[j]
    si, sj = strategies[i], strategies[j]

    # 判断节点是否为高声誉
    high_i = ri > theta
    high_j = rj > theta

    # 选择博弈类型
    if high_i and high_j:
        game_type = 'high'
    elif not high_i and not high_j:
        game_type = 'low'
    else:
        game_type = 'high' if random.random() < p else 'low'

    # 根据博弈类型计算支付
    if game_type == 'high':
        if si == 1 and sj == 1:
            pi = (ri + rj) / 2 - c  # 双方合作：共享声誉平均值
            pj = (ri + rj) / 2 - c
        elif si == 1 and sj == 0:
            pi = -c  # 合作者承担成本
            pj = rj / 2  # 背叛者收益
        elif si == 0 and sj == 1:
            pi = ri / 2  # 背叛者收益
            pj = -c  # 合作者承担成本
        else:
            pi = 0  # 双方背叛无收益
            pj = 0
    else:
        if si == 1 and sj == 1:
            pi = b_L - c
            pj = b_L - c
        elif si == 1 and sj == 0:
            pi = -c
            pj = b_L
        elif si == 0 and sj == 1:
            pi = b_L
            pj = -c
        else:
            pi = 0
            pj = 0

    # 声誉更新：合作获得 +delta，背叛扣减 delta
    if si == 1 and sj == 1:
        reputations[i] += delta
        reputations[j] += delta
    elif si == 1 and sj == 0:
        reputations[i] += 2 * delta
        reputations[j] -= 2 * delta
    elif si == 0 and sj == 1:
        reputations[i] -= 2 * delta
        reputations[j] += 2 * delta
    else:
        reputations[i] -= delta
        reputations[j] -= delta
    reputations[i] = np.clip(reputations[i], 0, 2)
    reputations[j] = np.clip(reputations[j], 0, 2)

    return pi, pj


# ---------------------------
# 策略更新函数
# ---------------------------
def strategy_update(i, strategies, payoffs, beta, m, reputations, G):
    neighbors = list(G.neighbors(i))
    if not neighbors:
        return strategies

    j = np.random.choice(neighbors)

    fi = (1 - m) * reputations[i] + m * payoffs[i]
    fj = (1 - m) * reputations[j] + m * payoffs[j]

    prob_update = 1 / (1 + np.exp(-beta * (fj - fi)))
    if random.random() < prob_update:
        strategies[i] = strategies[j]
    return strategies


# ---------------------------
# 绘制策略分布图
# ---------------------------
# ---------------------------
# 绘制策略分布图
# ---------------------------
def plot_strategy_distribution(strategies, reputations, theta, step, grid_size=50):
    # 创建策略分布矩阵
    strategy_matrix = np.zeros((grid_size, grid_size, 3))  # RGB颜色矩阵

    # 定义四种策略的颜色
    # 高声誉合作：深红色 [0.8, 0, 0]
    # 低声誉合作：浅红色 [1, 0.5, 0.5]
    # 高声誉背叛：深蓝色 [0, 0, 0.8]
    # 低声誉背叛：浅蓝色 [0.5, 0.5, 1]
    colors = {
        'high_coop': [0.8, 0, 0],  # 深红色
        'low_coop': [1, 0.5, 0.5],  # 浅红色
        'high_defect': [0, 0, 0.8],  # 深蓝色
        'low_defect': [0.5, 0.5, 1]  # 浅蓝色
    }

    # 填充策略矩阵
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            is_high = reputations[node_id] > theta
            is_coop = strategies[node_id] == 1

            if is_high and is_coop:
                strategy_matrix[i, j] = colors['high_coop']
            elif not is_high and is_coop:
                strategy_matrix[i, j] = colors['low_coop']
            elif is_high and not is_coop:
                strategy_matrix[i, j] = colors['high_defect']
            else:
                strategy_matrix[i, j] = colors['low_defect']

    # 计算各类策略的比例（仍然计算但不显示在图例中）
    high_coop = sum(1 for i in strategies.keys() if strategies[i] == 1 and reputations[i] > theta) / N
    high_defect = sum(1 for i in strategies.keys() if strategies[i] == 0 and reputations[i] > theta) / N
    low_coop = sum(1 for i in strategies.keys() if strategies[i] == 1 and reputations[i] <= theta) / N
    low_defect = sum(1 for i in strategies.keys() if strategies[i] == 0 and reputations[i] <= theta) / N

    # 绘制策略分布图 - 移除标题和图例
    plt.figure(figsize=(8, 8))
    plt.imshow(strategy_matrix, interpolation='nearest')
    # 移除标题
    # plt.title(f'策略分布 (步骤 {step})\nb_L = {b_L}, p = {p_game}, m = {m}', fontsize=14)

    # 移除图例
    # import matplotlib.patches as mpatches
    # red_dark_patch = mpatches.Patch(color=colors['high_coop'], label=f'high_coop ({high_coop:.2%})')
    # red_light_patch = mpatches.Patch(color=colors['low_coop'], label=f'low_coop ({low_coop:.2%})')
    # blue_dark_patch = mpatches.Patch(color=colors['high_defect'], label=f'high_defect ({high_defect:.2%})')
    # blue_light_patch = mpatches.Patch(color=colors['low_defect'], label=f'low_defect ({low_defect:.2%})')
    # plt.legend(handles=[red_dark_patch, red_light_patch, blue_dark_patch, blue_light_patch],
    #            loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=9)

    plt.axis('off')  # 隐藏坐标轴

    # 使用固定的边距，不使用tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # 调整底部边距

    # 保存图像
    plt.savefig(f"results/b=1.1/step_{step}.png", dpi=300, bbox_inches=None)
    plt.close()

    return high_coop, high_defect, low_coop, low_defect


# ---------------------------
# 主程序
# ---------------------------
def main():
    print("开始模拟策略分布演化...")

    # 创建50x50的方格网络
    grid_size = 50
    G = create_grid_network(grid_size, grid_size)

    # 初始化策略和声誉
    strategies = {i: random.choice([0, 1]) for i in G.nodes}
    reputations = {i: random.uniform(0, 2) for i in G.nodes}

    # 记录各类策略比例随时间的变化
    strategy_proportions = {
        'high_coop': [],
        'high_defect': [],
        'low_coop': [],
        'low_defect': [],
        'steps': []
    }

    # 初始阈值
    theta = update_theta(reputations)

    # 记录初始状态
    if 0 in record_steps:
        print("记录初始状态 (步骤 0)")
        high_coop, high_defect, low_coop, low_defect = plot_strategy_distribution(
            strategies, reputations, theta, 0, grid_size)

        strategy_proportions['high_coop'].append(high_coop)
        strategy_proportions['high_defect'].append(high_defect)
        strategy_proportions['low_coop'].append(low_coop)
        strategy_proportions['low_defect'].append(low_defect)
        strategy_proportions['steps'].append(0)

    # 主循环
    for t in range(1, T + 1):
        # 更新自适应阈值
        theta = update_theta(reputations)

        # 计算所有个体的支付
        payoffs = {i: 0 for i in G.nodes}
        for i in G.nodes:
            for j in G.neighbors(i):
                if i < j:  # 避免重复计算
                    pi, pj = play_game(i, j, strategies, reputations, theta, p_game, c, b_L)
                    payoffs[i] += pi
                    payoffs[j] += pj

        # 进行异步策略更新
        num_updates = 50  # 每次更新20个个体
        update_indices = random.sample(list(G.nodes), num_updates)
        for i in update_indices:
            strategies = strategy_update(i, strategies, payoffs, beta, m, reputations, G)

        # 记录特定步骤的策略分布
        if t in record_steps:
            print(f"记录步骤 {t}")
            high_coop, high_defect, low_coop, low_defect = plot_strategy_distribution(
                strategies, reputations, theta, t, grid_size)

            strategy_proportions['high_coop'].append(high_coop)
            strategy_proportions['high_defect'].append(high_defect)
            strategy_proportions['low_coop'].append(low_coop)
            strategy_proportions['low_defect'].append(low_defect)
            strategy_proportions['steps'].append(t)

            # 绘制策略比例随时间的变化
            plt.figure(figsize=(10, 6))
            plt.plot(strategy_proportions['steps'], strategy_proportions['high_coop'], color=[0.8, 0, 0],
                     label='High-Rep Cooperator',
                     linewidth=2)
            plt.plot(strategy_proportions['steps'], strategy_proportions['low_coop'], color=[1, 0.5, 0.5],
                     label='Low-Rep Cooperator',
                     linewidth=2)
            plt.plot(strategy_proportions['steps'], strategy_proportions['high_defect'], color=[0, 0, 0.8],
                     label='High-Rep Defector',
                     linewidth=2)
            plt.plot(strategy_proportions['steps'], strategy_proportions['low_defect'], color=[0.5, 0.5, 1],
                     label='Low-Rep Defector',
                     linewidth=2)

            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Strategy Proportion', fontsize=12)
            plt.title(f'Evolution of Strategy Proportions (b_L = {b_L}, p = {p_game}, m = {m})', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)

            # 修改图例为水平排列
            plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.15))

            # 使用固定的边距
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

            # 修改保存路径
            plt.savefig("results/b=1.1/strategy_proportions.png", dpi=300, bbox_inches=None)
            plt.savefig("results/b=1.1/strategy_proportions.pdf", format='pdf', bbox_inches=None)
            plt.close()

    print("Simulation completed, results saved to results/b=1.1/ directory")


if __name__ == "__main__":
    main()