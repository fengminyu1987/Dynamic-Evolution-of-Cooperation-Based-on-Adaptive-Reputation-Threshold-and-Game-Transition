import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import os
import pandas as pd

# ---------------------------
# 参数设置
# ---------------------------
N = 2500  # 网络中个体数目
k = 10  # 每个节点的度数
c = 1  # 合作成本
delta = 0.01  # 声誉变化幅度
beta = 1.0  # 费米规则强度
p_game = 0.9  # 高价值博弈选择概率（固定值）
b_L = 1.5  # 固定的低价值博弈回报
T = 3000  # 总迭代轮数


# ---------------------------
# 创建网络函数
# ---------------------------
# def create_regular_network():
#     print(f"创建规则网络: k={k}, N={N}")
#     return nx.random_regular_graph(k, N)

def create_ws_network():
    print(f"创建WS小世界网络: k={k}, N={N}")
    return nx.watts_strogatz_graph(N, 10, 0.5)


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
            pi = (ri + rj) / 2 - c
            pj = (ri + rj) / 2 - c
        elif si == 1 and sj == 0:
            pi = -c
            pj = rj / 2
        elif si == 0 and sj == 1:
            pi = ri / 2
            pj = -c
        else:
            pi = 0
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

    # 声誉更新
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
# 单次模拟函数 - 返回初始和最终声誉分布
# ---------------------------
def run_simulation(G, b_L, p_game, m, T, init_type="uniform"):
    N = G.number_of_nodes()
    strategies = {i: random.choice([0, 1]) for i in G.nodes}
    # 不同的初始声誉分布
    if init_type == "uniform":
        reputations = {i: random.uniform(0, 2) for i in G.nodes}
    elif init_type == "gaussian":
        reps = np.clip(np.random.normal(1, 0.5, N), 0, 2)
        reputations = {i: reps[i] for i in range(N)}
    elif init_type == "bimodal":
        reps = np.array([0] * int(N / 2) + [2] * int(N / 2))
        np.random.shuffle(reps)
        reputations = {i: reps[i] for i in range(N)}
    else:
        raise ValueError("未知的初始分布类型")
    initial_reputations = list(reputations.values())

    coop_rates = []  # 记录每步合作比例
    theta_values = []  # 记录每步声誉阈值

    for t in range(T):
        theta = update_theta(reputations)
        theta_values.append(theta)
        payoffs = {i: 0 for i in G.nodes}

        # 进行博弈 - 与所有邻居交互
        for i in G.nodes:
            for j in G.neighbors(i):
                if i < j:  # 避免重复计算
                    pi, pj = play_game(i, j, strategies, reputations, theta, p_game, c, b_L)
                    payoffs[i] += pi
                    payoffs[j] += pj

        # 策略更新 - 随机选择部分节点更新
        num_updates = 20
        update_indices = random.sample(list(G.nodes), num_updates)

        for i in update_indices:
            strategies = strategy_update(i, strategies, payoffs, beta, m, reputations, G)

        # 每步记录合作比例
        coop_rate = sum(strategies.values()) / N
        coop_rates.append(coop_rate)

    # 记录最终声誉分布
    final_reputations = list(reputations.values())
    final_coop_rate = sum(strategies.values()) / N

    return initial_reputations, final_reputations, final_coop_rate, coop_rates, theta_values


def plot_coop_and_theta_vs_time():
    T = 3000
    m = 0.5  # 固定m
    init_types = ["uniform", "gaussian", "bimodal"]
    init_labels = ["uniform", "gaussian", "bimodal"]
    G = create_ws_network()

    # 存储每种分布类型的模拟结果
    simulation_results = {}

    # 只运行一次模拟，存储结果
    for idx, init_type in enumerate(init_types):
        print(f"\n初始分布: {init_labels[idx]}")
        _, _, _, coop_rates, theta_values = run_simulation(G, b_L, p_game, m, T, init_type=init_type)
        simulation_results[init_type] = (coop_rates, theta_values)

    # 只保存合作比例到Excel
    df = pd.DataFrame({
        "step": range(T),
        "uniform_fc": simulation_results["uniform"][0],
        "gaussian_fc": simulation_results["gaussian"][0],
        "bimodal_fc": simulation_results["bimodal"][0],
    })
    df.to_excel("results/reputation/test_2.xlsx", index=False)

    # 合作比例图
    plt.figure(figsize=(10, 6))
    for idx, init_type in enumerate(init_types):
        coop_rates, _ = simulation_results[init_type]
        plt.plot(range(T), coop_rates, label=init_labels[idx])
    plt.xlabel('step', fontsize=20)
    plt.ylabel('fc', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16, width=2, length=7)
    plt.tick_params(axis='both', which='minor', labelsize=14, width=1.5, length=4)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.savefig(f"results/reputation/test_1.png", dpi=300)
    # plt.savefig(f"results/reputation/test_1.pdf", format='pdf')
    plt.show()
    print("不同初始声誉分布下合作比例随时间变化曲线已绘制并保存。")

if __name__ == "__main__":
    plot_coop_and_theta_vs_time()