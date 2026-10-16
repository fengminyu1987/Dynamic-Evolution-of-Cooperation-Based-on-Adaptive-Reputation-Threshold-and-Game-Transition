import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import os

# 创建结果目录
os.makedirs("results", exist_ok=True)

# ---------------------------
# 参数设置
# ---------------------------
N = 2500
k = 10  # WS网络的度数
p_ws = 0.5  # WS网络的重连概率

# 模型参数
c = 1  # 合作成本
b_L = 1.1  # 低价值博弈基础回报
delta = 0.01  # 声誉变化幅度（合作或背叛后）
beta = 1.0  # 费米规则强度
T = 2000  # 总迭代轮数

# 外层循环参数 - p_game值列表
p_game_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 内层循环参数 - m值列表
m_values = np.linspace(0, 1, 11)  # 设置适应度权重m的不同值

# 存储所有结果
all_results = {}

# ---------------------------
# 自适应阈值更新函数
# ---------------------------
def update_theta(reputations):
    avg_r = np.mean(list(reputations.values()))
    return avg_r

# ---------------------------
# 博弈函数：根据博弈类型计算支付
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
            pi = (ri + rj) / 2 - c # 双方合作：共享声誉平均值
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
# 策略更新函数（费米规则，基于加权适应度）
# ---------------------------
def strategy_update(i, strategies, payoffs, beta, m, reputations, G):
    neighbors = list(G.neighbors(i))
    if not neighbors:
        return strategies

    # 随机选择邻居
    j = np.random.choice(neighbors)

    # 计算个体 i 和 j 的适应度
    fi = (1 - m) * reputations[i] + m * payoffs[i]
    fj = (1 - m) * reputations[j] + m * payoffs[j]

    # 计算策略更新的概率
    prob_update = 1 / (1 + np.exp(-beta * (fj - fi)))
    if random.random() < prob_update:
        strategies[i] = strategies[j]
    return strategies

# ---------------------------
# 进行多次实验，记录合作比例随 m 的变化
# ---------------------------
def simulate_for_m(m_values, p_game, G):
    cooperation_rates = []

    for m in m_values:
        # 初始化初始值
        strategies = {i: random.choice([0, 1]) for i in G.nodes}
        reputations = {i: random.uniform(0, 2) for i in G.nodes}

        coop_rate_list = []

        for t in range(T):
            # 更新自适应阈值
            theta = update_theta(reputations)

            # 计算所有个体的支付（简化版）
            payoffs = {i: 0 for i in G.nodes}
            for i in G.nodes:
                for j in G.neighbors(i):
                    if i < j:  # 避免重复计算
                        pi, pj = play_game(i, j, strategies, reputations, theta, p_game, c, b_L)
                        payoffs[i] += pi
                        payoffs[j] += pj

            # 进行异步策略更新
            num_updates = 20  # 每次更新 20 个个体
            update_indices = random.sample(list(G.nodes), num_updates)
            for i in update_indices:
                strategies = strategy_update(i, strategies, payoffs, beta, m, reputations, G)

            # 记录最后200轮的合作比例
            if t >= T - 200:
                coop_rate = sum(strategies.values()) / N
                coop_rate_list.append(coop_rate)

        avg_coop_rate = np.mean(coop_rate_list)
        cooperation_rates.append(avg_coop_rate)
        print(f"p = {p_game:.1f}, m = {m:.2f}: Average Cooperation Rate = {avg_coop_rate:.3f}")

    return cooperation_rates

# ---------------------------
# 主程序 - 嵌套循环
# ---------------------------
# 外层循环遍历不同的p_game值
for p_game in p_game_values:
    print(f"\n正在运行 p_game = {p_game} 的实验...")
    
    # 创建WS小世界网络
    G = nx.watts_strogatz_graph(N, k=k, p=p_ws)
    
    # 内层循环：对每个p_game值，运行不同的m值
    cooperation_rates = simulate_for_m(m_values, p_game, G)
    
    # 保存结果
    all_results[f"p={p_game}"] = cooperation_rates
    
    # 格式化输出结果 - 所有11个值在一行
    print(f"\np_game = {p_game} 的最终合作比例结果：")
    formatted_values = [f"{x:.3f}" for x in cooperation_rates]
    print("[" + ", ".join(formatted_values) + "]")
    
    # 保存到文件
    with open(f"results/p={p_game}_results.txt", "w") as f:
        f.write("[" + ", ".join(formatted_values) + "]")

# 绘制热图
plt.figure(figsize=(10, 8))
heatmap_data = np.array([all_results[f"p={p}"] for p in p_game_values])
plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
plt.colorbar(label='Average Cooperation Rate')
plt.xlabel('Fitness Weight (m)')
plt.ylabel('Game Type Probability (p)')
plt.title('Cooperation Rate vs. Fitness Weight and Game Type Probability')
plt.xticks(np.arange(len(m_values)), [f"{m:.1f}" for m in m_values])
plt.yticks(np.arange(len(p_game_values)), [f"{p:.1f}" for p in p_game_values])
plt.savefig("results/heatmap.png", dpi=300)
plt.show()