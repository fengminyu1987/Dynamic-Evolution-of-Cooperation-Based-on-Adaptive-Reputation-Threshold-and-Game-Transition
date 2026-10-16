import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

# ---------------------------
# 参数设置
# ---------------------------
N = 2500  # 网络中个体数目
k = 4  # 每个节点的度数
G = nx.random_regular_graph(k, N)

# 初始策略：0=背叛，1=合作
strategies = {i: random.choice([0, 1]) for i in G.nodes}

# 初始声誉 r_i ∈ [0, 2]（均匀分布）
reputations = {i: random.uniform(0, 2) for i in G.nodes}

# 模型参数
c = 1  # 合作成本
b_L = 1.1  # 低价值博弈基础回报
delta = 0.01  # 声誉变化幅度（合作或背叛后）
p_game = 0.1  # 异质博弈中，一高一低时以 p 选择高价值博弈
theta_prev = sum(reputations.values()) / len(reputations)
beta = 1.0  # 费米规则强度
T = 2000  # 总迭代轮数

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
        game_type = 'high' if random.random() < p_game else 'low'

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
def strategy_update(i, strategies, payoffs, beta, m, reputations):
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
def simulate_for_m(m_values):
    cooperation_rates = []

    for m in m_values:
        # 初始化初始值
        strategies = {i: random.choice([0, 1]) for i in G.nodes}
        reputations = {i: random.uniform(0, 2) for i in G.nodes}
        # theta_prev = 1.0  # 初始阈值

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
                strategies = strategy_update(i, strategies, payoffs, beta, m, reputations)

            # 记录最后500轮的合作比例
            if t >= T - 500:
                coop_rate = sum(strategies.values()) / N
                coop_rate_list.append(coop_rate)

        avg_coop_rate = np.mean(coop_rate_list)
        cooperation_rates.append(avg_coop_rate)
        print(f"m = {m:.2f}: Average Cooperation Rate = {avg_coop_rate:.3f}")

    return cooperation_rates

# ---------------------------
# 主程序
# ---------------------------
m_values = np.linspace(0, 1, 11)  # 设置适应度权重 m 的不同值
cooperation_rates = simulate_for_m(m_values)

# 格式化输出结果 - 所有11个值在一行
print("\n最终合作比例结果：")
formatted_values = [f"{x:.3f}" for x in cooperation_rates]
print("[" + ", ".join(formatted_values) + "]")

# # 绘制结果
# plt.plot(m_values, cooperation_rates, marker='o')
# plt.xlabel('Fitness Weight (m)')
# plt.ylabel('Average Cooperation Rate')
# plt.title('Average Cooperation Rate vs. Fitness Weight (m)')
# plt.grid(True)
# plt.show()