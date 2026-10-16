import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import os

# 创建结果目录
os.makedirs("results/time_evolution", exist_ok=True)

# ---------------------------
# 参数设置
# ---------------------------
N = 2500  # 网络中个体数目
k = 4  # 每个节点的度数

# 模型参数
c = 1  # 合作成本
delta = 0.01  # 声誉变化幅度
p_game = 0.9  # 高价值博弈选择概率（固定值）
beta = 1.0  # 费米规则强度
T = 2000  # 总迭代轮数


# ---------------------------
# 创建网络函数
# ---------------------------
def create_regular_network():
    return nx.random_regular_graph(k, N)


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
# 单次模拟函数 - 返回最终的声誉阈值
# ---------------------------
def run_simulation_for_theta(G, b_L, p_game, m, T=2000):
    # 初始化
    strategies = {i: random.choice([0, 1]) for i in G.nodes}
    reputations = {i: random.uniform(0, 2) for i in G.nodes}
    theta_values = []

    for t in range(T):
        theta = update_theta(reputations)
        payoffs = {i: 0 for i in G.nodes}

        # 进行博弈
        for i in G.nodes:
            for j in G.neighbors(i):
                if i < j:
                    pi, pj = play_game(i, j, strategies, reputations, theta, p_game, c, b_L)
                    payoffs[i] += pi
                    payoffs[j] += pj

        # 策略更新
        num_updates = 20
        update_indices = random.sample(list(G.nodes), num_updates)
        for i in update_indices:
            strategies = strategy_update(i, strategies, payoffs, beta, m, reputations, G)

        # 记录当前声誉阈值
        if t >= T - 500:  # 只记录最后500步
            theta_values.append(theta)

    # 返回最后500步的平均声誉阈值
    return np.mean(theta_values)


# ---------------------------
# 主程序：绘制声誉阈值随m的变化（格子网络）
# ---------------------------
def plot_theta_vs_m_HL():
    # 设置b_L的取值
    bL_values = [1.1, 1.5, 2.0]  # 可以根据需要添加更多值
    
    # 设置m的取值范围
    m_values = np.linspace(0.0, 1.0, 11)  # 从0到1，共11个点
    
    # 创建格子网络
    G_regular = create_regular_network()
    
    # 存储不同m值下的声誉阈值
    theta_regular = {b_L: [] for b_L in bL_values}
    
    # 设置图表
    plt.figure(figsize=(10, 6))
    
    # 设置颜色和标记
    colors = ['green', 'orange', 'blue']
    markers = ['o', 's', '^']
    
    # 对每个b_L值和m值进行模拟
    for b_idx, b_L in enumerate(bL_values):
        print(f"\n模拟 b_L = {b_L:.2f}")
        
        for m_val in m_values:
            print(f"  处理 m = {m_val:.2f}")
            
            # 格子网络
            print("    处理格子网络")
            theta_reg = run_simulation_for_theta(G_regular, b_L, p_game, m_val, T)
            theta_regular[b_L].append(theta_reg)
    
    # 保存数据到CSV文件
    os.makedirs("results/data", exist_ok=True)
    data_file = "results/data/theta_vs_m_SL_data.csv"
    
    with open(data_file, 'w') as f:
        # 写入表头
        f.write("m")
        for b_L in bL_values:
            f.write(f",b_L={b_L}")
        f.write("\n")
        
        # 写入数据
        for i, m_val in enumerate(m_values):
            f.write(f"{m_val}")
            for b_L in bL_values:
                f.write(f",{theta_regular[b_L][i]}")
            f.write("\n")
    
    print(f"数据已保存到 {data_file}")
    
    # 绘制结果
    for b_idx, b_L in enumerate(bL_values):
        # 格子网络结果
        plt.plot(m_values, theta_regular[b_L], marker=markers[b_idx], color=colors[b_idx], 
                 linestyle='-', linewidth=1.5, label=f'HL b={b_L}')
    
    # 设置图表属性
    plt.xlabel('m', fontsize=14)
    plt.ylabel('θ', fontsize=14)
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.legend(fontsize=10, loc='best')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"results/time_evolution/theta_vs_m_SL.png", dpi=300)
    plt.savefig(f"results/time_evolution/theta_vs_m_SL.pdf", format='pdf')
    plt.show()
    
    print("模拟完成，结果已保存")


# 运行模拟
if __name__ == "__main__":
    plot_theta_vs_m_HL()  # 格子网络下声誉阈值随m的变化