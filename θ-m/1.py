import matplotlib.pyplot as plt
import pandas as pd

# 读取数据
data_path = r'd:\桌面\manuscript2\实验\θ-m\results\data\theta_vs_m_WS_data.csv'
df = pd.read_csv(data_path)

# 提取m和各b_L列
m = df['m']
bL_cols = [col for col in df.columns if col.startswith('b_L=')]

# 颜色和标记
colors = ['orange', 'blue', 'red']
markers = ['o', 's', '^']

plt.figure(figsize=(10, 6))
for idx, col in enumerate(bL_cols):
    bL_value = col.split('=')[1]  # 提取数值部分
    plt.plot(m, df[col], marker=markers[idx], color=colors[idx],
             linestyle='-', linewidth=1.5, label=f'WS $b_l={bL_value}$')

plt.xlabel('m', fontsize=20)
plt.ylabel('θ', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True, linestyle='-', alpha=0.7)
plt.legend(fontsize=15, loc='best')
plt.tight_layout()

plt.savefig("results/time_evolution/fig5b.png",dpi=300)
plt.savefig("results/time_evolution/fig5b.pdf", format='pdf')

plt.show()