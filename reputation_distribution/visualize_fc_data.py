import pandas as pd
import matplotlib.pyplot as plt
import os

# 创建结果目录（如果不存在）
os.makedirs("results/reputation", exist_ok=True)

# 读取Excel文件
file_path = "results/reputation/4_fc_0.6.xlsx"
df = pd.read_excel(file_path)

# 设置图形大小和风格
plt.figure(figsize=(10, 6))

# 绘制三种不同初始分布的合作比例曲线
plt.plot(df['step'], df['uniform_fc'], label='uniform', linewidth=2)
plt.plot(df['step'], df['gaussian_fc'], label='gaussian', linewidth=2)
plt.plot(df['step'], df['bimodal_fc'], label='bimodal', linewidth=2)

# 设置图表标题和标签
plt.xlabel('step', fontsize=25)
plt.ylabel('$f_c$', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20, width=2, length=7)
# plt.tick_params(axis='both', which='minor', labelsize=14, width=1.5, length=4)
# 将图例位置调整为右侧中间位置
plt.legend(fontsize=18, loc='right', bbox_to_anchor=(1.0, 0.8))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# # # 保存图表为PNG和PDF格式
plt.savefig("results/reputation/fig6a(2).png", dpi=300)
plt.savefig("results/reputation/fig6a(2).pdf", format='pdf')

# 显示图表
plt.show()