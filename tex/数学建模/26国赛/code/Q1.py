import os
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "Q1结果"); os.makedirs(OUT_DIR, exist_ok=True)
df=pd.read_excel(os.path.join(BASE_DIR, "附件1.xlsx"))
pri=df.iloc[:144, 1].values.astype(float)      # lambda_t (元/kWh)
load=df.iloc[:144, 2].values.astype(float)       # P_L (kW)
pv=df.iloc[:144, 3].values.astype(float)         # P_PV (kW)
T=144
dt=1.0/6.0   # 10分钟 = 1/6 小时
eta=0.9        # 充放电效率
Pmax=5000.0   # 最大充放电功率
Smin,Smax,Sben = 1200.0,10800.0,6000.0   # 储能最小/最大电量与初始电量
# 决策变量定义
n_vars=4*T
c=np.zeros(n_vars)
c[0::4]=pri * dt
bounds=[bnd for t in range(T) for bnd in ((0, None), (0, Pmax), (0, Pmax), (0, pv[t]))]
I = np.eye(n_vars)
rse = np.zeros(n_vars)
rse[1::4]=eta*dt
rse[2::4]=-(1.0/eta)*dt
A_eq = np.vstack([I[0::4] - I[1::4] + I[2::4] - I[3::4], rse])
b_eq = np.append(load - pv, 0.0)
tri = np.triu(np.ones((T, T)))
A_max = np.kron(tri, [0, eta * dt, -(1.0 / eta) * dt, 0])
A_ub = np.empty((2 * T, n_vars))
A_ub[0::2], A_ub[1::2] = A_max, -A_max
b_ub = np.empty(2 * T)
b_ub[0::2], b_ub[1::2] = Smax - Sben, Sben - Smin
# 求解线性规划
res=linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
if not res.success:
    raise RuntimeError(f"优化求解失败: {res.message}")
print("求解成功！目标最优购电费用为:", round(res.fun, 2), "元")
print("全天计划购电量:", round(float(res.x[0::4].sum() * dt), 2), "kWh")
# 解析各变量功率 (kW)
pbuy, pch, pdis, pw = res.x.reshape(T, 4).T
# 能量转换 (kWh)
ebuy,ech,edis=pbuy*dt,pch*dt,pdis*dt
# 储能状态轨迹 (kWh)
soc = np.concatenate([[Sben], Sben + np.cumsum(eta * pch - pdis / eta) * dt])
# 构建"六个功率数据表"
def fmt_time(minutes):
    m = minutes % 1440
    return f"{m // 60}:{m % 60:02d}" + ("+1" if minutes >= 1440 else "")
time_labels = [f"{fmt_time((t + 1) * 10)}-{fmt_time((t + 2) * 10)}" for t in range(T)]
# 表1：指定时间段购电量 + 全天购电量/购电费
print("\n表1  指定时间段购电量及全天购电量和购电费")
for m in [600, 720, 840, 960, 1080, 1200]:   # 10:00 12:00 14:00 16:00 18:00 20:00 起始
    t = m // 10 - 1    
    lbl = f"{fmt_time(m)}-{fmt_time(m + 10)}"
    print(f"  {lbl}   购电量 = {ebuy[t]:.4f} kWh")
print(f"全天购电量 = {ebuy.sum():.4f} kWh")
print(f"  全天购电费 = {res.fun:.2f} 元")
result_path = os.path.join(OUT_DIR, "result1_filled.xlsx")
try:
    wb = openpyxl.load_workbook(result_path)
except Exception:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
# 写入 "计划购电量" 工作表
ws1 = wb["计划购电量"] if "计划购电量" in wb.sheetnames else wb.create_sheet("计划购电量")
ws1.cell(row=1, column=1, value="时间段")
ws1.cell(row=1, column=2, value="购电量")
for t in range(T):
    ws1.cell(row=t + 2, column=1, value=time_labels[t])
    ws1.cell(row=t + 2, column=2, value=round(float(ebuy[t]), 4))
ws1.cell(row=T + 2, column=1, value="全天购电量")
ws1.cell(row=T + 2, column=2, value=round(float(ebuy.sum()), 4))
ws1.cell(row=T + 3, column=1, value="全天购电费")
ws1.cell(row=T + 3, column=2, value=round(float(res.fun), 2))
#写入 "充放电量" 工作表
intervals = [("0:00-4:00", 0, 24),("4:00-8:00", 24, 48),
("8:00-12:00", 48, 72),("12:00-16:00", 72, 96),("16:00-20:00", 96, 120),
("20:00-24:00", 120, 144)]
# 表2：指定时间段充放电量 + 0:00/24:00 储电量
print("\n表2  指定时间段充放电量及 0:00 和 24:00 储电量")
for name, s_, e_ in intervals:
    print(f"  {name:<12} 充电量 = {ech[s_:e_].sum():.4f} kWh   放电量 = {edis[s_:e_].sum():.4f} kWh")
print(f"  0:00 储电量 = {soc[0]:.4f} kWh")
print(f"  24:00 储电量 = {soc[-1]:.4f} kWh")
if "充放电量" in wb.sheetnames:
    wb.remove(wb["充放电量"])
ws2 = wb.create_sheet("充放电量")
ws2.cell(row=1, column=1, value="时间段")
ws2.cell(row=1, column=2, value="充电量")
ws2.cell(row=1, column=3, value="放电量")
ws2.cell(row=1, column=4, value="时刻")
ws2.cell(row=1, column=5, value="储电量")
for i,(name, start, end) in enumerate(intervals):
    r=i+2
    ws2.cell(row=r, column=1, value=name)
    ws2.cell(row=r, column=2, value=round(float(np.sum(ech[start:end])), 4))
    ws2.cell(row=r, column=3, value=round(float(np.sum(edis[start:end])), 4))
    if i==0:
        ws2.cell(row=r, column=4, value="0:00")
        ws2.cell(row=r, column=5, value=round(float(soc[0]), 4))
    elif i==1:
        ws2.cell(row=r, column=4, value="24:00")
        ws2.cell(row=r, column=5, value=round(float(soc[-1]), 4))
try:
    wb.save(result_path)
    print(f"\n结果已保存至: {result_path}")
    print(f"  工作表: {wb.sheetnames}")
except PermissionError:
    wb.save(result_path.replace(".xlsx", "_new.xlsx"))  # 文件被占用时改存副本
# 画图
fig_dir = OUT_DIR
# 全局样式
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Arial'], 'axes.unicode_minus': False,
    'font.size': 14, 'axes.labelsize': 16, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'legend.fontsize': 14, 'axes.spines.top': False, 'axes.linewidth': 1.6,
    'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'xtick.major.size': 6, 'ytick.major.size': 6,})
# 配色
C_BUY,C_PV,C_DIS= '#2E74B5', '#27AE60', '#E67E22'
C_LOAD, C_CH, C_W= '#34495e', '#E74C3C', '#95A5A6'
C_SOC,C_PRICE,C_MAX='#7F8C8D', '#E74C3C', '#C0392B'
C_MIN,C_STORE= '#2980B9', '#E67E22'
# 时间轴
t_mid=(np.arange(T)+0.5)*dt
t_soc=np.arange(T+1)*dt
# 购电去向分配
pv2load = np.minimum(pv, load)
rest = load - pv2load
dis2load = np.minimum(pdis, rest)
rest -= dis2load
buy2load = np.minimum(pbuy, rest)
buy2store = pbuy - buy2load
# 光伏去向分配
pv_rem=pv-pv2load
pv2store=np.minimum(pv_rem, pch)
pv2waste=pv_rem-pv2store
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
# 图1：微网功率平衡
fig,(ax1,ax2)=plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True,
    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.08})
ax1.stackplot(t_mid, pbuy, pv, pdis, labels=['购电', '光伏', '放电'], alpha=0.80, colors=[C_BUY, C_PV, C_DIS])
ax1.set_ylabel('供给功率 (kW)', fontsize=16)
ax1.set_xlim(0, 24)
ax1.set_ylim(0, 16000)
ax1.legend(loc='upper left', ncol=3, fontsize=14, frameon=False)
ax1.grid(True, alpha=0.3, axis='y')
ax1.tick_params(labelsize=13)
ax2.stackplot(t_mid, load, pch, pw,
              labels=['负荷', '充电', '富余功率'], alpha=0.80,
              colors=[C_LOAD, C_CH, C_W])
ax2.set_xlabel('时间 (h)', fontsize=16)
ax2.set_ylabel('需求功率 (kW)', fontsize=16)
ax2.set_xlim(0, 24)
ax2.set_xticks(range(0, 25, 2))
ax2.set_ylim(0, 16000)
ax2.legend(loc='upper left', ncol=3, fontsize=14, frameon=False)
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(labelsize=13)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, '第一问微网功率平衡.png'),dpi=300, bbox_inches='tight')
plt.close()
# 图2：计划购电功率与电价分时对照
fig, ax1 = plt.subplots(figsize=(11.5, 5.2))
ax1.fill_between(t_mid, pbuy, step='mid', alpha=0.25, color=C_BUY)
ax1.step(t_mid, pbuy, where='mid', color=C_BUY, linewidth=2.2, label='购电功率')
ax1.set_xlabel('时间 (h)', fontsize=16)
ax1.set_ylabel('购电功率 (kW)', color=C_BUY, fontsize=16)
ax1.tick_params(axis='x', labelsize=13)
ax1.tick_params(axis='y', labelcolor=C_BUY, labelsize=13)
ax1.set_xlim(0, 24)
ax1.set_xticks(range(0, 25, 2))
ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.spines['right'].set_visible(True)
ax2.step(t_mid, pri, where='mid', color=C_PRICE, linewidth=1.8, alpha=0.75, label='电价')
ax2.set_ylabel('电价 (元/kWh)', color=C_PRICE, fontsize=16)
ax2.tick_params(axis='y', labelcolor=C_PRICE, labelsize=13)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
    loc='upper center', fontsize=14, frameon=False, ncol=2,
    bbox_to_anchor=(0.5, 1.02))
plt.subplots_adjust(left=0.09, right=0.91, top=0.92, bottom=0.16)
plt.savefig(os.path.join(fig_dir, '第一问计划购电功率与电价分时对照.png'),
    dpi=300, bbox_inches='tight')
plt.close()
# 图3：储能充放电
fig, ax1 = plt.subplots(figsize=(11.5, 5.8))
width = dt * 0.7
ax1.bar(t_mid, pch, width=width, color=C_CH, alpha=0.55, label='充电功率')
ax1.bar(t_mid, -pdis, width=width, color=C_PV, alpha=0.55, label='放电功率')
ax1.axhline(0, color='black', linewidth=0.6)
ax1.set_xlabel('时间 (h)', fontsize=16)
ax1.set_ylabel('充/放电功率 (kW)', fontsize=16)
ax1.set_xlim(0, 24)
ax1.set_xticks(range(0, 25, 2))
ax1.set_ylim(-6500, 6500)
ax1.tick_params(labelsize=13)
ax1.grid(True, alpha=0.25, axis='y', linestyle='--')
ax2 = ax1.twinx()
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_linewidth(1.5)
ax2.axhline(Smax, color=C_MAX, linestyle='--', linewidth=1.0, alpha=0.45)
ax2.axhline(Smin, color=C_MIN, linestyle='--', linewidth=1.0, alpha=0.45)
ax2.axhline(Sben, color=C_SOC, linestyle=':', linewidth=0.8, alpha=0.35)
ax2.fill_between(t_soc, soc, Sben, where=soc >= Sben, alpha=0.08, color=C_DIS)
ax2.fill_between(t_soc, soc, Sben, where=soc < Sben, alpha=0.08, color=C_BUY)
ax2.plot(t_soc, soc, color=C_SOC, linewidth=2.8,
    marker='o', markersize=2, label='储电量 $S_t$', zorder=10)
ax2.set_ylabel('储电量 (kWh)', fontsize=16, color=C_SOC)
ax2.tick_params(axis='y', labelcolor=C_SOC, labelsize=13)
ax2.set_ylim(0, 12000)
legend_elements = [Patch(facecolor=C_CH, alpha=0.55, label='充电功率'),
    Patch(facecolor=C_PV, alpha=0.55, label='放电功率'),
    Line2D([0], [0], color=C_SOC, linewidth=2.8, marker='o', markersize=3, label='储电量 $S_t$'),]
ax1.legend(handles=legend_elements, loc='upper center', fontsize=14,
    frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.02))
plt.subplots_adjust(left=0.09, right=0.90, top=0.93, bottom=0.16)
plt.savefig(os.path.join(fig_dir, '第一问储能充放电.png'),
    dpi=300, bbox_inches='tight')
plt.close()
# 图4：计划购电的功率去向
fig, ax = plt.subplots(figsize=(11.5, 5.8))
ax.stackplot(t_mid, buy2load, buy2store,
    labels=['购电 → 直接供给负荷', '购电 → 储能充电'],
    alpha=0.75, colors=[C_BUY, C_STORE])
ax.set_xlabel('时间 (h)', fontsize=16)
ax.set_ylabel('购电功率 (kW)', fontsize=16)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))
ax.set_ylim(0, 12000)
ax.tick_params(labelsize=13)
ax.grid(True, alpha=0.25, axis='y', linestyle='--')
legend_elements = [Patch(facecolor=C_BUY, alpha=0.75, label='购电 → 直接供给负荷'),
    Patch(facecolor=C_STORE, alpha=0.75, label='购电 → 储能充电'),]
ax.legend(handles=legend_elements, loc='upper center', fontsize=14,
    frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.02))
plt.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.16)
plt.savefig(os.path.join(fig_dir, '第一问计划购电的功率去向.png'),
    dpi=300, bbox_inches='tight')
plt.close()
# 图5：光伏发电的功率去向
fig,ax=plt.subplots(figsize=(11.5, 5.8))
ax.stackplot(t_mid, pv2load, pv2store, pv2waste,
    labels=['光伏 → 直接供给负荷', '光伏 → 储能充电', '富余功率'],
    alpha=0.75, colors=[C_PV, C_STORE, C_W])
ax.set_xlabel('时间 (h)', fontsize=16)
ax.set_ylabel('光伏功率 (kW)', fontsize=16)
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))
ax.set_ylim(0, 9000)
ax.tick_params(labelsize=13)
ax.grid(True, alpha=0.25, axis='y', linestyle='--')
legend_elements = [Patch(facecolor=C_PV, alpha=0.75, label='光伏 → 直接供给负荷'),
    Patch(facecolor=C_STORE, alpha=0.75, label='光伏 → 储能充电'),
    Patch(facecolor=C_W, alpha=0.75, label='富余功率'),]
ax.legend(handles=legend_elements, loc='upper center', fontsize=14,
    frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.02))
plt.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.16)
plt.savefig(os.path.join(fig_dir, '第一问光伏发电的功率去向.png'),
    dpi=300, bbox_inches='tight')
plt.close()
print(f'\n问题一图片已保存到: {fig_dir}')