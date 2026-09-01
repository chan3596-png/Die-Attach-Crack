"""
DA Comprehensive Analysis Script
A급 (즉시 가능) + B급 (숨겨진 컬럼) 전체 분석
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

OUT = r'iii_die attatch'
STYLE = {'figure.facecolor': 'white', 'axes.facecolor': 'white'}
plt.rcParams.update(STYLE)

# ── 데이터 로드 ──────────────────────────────────────────────
print("[1/14] Loading data...")
df = pd.read_excel(r'iii_die attatch\20000_BGTTV.xlsx', sheet_name=1)
target = 'DA_Crack_Defect'
df[target] = pd.to_numeric(df[target], errors='coerce').fillna(0).astype(int)
df['Defect_Label'] = df[target].map({0: 'OK', 1: 'NG'})

EQ_ORDER = ['EQ_A', 'EQ_B', 'EQ_C']
PALETTE  = {'OK': '#2ecc71', 'NG': '#e74c3c'}

# ── Helper ───────────────────────────────────────────────────
def save(fig, name):
    fig.savefig(f'{OUT}/{name}', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


# ══════════════════════════════════════════════════════
# A급 ①  Scatter Plot: Bonding Pressure vs Crack (장비별)
# ══════════════════════════════════════════════════════
print("[2/14] Scatter Plot - Pressure vs Crack...")
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
for ax, eq in zip(axes, EQ_ORDER):
    sub = df[df['DA_Equipment'] == eq]
    for label, color in PALETTE.items():
        grp = sub[sub['Defect_Label'] == label]
        ax.scatter(grp['DA_Bonding_Pressure_N'],
                   np.random.uniform(-0.3, 0.3, len(grp)) + (0 if label=='OK' else 1),
                   alpha=0.15, s=8, c=color, label=label)
    ok_mean = sub[sub[target]==0]['DA_Bonding_Pressure_N'].mean()
    ng_mean = sub[sub[target]==1]['DA_Bonding_Pressure_N'].mean()
    ax.axvline(ok_mean, color='#2ecc71', lw=2, ls='--', label=f'OK mean={ok_mean:.1f}N')
    ax.axvline(ng_mean, color='#e74c3c', lw=2, ls='--', label=f'NG mean={ng_mean:.1f}N')
    ax.set_title(f'{eq}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Bonding Pressure (N)')
    ax.set_yticks([0, 1]); ax.set_yticklabels(['OK', 'NG'])
    ax.legend(fontsize=7, loc='upper left')
fig.suptitle('Scatter: Bonding Pressure vs Crack Status by Equipment', fontsize=14, fontweight='bold')
save(fig, 'fig_scatter_pressure_crack.png')

# ══════════════════════════════════════════════════════
# A급 ②③  ROC Curve + AUC  &  Confusion Matrix
# ══════════════════════════════════════════════════════
print("[3/14] ROC Curve + Confusion Matrix...")
# Feature engineering
df_model = df.copy()
df_model['EQ_B'] = (df_model['DA_Equipment'] == 'EQ_B').astype(int)
df_model['EQ_C'] = (df_model['DA_Equipment'] == 'EQ_C').astype(int)

feats = ['EQ_B', 'EQ_C', 'DA_Bonding_Pressure_N', 'DA_Head_Speed_mm_s', 'DA_Placement_Offset_um']
X = df_model[feats].fillna(0)
y = df_model[target]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_tr, y_tr)
gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm.fit(X_tr, y_tr)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC
for model, label, color in [(lr, 'Logistic Regression', '#3498db'),
                              (gbm, 'GBM', '#e74c3c')]:
    fpr, tpr, _ = roc_curve(y_te, model.predict_proba(X_te)[:, 1])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve', fontsize=13, fontweight='bold')
axes[0].legend()

# Confusion Matrix (GBM)
y_pred = gbm.predict(X_te)
cm = confusion_matrix(y_te, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['OK', 'NG'])
disp.plot(ax=axes[1], colorbar=False, cmap='Blues')
axes[1].set_title('Confusion Matrix (GBM)', fontsize=13, fontweight='bold')

# FNR 계산
tn, fp, fn, tp = cm.ravel()
fnr = fn / (fn + tp)
fpr_val = fp / (fp + tn)
axes[1].set_xlabel(f'Predicted  |  FNR(Miss Rate)={fnr:.1%}  |  FPR={fpr_val:.1%}')
save(fig, 'fig_roc_cm.png')

# ══════════════════════════════════════════════════════
# A급 ④  EQ_B 딥다이브
# ══════════════════════════════════════════════════════
print("[4/14] EQ_B Deep Dive...")
params = ['DA_Bonding_Pressure_N', 'DA_Head_Speed_mm_s', 'DA_Placement_Offset_um']
param_labels = ['Bonding Pressure (N)', 'Head Speed (mm/s)', 'Placement Offset (um)']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors_eq = {'EQ_A': '#3498db', 'EQ_B': '#e74c3c', 'EQ_C': '#2ecc71'}
for ax, col, lbl in zip(axes, params, param_labels):
    for eq in EQ_ORDER:
        sub = df[df['DA_Equipment'] == eq][col].dropna()
        ax.hist(sub, bins=40, alpha=0.5, label=eq, color=colors_eq[eq], density=True)
    ax.set_title(lbl, fontsize=12, fontweight='bold')
    ax.set_xlabel(lbl)
    ax.set_ylabel('Density')
    ax.legend()
    # EQ_B 평균 표시
    eq_b_mean = df[df['DA_Equipment']=='EQ_B'][col].mean()
    ax.axvline(eq_b_mean, color='#e74c3c', ls='--', lw=2, label=f'EQ_B mean={eq_b_mean:.1f}')
fig.suptitle('EQ_B Deep Dive: Parameter Distribution vs Other Equipment', fontsize=14, fontweight='bold')
save(fig, 'fig_eqb_deepdive.png')

# ══════════════════════════════════════════════════════
# A급 ⑤  수율 개선 시뮬레이션
# ══════════════════════════════════════════════════════
print("[5/14] Yield Improvement Simulation...")
n_total = len(df)
eq_counts = df['DA_Equipment'].value_counts()

# 현재 불량률
current_rate = df[target].mean() * 100

# 시나리오 1: EQ_B를 EQ_A 수준으로 개선
rate_a = df[df['DA_Equipment']=='EQ_A'][target].mean()
rate_b = df[df['DA_Equipment']=='EQ_B'][target].mean()
rate_c = df[df['DA_Equipment']=='EQ_C'][target].mean()

improved_b = rate_a  # EQ_B → EQ_A 수준
improved_total = (eq_counts.get('EQ_A',0)*rate_a + eq_counts.get('EQ_B',0)*improved_b + eq_counts.get('EQ_C',0)*rate_c) / n_total * 100

# 시나리오 2: EQ_B 압력을 45-55N 범위로 제한
df_b = df[df['DA_Equipment']=='EQ_B'].copy()
df_b_opt = df_b[(df_b['DA_Bonding_Pressure_N'] >= 45) & (df_b['DA_Bonding_Pressure_N'] <= 55)]
rate_b_opt = df_b_opt[target].mean() if len(df_b_opt) > 0 else rate_b * 0.6
improved_total_2 = (eq_counts.get('EQ_A',0)*rate_a + eq_counts.get('EQ_B',0)*rate_b_opt + eq_counts.get('EQ_C',0)*rate_c) / n_total * 100

scenarios = ['Current State', 'Scenario 1:\nEQ_B Hardware Fix\n(EQ_B → EQ_A level)', 'Scenario 2:\nEQ_B Pressure\nOptimize (45-55N)']
values = [current_rate, improved_total, improved_total_2]
bar_colors = ['#e74c3c', '#f39c12', '#2ecc71']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(scenarios, values, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.set_ylabel('Overall Defect Rate (%)', fontsize=12)
ax.set_title('Yield Improvement Simulation\n(What if EQ_B is fixed?)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(values) * 1.25)
ax.axhline(current_rate, color='#e74c3c', ls='--', alpha=0.4)
improvement = current_rate - improved_total
ax.annotate(f'Improvement: -{improvement:.2f}%p', xy=(1, improved_total),
            xytext=(1.5, improved_total + 3),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=11, color='#27ae60', fontweight='bold')
save(fig, 'fig_yield_simulation.png')

# ══════════════════════════════════════════════════════
# A급 ⑥  Calibration Curve
# ══════════════════════════════════════════════════════
print("[6/14] Calibration Curve...")
fig, ax = plt.subplots(figsize=(8, 6))
prob_true, prob_pred = calibration_curve(y_te, lr.predict_proba(X_te)[:, 1], n_bins=10)
ax.plot(prob_pred, prob_true, 'o-', color='#3498db', lw=2, label='Logistic Regression')
prob_true_g, prob_pred_g = calibration_curve(y_te, gbm.predict_proba(X_te)[:, 1], n_bins=10)
ax.plot(prob_pred_g, prob_true_g, 's-', color='#e74c3c', lw=2, label='GBM')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect Calibration')
ax.fill_between([0,1],[0,1],[0,0], alpha=0.05, color='gray')
ax.set_xlabel('Mean Predicted P(Crack)')
ax.set_ylabel('Actual Fraction of Cracks')
ax.set_title('Calibration Curve\n(Predicted Probability vs Actual Rate)', fontsize=13, fontweight='bold')
ax.legend()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
save(fig, 'fig_calibration_curve.png')

# ══════════════════════════════════════════════════════
# A급 ⑦  Run Chart / p-chart
# ══════════════════════════════════════════════════════
print("[7/14] Run Chart (p-chart by LOT order)...")
df_run = df.copy()
df_run['LOT_num'] = df_run['Lot_ID'].str.extract(r'(\d+)').astype(int)
df_run = df_run.sort_values('LOT_num')
window = 200
roll_rate = df_run.groupby('DA_Equipment').apply(
    lambda g: g.set_index('LOT_num')[target].rolling(window, min_periods=50).mean() * 100
).reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
for eq, color in colors_eq.items():
    sub = df_run[df_run['DA_Equipment'] == eq].sort_values('LOT_num')
    roll = sub[target].rolling(window, min_periods=50).mean() * 100
    ax.plot(sub['LOT_num'].values, roll.values, color=color, lw=2, label=eq, alpha=0.85)
ax.axhline(df_run[target].mean()*100, color='black', ls='--', lw=1.5, label=f'Overall mean={df_run[target].mean()*100:.1f}%')
ax.set_xlabel('LOT Sequence (time order →)')
ax.set_ylabel('Rolling Defect Rate (%) [window=200]')
ax.set_title('p-Chart: Defect Rate Drift Over LOT Sequence by Equipment', fontsize=13, fontweight='bold')
ax.legend()
save(fig, 'fig_pchart_lot.png')

# ══════════════════════════════════════════════════════
# B급 ①  Primary Defect Cause 파레토
# ══════════════════════════════════════════════════════
print("[8/14] Primary Defect Cause Pareto...")
cause_counts = df['Primary_Defect_Cause'].value_counts()
cause_pct = cause_counts / len(df) * 100
cum_pct = cause_pct.cumsum()

fig, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.bar(cause_counts.index, cause_pct.values, color=['#e74c3c' if i==0 else '#3498db' for i in range(len(cause_counts))])
ax1.set_ylabel('Defect Rate (%)', fontsize=12)
ax1.set_title('Pareto Chart: Primary Defect Cause Distribution', fontsize=13, fontweight='bold')
for bar, val in zip(bars, cause_pct.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2 = ax1.twinx()
ax2.plot(range(len(cum_pct)), cum_pct.values, 'ko-', lw=2, label='Cumulative %')
ax2.axhline(80, color='gray', ls='--', lw=1, label='80% line')
ax2.set_ylabel('Cumulative %')
ax2.set_ylim(0, 110)
ax2.legend(loc='lower right')
plt.xticks(rotation=20, ha='right')
save(fig, 'fig_defect_cause_pareto.png')

# ══════════════════════════════════════════════════════
# B급 ②  제품 타입 × 장비 9개 조합 히트맵
# ══════════════════════════════════════════════════════
print("[9/14] Product × Equipment 9-combo heatmap...")
pivot = df.groupby(['MOLD_Product_Type', 'DA_Equipment'])[target].mean().unstack() * 100
pivot = pivot[EQ_ORDER]

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', vmin=0, vmax=60,
            linewidths=0.5, ax=ax, annot_kws={'size': 14, 'weight': 'bold'},
            cbar_kws={'label': 'Defect Rate (%)'})
ax.set_title('Defect Rate (%) by Product Type × Equipment (9 Combinations)', fontsize=13, fontweight='bold')
ax.set_xlabel('Equipment (DA_Equipment)')
ax.set_ylabel('Product Type')
save(fig, 'fig_product_eq_heatmap.png')

# ══════════════════════════════════════════════════════
# B급 ③  전공정(BG) 영향도
# ══════════════════════════════════════════════════════
print("[10/14] Upstream BG Process Analysis...")
bg_cols = ['BG_TTV_Before_um', 'BG_TTV_After_um', 'BG_Crack_Count_Before', 'BG_Scratch_Count_Before']
bg_labels = ['BG TTV Before (um)', 'BG TTV After (um)', 'BG Crack Count Before', 'BG Scratch Count Before']
corrs = [df[col].corr(df[target]) for col in bg_cols]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Left: correlation bar
colors_corr = ['#e74c3c' if c > 0.05 else '#95a5a6' for c in corrs]
bars = axes[0].barh(bg_labels, corrs, color=colors_corr)
axes[0].axvline(0, color='black', lw=1)
axes[0].axvline(0.05, color='#e74c3c', ls='--', lw=1.5, label='|r|=0.05 threshold')
axes[0].axvline(-0.05, color='#e74c3c', ls='--', lw=1.5)
for bar, val in zip(bars, corrs):
    axes[0].text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=10)
axes[0].set_title('Upstream BG Process\nCorrelation with DA Crack', fontsize=12, fontweight='bold')
axes[0].legend()

# Right: BG TTV After boxplot OK vs NG
sns.boxplot(x='Defect_Label', y='BG_TTV_After_um', data=df, ax=axes[1],
            order=['OK', 'NG'], palette=PALETTE, width=0.4)
axes[1].set_title('BG TTV After vs DA Crack Status', fontsize=12, fontweight='bold')
axes[1].set_xlabel('DA Crack Defect')
axes[1].set_ylabel('BG TTV After (um)')
save(fig, 'fig_bg_upstream_analysis.png')

# ══════════════════════════════════════════════════════
# B급 ④  Cure Time 영향 분석
# ══════════════════════════════════════════════════════
print("[11/14] MOLD Cure Time Analysis...")
df['Cure_Time_Group'] = pd.qcut(df['MOLD_Cure_Time_sec'], q=3, labels=['Short', 'Medium', 'Long'])
cure_rate = df.groupby('Cure_Time_Group')[target].mean() * 100
cure_corr = df['MOLD_Cure_Time_sec'].corr(df[target])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bars = axes[0].bar(cure_rate.index, cure_rate.values, color=['#3498db','#f39c12','#e74c3c'], width=0.5)
for bar, val in zip(bars, cure_rate.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[0].set_title(f'Defect Rate by Cure Time Group\n(r with Crack = {cure_corr:.4f})', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Defect Rate (%)')
axes[0].set_xlabel('Cure Time Group')

sns.boxplot(x='Defect_Label', y='MOLD_Cure_Time_sec', data=df, ax=axes[1],
            order=['OK','NG'], palette=PALETTE, width=0.4)
axes[1].set_title('Cure Time vs Crack Status', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Crack Status')
axes[1].set_ylabel('Cure Time (sec)')
save(fig, 'fig_cure_time_analysis.png')

# ══════════════════════════════════════════════════════
# B급 ⑤  SAW 공정 이상 확인
# ══════════════════════════════════════════════════════
print("[12/14] SAW Process Analysis...")
saw_corr = df['SAW_Blade_Wear_pct'].corr(df[target])
spc_rate = df.groupby('SAW_SPC_Status')[target].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Blade wear scatter
axes[0].scatter(df['SAW_Blade_Wear_pct'], df[target] + np.random.uniform(-0.2, 0.2, len(df)),
                alpha=0.05, s=5, c='#3498db')
axes[0].set_title(f'SAW Blade Wear vs Crack\n(r = {saw_corr:.4f})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('SAW Blade Wear (%)')
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(['OK', 'NG'])

# SPC status
bars = axes[1].bar(spc_rate.index, spc_rate.values, color=['#2ecc71','#e74c3c'], width=0.4)
for bar, val in zip(bars, spc_rate.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[1].set_title('Defect Rate by SAW SPC Status', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Defect Rate (%)')
save(fig, 'fig_saw_analysis.png')

# ══════════════════════════════════════════════════════
# 추가 ①  EQ별 Head Speed 전략 비교
# ══════════════════════════════════════════════════════
print("[13/14] Head Speed stratified analysis...")
fig, ax = plt.subplots(figsize=(9, 5))
speed_corrs = df.groupby('DA_Equipment').apply(
    lambda g: g['DA_Head_Speed_mm_s'].corr(g[target])
).reset_index()
speed_corrs.columns = ['DA_Equipment', 'corr']
overall_corr = df['DA_Head_Speed_mm_s'].corr(df[target])
speed_corrs_all = pd.concat([
    pd.DataFrame({'DA_Equipment': ['Overall'], 'corr': [overall_corr]}),
    speed_corrs
])
colors_bar = ['#95a5a6'] + ['#3498db' if v > 0 else '#e74c3c' for v in speed_corrs['corr']]
bars = ax.bar(speed_corrs_all['DA_Equipment'], speed_corrs_all['corr'], color=colors_bar, width=0.5)
ax.axhline(0, color='black', lw=1)
for bar, val in zip(bars, speed_corrs_all['corr']):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.002 if val >= 0 else -0.008),
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Head Speed vs Crack: Correlation by Equipment\n(Stratification Test)', fontsize=13, fontweight='bold')
ax.set_ylabel('Correlation with Crack Defect (r)')
save(fig, 'fig_headspeed_stratified.png')

# ══════════════════════════════════════════════════════
# 추가 ②  최종 분석 Summary 표
# ══════════════════════════════════════════════════════
print("[14/14] Summary Table...")
summary = {
    'Variable': ['Placement Offset', 'Bonding Pressure', 'Head Speed',
                 'DA Equipment (EQ_B)', 'BG TTV After', 'Cure Time', 'SAW Blade Wear'],
    'Overall r': [
        df['DA_Placement_Offset_um'].corr(df[target]),
        df['DA_Bonding_Pressure_N'].corr(df[target]),
        df['DA_Head_Speed_mm_s'].corr(df[target]),
        (df['DA_Equipment']=='EQ_B').astype(int).corr(df[target]),
        df['BG_TTV_After_um'].corr(df[target]),
        df['MOLD_Cure_Time_sec'].corr(df[target]),
        df['SAW_Blade_Wear_pct'].corr(df[target]),
    ],
    'Verdict': [
        'CONFOUND - Proxy Variable',
        'ROOT CAUSE (Causal)',
        'Weak Negative',
        'ROOT CAUSE (Dominant)',
        'Negligible',
        'Negligible',
        'Negligible',
    ]
}
df_summary = pd.DataFrame(summary)
df_summary['Overall r'] = df_summary['Overall r'].round(4)

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis('off')
tbl = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
tbl.scale(1, 2.2)
# Color rows by verdict
row_colors = {'CONFOUND - Proxy Variable': '#fadbd8',
              'ROOT CAUSE (Causal)': '#d5f5e3',
              'ROOT CAUSE (Dominant)': '#d5f5e3',
              'Weak Negative': '#fef9e7',
              'Negligible': '#f2f3f4'}
for i, v in enumerate(summary['Verdict']):
    for j in range(3):
        tbl[i+1, j].set_facecolor(row_colors.get(v, 'white'))
for j in range(3):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title('Comprehensive Variable Analysis Summary', fontsize=14, fontweight='bold', y=0.95)
save(fig, 'fig_final_summary_table.png')

print("\n✅ All 14 analyses completed. Files saved in:", OUT)
