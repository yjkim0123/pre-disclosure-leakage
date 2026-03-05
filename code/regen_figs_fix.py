#!/usr/bin/env python3
"""Fix fig3 (year) and fig4 (quarter) — correct column names"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import defaultdict
from scipy import stats

plt.rcParams.update({
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

df = pickle.load(open('/Users/yongjun_kim/Documents/project_dart/data/leakage_market_model.pkl', 'rb'))
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

outdir = '/Users/yongjun_kim/Documents/project_dart/paper/figures'

# ============================================================
# Fig 3: Year → LINE CHART with markers (FIXED column names)
# ============================================================
year_data = defaultdict(lambda: {'car_pre': [], 'car_event': []})

for _, row in df.iterrows():
    y = row['year']
    
    # CAR[-3,-1] using ar_mm_* columns
    car_pre = 0
    for d in [-3, -2, -1]:
        col = f'ar_mm_{d}'
        if col in df.columns:
            v = row[col]
            if not np.isnan(v):
                car_pre += v
    year_data[y]['car_pre'].append(car_pre)  # already in %
    
    # CAR[0,+1]
    car_event = 0
    for d in [0, 1]:
        col = f'ar_mm_{d}'
        if col in df.columns:
            v = row[col]
            if not np.isnan(v):
                car_event += v
    year_data[y]['car_event'].append(car_event)  # already in %

years = sorted(year_data.keys())
pre_means = [np.mean(year_data[y]['car_pre']) for y in years]
pre_se = [stats.sem(year_data[y]['car_pre']) for y in years]
event_means = [np.mean(year_data[y]['car_event']) for y in years]
event_se = [stats.sem(year_data[y]['car_event']) for y in years]
counts = [len(year_data[y]['car_pre']) for y in years]

print("\nYear data:")
for y, pm, em, n in zip(years, pre_means, event_means, counts):
    print(f"  {y}: CAR[-3,-1]={pm:+.3f}%, CAR[0,+1]={em:+.3f}%, N={n}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(years, pre_means, yerr=[1.96*s for s in pre_se], 
            fmt='s-', color='#e74c3c', linewidth=2.5, markersize=10, capsize=5, 
            label='CAR[-3,-1] (Pre-disclosure)', zorder=5)
ax.errorbar(years, event_means, yerr=[1.96*s for s in event_se],
            fmt='D-', color='#3498db', linewidth=2.5, markersize=10, capsize=5,
            label='CAR[0,+1] (Event-day)', zorder=5)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

for i, (y, n) in enumerate(zip(years, counts)):
    ypos = min(pre_means[i], event_means[i]) - 0.3
    ax.annotate(f'N={n}', (y, ypos), ha='center', fontsize=9, color='gray')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('CAR (%)', fontsize=14)
ax.set_title('Pre-Disclosure and Event-Day CARs by Year', fontsize=14)
ax.legend(fontsize=12, loc='best')
ax.set_xticks(years)
plt.tight_layout()
plt.savefig(f'{outdir}/fig3_year.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{outdir}/fig3_year.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 3 fixed!")

# ============================================================
# Fig 4: Quarter → LOLLIPOP CHART (FIXED)
# ============================================================
quarter_data = defaultdict(list)

for _, row in df.iterrows():
    q = row['quarter']
    car_event = 0
    for d in [0, 1]:
        col = f'ar_mm_{d}'
        if col in df.columns:
            v = row[col]
            if not np.isnan(v):
                car_event += v
    quarter_data[q].append(car_event)  # already in %

quarters = sorted(quarter_data.keys())
q_means = [np.mean(quarter_data[q]) for q in quarters]
q_se = [stats.sem(quarter_data[q]) for q in quarters]
q_counts = [len(quarter_data[q]) for q in quarters]
q_labels = [f'Q{q}\n(N={n})' for q, n in zip(quarters, q_counts)]
colors = ['#e74c3c' if m < 0 else '#2ecc71' for m in q_means]

print("\nQuarter data:")
for q, m, n in zip(quarters, q_means, q_counts):
    print(f"  Q{q}: CAR[0,+1]={m:+.3f}%, N={n}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hlines(y=range(len(quarters)), xmin=0, xmax=q_means, colors=colors, linewidth=3)
ax.scatter(q_means, range(len(quarters)), c=colors, s=150, zorder=5, edgecolors='white', linewidth=2)

for i, (m, se) in enumerate(zip(q_means, q_se)):
    ax.plot([m - 1.96*se, m + 1.96*se], [i, i], color=colors[i], linewidth=1.5, alpha=0.5)
    ax.annotate(f'{m:+.2f}%', (m, i), textcoords="offset points", xytext=(15, 0),
                fontsize=11, fontweight='bold', color=colors[i])

ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax.set_yticks(range(len(quarters)))
ax.set_yticklabels(q_labels, fontsize=12)
ax.set_xlabel('Event-Day CAR[0,+1] (%)', fontsize=14)
ax.set_title('Event-Day CARs by Fiscal Quarter', fontsize=14)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{outdir}/fig4_quarter.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{outdir}/fig4_quarter.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 4 fixed!")
