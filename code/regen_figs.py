#!/usr/bin/env python3
"""Regenerate figures with diverse chart types"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 6),
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Load data
with open('/Users/yongjun_kim/Documents/project_dart/data/leakage_market_model.pkl', 'rb') as f:
    df = pickle.load(f)

with open('/Users/yongjun_kim/Documents/project_dart/data/market_model_summary.json', 'r') as f:
    summary = json.load(f)

outdir = '/Users/yongjun_kim/Documents/project_dart/paper/figures'

# ============================================================
# Fig 2: Volume → LINE CHART with shaded area
# ============================================================
days = list(range(-5, 6))
vol_data = []
for d in days:
    col = f'volume_ratio_d{d}'
    if col in df.columns:
        vals = df[col].dropna()
        vol_data.append({
            'day': d,
            'mean': vals.mean(),
            'se': vals.sem(),
        })

vol_days = [v['day'] for v in vol_data]
vol_means = [v['mean'] for v in vol_data]
vol_se = [v['se'] for v in vol_data]

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(vol_days, 
                [m - 1.96*s for m, s in zip(vol_means, vol_se)],
                [m + 1.96*s for m, s in zip(vol_means, vol_se)],
                alpha=0.2, color='steelblue')
ax.plot(vol_days, vol_means, 'o-', color='steelblue', linewidth=2.5, markersize=8, zorder=5)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Normal Volume (1.0x)')
ax.axvline(x=0, color='red', linestyle=':', alpha=0.5, linewidth=1)
ax.set_xlabel('Event Day (0 = Disclosure Date)', fontsize=14)
ax.set_ylabel('Abnormal Volume Ratio', fontsize=14)
ax.set_title('Abnormal Trading Volume Around Earnings Disclosures (N=2,221)', fontsize=14)
ax.set_xticks(days)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f'{outdir}/fig2_volume.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{outdir}/fig2_volume.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 2: Line chart with shaded CI")

# ============================================================
# Fig 3: Year → LINE CHART with markers
# ============================================================
year_data = {}
for _, row in df.iterrows():
    y = int(row.get('disc_date', row.get('filing_date', '2020')).strftime('%Y') if hasattr(row.get('disc_date', row.get('filing_date')), 'strftime') else str(row.get('disc_date', row.get('filing_date')))[:4])
    if y not in year_data:
        year_data[y] = {'car_pre': [], 'car_event': []}
    
    # CAR[-3,-1]
    car_pre = 0
    for d in [-3, -2, -1]:
        col = f'ar_d{d}'
        if col in df.columns and not np.isnan(row.get(col, np.nan)):
            car_pre += row[col]
    year_data[y]['car_pre'].append(car_pre)
    
    # CAR[0,+1]
    car_event = 0
    for d in [0, 1]:
        col = f'ar_d{d}'
        if col in df.columns and not np.isnan(row.get(col, np.nan)):
            car_event += row[col]
    year_data[y]['car_event'].append(car_event)

years = sorted(year_data.keys())
pre_means = [np.mean(year_data[y]['car_pre']) * 100 for y in years]
pre_se = [np.std(year_data[y]['car_pre']) / np.sqrt(len(year_data[y]['car_pre'])) * 100 for y in years]
event_means = [np.mean(year_data[y]['car_event']) * 100 for y in years]
event_se = [np.std(year_data[y]['car_event']) / np.sqrt(len(year_data[y]['car_event'])) * 100 for y in years]
counts = [len(year_data[y]['car_pre']) for y in years]

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(years, pre_means, yerr=[1.96*s for s in pre_se], 
            fmt='s-', color='#e74c3c', linewidth=2, markersize=10, capsize=5, 
            label='CAR[-3,-1] (Pre-disclosure)', zorder=5)
ax.errorbar(years, event_means, yerr=[1.96*s for s in event_se],
            fmt='D-', color='#3498db', linewidth=2, markersize=10, capsize=5,
            label='CAR[0,+1] (Event-day)', zorder=5)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
# Add N labels
for i, (y, n) in enumerate(zip(years, counts)):
    ax.annotate(f'N={n}', (y, min(pre_means[i], event_means[i]) - 0.5),
                ha='center', fontsize=9, color='gray')
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('CAR (%)', fontsize=14)
ax.set_title('Pre-Disclosure and Event-Day CARs by Year', fontsize=14)
ax.legend(fontsize=12)
ax.set_xticks(years)
plt.tight_layout()
plt.savefig(f'{outdir}/fig3_year.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{outdir}/fig3_year.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Fig 3: Line chart with error bars")

# ============================================================
# Fig 4: Quarter → LOLLIPOP CHART (horizontal)
# ============================================================
quarter_data = {}
for _, row in df.iterrows():
    dt = row.get('disc_date', row.get('filing_date'))
    if hasattr(dt, 'month'):
        q = (dt.month - 1) // 3 + 1
    else:
        q = 1
    if q not in quarter_data:
        quarter_data[q] = []
    car_event = 0
    for d in [0, 1]:
        col = f'ar_d{d}'
        if col in df.columns and not np.isnan(row.get(col, np.nan)):
            car_event += row[col]
    quarter_data[q].append(car_event)

quarters = sorted(quarter_data.keys())
q_means = [np.mean(quarter_data[q]) * 100 for q in quarters]
q_se = [np.std(quarter_data[q]) / np.sqrt(len(quarter_data[q])) * 100 for q in quarters]
q_counts = [len(quarter_data[q]) for q in quarters]
q_labels = [f'Q{q}\n(N={n})' for q, n in zip(quarters, q_counts)]
colors = ['#e74c3c' if m < 0 else '#2ecc71' for m in q_means]

fig, ax = plt.subplots(figsize=(8, 5))
ax.hlines(y=range(len(quarters)), xmin=0, xmax=q_means, colors=colors, linewidth=3)
ax.plot(q_means, range(len(quarters)), 'o', color='white', markersize=12, zorder=4)
ax.scatter(q_means, range(len(quarters)), c=colors, s=120, zorder=5, edgecolors='white', linewidth=2)

# Add CI whiskers
for i, (m, se) in enumerate(zip(q_means, q_se)):
    ax.plot([m - 1.96*se, m + 1.96*se], [i, i], color=colors[i], linewidth=1, alpha=0.5)
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
print("✅ Fig 4: Horizontal lollipop chart")

print("\n🎉 All figures regenerated!")
