#!/usr/bin/env python3
"""Generate figures for Information Leakage paper"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
FIG_DIR = '/Users/yongjun_kim/Documents/project_dart/paper/figures'
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.grid': True, 'grid.alpha': 0.3,
    'figure.figsize': (7, 4.5)
})

def p(msg): print(msg, flush=True)

# Load data
earnings = pd.read_pickle(f'{DATA_DIR}/earnings_expanded.pkl')
texts = pd.read_pickle(f'{DATA_DIR}/texts_expanded.pkl')
merged = earnings.merge(texts[['rcept_no','text','text_len']], on='rcept_no', how='left')
merged['disc_date'] = pd.to_datetime(merged['rcept_dt'])

import FinanceDataReader as fdr
import time

kospi = fdr.DataReader('^KS11', '2019-12-01', '2026-03-04')
code_map = dict(zip(merged['company_name'], merged['stock_code']))
price_data = {}
for company, code in code_map.items():
    try:
        pr = fdr.DataReader(code, '2019-12-01', '2026-03-04')
        if pr is not None and len(pr) > 100:
            price_data[company] = pr
    except:
        pass
    time.sleep(0.05)
p(f"Loaded {len(price_data)} stocks")

# Compute event window
def compute_daily_ar(prices, disc_date, ki, day_offset):
    try:
        td = prices.index
        disc_pos = td.searchsorted(disc_date)
        target = disc_pos + day_offset
        prev = target - 1
        if target < 0 or target >= len(td) or prev < 0: return np.nan
        r = (prices.iloc[target]['Close'] - prices.iloc[prev]['Close']) / prices.iloc[prev]['Close']
        mtd = ki.index
        mdisc = mtd.searchsorted(td[disc_pos] if disc_pos < len(td) else disc_date)
        mt, mp = mdisc + day_offset, mdisc + day_offset - 1
        if mt < 0 or mt >= len(mtd) or mp < 0: return r*100
        mr = (ki.iloc[mt]['Close'] - ki.iloc[mp]['Close']) / ki.iloc[mp]['Close']
        return (r - mr) * 100
    except: return np.nan

def compute_av(prices, disc_date, day_offset):
    try:
        td = prices.index
        disc_pos = td.searchsorted(disc_date)
        target = disc_pos + day_offset
        if target < 0 or target >= len(td): return np.nan
        vol = prices.iloc[target]['Volume']
        start, end = max(0, disc_pos-25), max(0, disc_pos-5)
        if end <= start: return np.nan
        baseline = prices.iloc[start:end]['Volume'].mean()
        return vol / baseline if baseline > 0 else np.nan
    except: return np.nan

results = []
for i, (_, row) in enumerate(merged.iterrows()):
    comp = row['company_name']
    dd = row['disc_date']
    if comp not in price_data: continue
    pr = price_data[comp]
    rec = {'company': comp, 'disc_date': dd}
    for d in range(-5, 6):
        rec[f'ar_{d}'] = compute_daily_ar(pr, dd, kospi, d)
        rec[f'av_{d}'] = compute_av(pr, dd, d)
    results.append(rec)
df = pd.DataFrame(results)
df['year'] = df['disc_date'].dt.year
p(f"Events: {len(df)}")

# ==========================================
# Figure 1: AR and CAR over event window
# ==========================================
p("Fig 1: AR and CAR")
days = list(range(-5, 6))
ar_means = [df[f'ar_{d}'].mean() for d in days]
ar_ses = [df[f'ar_{d}'].sem() for d in days]
car_means = np.cumsum(ar_means)

fig, ax1 = plt.subplots(figsize=(7, 4.5))
colors = ['#e74c3c' if d < 0 else '#3498db' if d > 0 else '#2ecc71' for d in days]
bars = ax1.bar(days, ar_means, color=colors, alpha=0.7, width=0.6, label='AR (%)', zorder=3)
ax1.errorbar(days, ar_means, yerr=[1.96*s for s in ar_ses], fmt='none', color='black', capsize=3, zorder=4)

ax2 = ax1.twinx()
ax2.plot(days, car_means, 'ko-', linewidth=2, markersize=6, label='CAR (%)', zorder=5)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax1.set_xlabel('Event Day (0 = Disclosure Date)')
ax1.set_ylabel('Abnormal Return, AR (%)')
ax2.set_ylabel('Cumulative AR, CAR (%)')
ax1.set_xticks(days)
ax1.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.6, linewidth=1.5)
ax1.annotate('Pre-disclosure\n(91% of total CAR)', xy=(-3, 0.25), fontsize=9, color='#e74c3c', ha='center', fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=9)
plt.title('Abnormal Returns Around Earnings Disclosures (N=2,221)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig1_ar_car.pdf', bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig1_ar_car.png', bbox_inches='tight')
plt.close()
p("  Done")

# ==========================================
# Figure 2: Abnormal Volume
# ==========================================
p("Fig 2: Abnormal Volume")
av_means = [df[f'av_{d}'].mean() for d in days]
av_ses = [df[f'av_{d}'].sem() for d in days]

fig, ax = plt.subplots(figsize=(7, 4.5))
colors_v = ['#e74c3c' if d < 0 else '#3498db' if d > 0 else '#2ecc71' for d in days]
ax.bar(days, av_means, color=colors_v, alpha=0.7, width=0.6, zorder=3)
ax.errorbar(days, av_means, yerr=[1.96*s for s in av_ses], fmt='none', color='black', capsize=3, zorder=4)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Normal Volume (1.0x)')
ax.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.6, linewidth=1.5)

# Annotate significant pre-disclosure days
for d in [-3, -2, -1]:
    vals = df[f'av_{d}'].dropna()
    t, pv = stats.ttest_1samp(vals, 1.0)
    if pv < 0.05:
        ax.annotate(f'p={pv:.3f}', xy=(d, av_means[d+5]+0.03), fontsize=8, ha='center', color='#e74c3c')

ax.set_xlabel('Event Day (0 = Disclosure Date)')
ax.set_ylabel('Abnormal Volume Ratio')
ax.set_xticks(days)
ax.legend(fontsize=9)
plt.title('Abnormal Trading Volume Around Earnings Disclosures (N=2,221)')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig2_volume.pdf', bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig2_volume.png', bbox_inches='tight')
plt.close()
p("  Done")

# ==========================================
# Figure 3: Year-by-Year CAR[0,+1]
# ==========================================
p("Fig 3: Year trend")
df['car_event'] = df[['ar_0','ar_1']].sum(axis=1)
df['car_pre'] = df[['ar_-3','ar_-2','ar_-1']].sum(axis=1)

year_data = []
for y in sorted(df['year'].unique()):
    sub = df[df['year']==y]
    car_e = sub['car_event'].dropna()
    car_p = sub['car_pre'].dropna()
    year_data.append({
        'year': y, 'n': len(car_e),
        'car_event': car_e.mean(), 'car_event_se': car_e.sem(),
        'car_pre': car_p.mean(), 'car_pre_se': car_p.sem()
    })
yd = pd.DataFrame(year_data)

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(yd))
w = 0.35
ax.bar(x - w/2, yd['car_pre'], w, color='#e74c3c', alpha=0.7, label='CAR[-3,-1] (Pre)')
ax.bar(x + w/2, yd['car_event'], w, color='#3498db', alpha=0.7, label='CAR[0,+1] (Event)')
ax.errorbar(x - w/2, yd['car_pre'], yerr=1.96*yd['car_pre_se'], fmt='none', color='black', capsize=3)
ax.errorbar(x + w/2, yd['car_event'], yerr=1.96*yd['car_event_se'], fmt='none', color='black', capsize=3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(yd['year'].astype(int))
ax.set_xlabel('Year')
ax.set_ylabel('CAR (%)')
ax.legend(fontsize=9)
# Add n labels
for i, row in yd.iterrows():
    ax.annotate(f'n={int(row["n"])}', xy=(i, -1.5), fontsize=7, ha='center', color='gray')
plt.title('Pre-Disclosure vs. Event-Day CARs by Year')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig3_year.pdf', bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig3_year.png', bbox_inches='tight')
plt.close()
p("  Done")

# ==========================================
# Figure 4: Quarter effect
# ==========================================
p("Fig 4: Quarter")
df['quarter'] = df['disc_date'].dt.quarter
fig, ax = plt.subplots(figsize=(5, 4))
q_data = []
for q in [1,2,3,4]:
    sub = df[df['quarter']==q]['car_event'].dropna()
    q_data.append({'q': f'Q{q}', 'mean': sub.mean(), 'se': sub.sem(), 'n': len(sub)})
qd = pd.DataFrame(q_data)
colors_q = ['#e74c3c', '#2ecc71', '#f39c12', '#3498db']
ax.bar(qd['q'], qd['mean'], color=colors_q, alpha=0.7)
ax.errorbar(qd['q'], qd['mean'], yerr=1.96*qd['se'], fmt='none', color='black', capsize=4)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for i, row in qd.iterrows():
    ax.annotate(f'n={int(row["n"])}', xy=(i, row['mean'] + (0.1 if row['mean']>0 else -0.15)), fontsize=9, ha='center')
ax.set_ylabel('CAR[0,+1] (%)')
ax.set_xlabel('Fiscal Quarter')
plt.title('Event-Day CAR by Quarter')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig4_quarter.pdf', bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig4_quarter.png', bbox_inches='tight')
plt.close()
p("  Done")

# ==========================================
# Figure 5: Pre-disclosure CAR distribution
# ==========================================
p("Fig 5: Pre CAR distribution")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

car_pre = df['car_pre'].dropna()
ax1.hist(car_pre, bins=50, color='#e74c3c', alpha=0.7, edgecolor='white')
ax1.axvline(x=car_pre.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean={car_pre.mean():.3f}%')
ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
ax1.set_xlabel('CAR[-3,-1] (%)')
ax1.set_ylabel('Frequency')
ax1.set_title('Pre-Disclosure CAR Distribution')
ax1.legend(fontsize=9)

car_event = df['car_event'].dropna()
ax2.hist(car_event, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
ax2.axvline(x=car_event.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean={car_event.mean():.3f}%')
ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
ax2.set_xlabel('CAR[0,+1] (%)')
ax2.set_ylabel('Frequency')
ax2.set_title('Event-Day CAR Distribution')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig5_distributions.pdf', bbox_inches='tight')
plt.savefig(f'{FIG_DIR}/fig5_distributions.png', bbox_inches='tight')
plt.close()
p("  Done")

p("\n✅ All figures saved to paper/figures/")
