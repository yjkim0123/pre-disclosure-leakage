#!/usr/bin/env python3
"""Compute daily abnormal volume ratios and regenerate volume figure"""
import pickle, json
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 6),
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Load events
df = pickle.load(open('/Users/yongjun_kim/Documents/project_dart/data/earnings_expanded.pkl', 'rb'))
print(f"Total events: {len(df)}")
print(f"Columns: {df.columns.tolist()[:10]}")

# Get unique companies
companies = df['stock_code'].unique() if 'stock_code' in df.columns else df['corp_code'].unique()
print(f"Unique companies: {len(companies)}")

# Check column names
code_col = 'stock_code' if 'stock_code' in df.columns else 'corp_code'
date_col = 'disc_date' if 'disc_date' in df.columns else 'rcept_dt'
print(f"Using: code={code_col}, date={date_col}")

# Collect volume data per event
vol_ratios = defaultdict(list)  # day -> list of ratios

cached_prices = {}
errors = 0
success = 0

for idx, row in df.iterrows():
    code = str(row[code_col]).zfill(6)
    disc_date = pd.Timestamp(row[date_col])
    
    if code not in cached_prices:
        try:
            start = '2019-06-01'
            end = '2026-04-01'
            p = fdr.DataReader(code, start, end)
            if 'Volume' in p.columns and len(p) > 60:
                cached_prices[code] = p
            else:
                cached_prices[code] = None
        except:
            cached_prices[code] = None
    
    prices = cached_prices[code]
    if prices is None:
        errors += 1
        continue
    
    # Find event day index
    trading_days = prices.index
    event_idx = trading_days.searchsorted(disc_date)
    
    if event_idx < 30 or event_idx + 6 > len(trading_days):
        errors += 1
        continue
    
    # Pre-event volume: [-25, -6]
    pre_vol = prices.iloc[event_idx-25:event_idx-5]['Volume'].mean()
    
    if pre_vol <= 0:
        errors += 1
        continue
    
    # Daily volume ratios [-5, +5]
    for d in range(-5, 6):
        actual_idx = event_idx + d
        if 0 <= actual_idx < len(trading_days):
            day_vol = prices.iloc[actual_idx]['Volume']
            vol_ratios[d].append(day_vol / pre_vol)
    
    success += 1
    if success % 200 == 0:
        print(f"  Processed {success} events...")

print(f"\nSuccess: {success}, Errors: {errors}")

# Compute stats
days = list(range(-5, 6))
means = []
sems = []
pvals = []

from scipy import stats

for d in days:
    vals = np.array(vol_ratios[d])
    vals = vals[~np.isnan(vals) & ~np.isinf(vals)]
    # Winsorize extreme outliers
    vals = vals[(vals > 0.01) & (vals < 50)]
    m = np.mean(vals)
    se = np.std(vals) / np.sqrt(len(vals))
    t_stat = (m - 1.0) / se
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), len(vals)-1))
    means.append(m)
    sems.append(se)
    pvals.append(p_val)
    print(f"Day {d:+d}: mean={m:.3f}x, SE={se:.3f}, t={t_stat:.2f}, p={p_val:.4f}, N={len(vals)}")

# Plot: Line chart with shaded CI
fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(days, 
                [m - 1.96*s for m, s in zip(means, sems)],
                [m + 1.96*s for m, s in zip(means, sems)],
                alpha=0.2, color='steelblue')
ax.plot(days, means, 'o-', color='steelblue', linewidth=2.5, markersize=8, zorder=5)
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Normal Volume (1.0x)')
ax.axvline(x=0, color='red', linestyle=':', alpha=0.5, linewidth=1)

# Add p-value annotations for significant days
for d, m, p in zip(days, means, pvals):
    if p < 0.05:
        ax.annotate(f'p={p:.3f}' if p >= 0.001 else 'p<0.001', 
                   (d, m), textcoords="offset points", xytext=(0, 12),
                   ha='center', fontsize=8, color='darkblue')

ax.set_xlabel('Event Day (0 = Disclosure Date)', fontsize=14)
ax.set_ylabel('Abnormal Volume Ratio', fontsize=14)
ax.set_title(f'Abnormal Trading Volume Around Earnings Disclosures (N={success:,})', fontsize=14)
ax.set_xticks(days)
ax.legend(fontsize=12)
plt.tight_layout()

outdir = '/Users/yongjun_kim/Documents/project_dart/paper/figures'
plt.savefig(f'{outdir}/fig2_volume.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'{outdir}/fig2_volume.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Fig 2 regenerated with real volume data!")
