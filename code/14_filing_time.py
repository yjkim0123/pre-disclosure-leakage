#!/usr/bin/env python3
"""
Filing time analysis - Most DART filings are after market close (800 prefix)
This means Day 0 return is actually PRE-disclosure!
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
from scipy import stats
import json

DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'

def p(msg): print(msg, flush=True)

# Load market model results
df = pd.read_pickle(f'{DATA_DIR}/leakage_market_model.pkl')
earnings = pd.read_pickle(f'{DATA_DIR}/earnings_expanded.pkl')

# Add filing prefix
prefix_map = dict(zip(earnings['rcept_no'].str[:8] + '_' + earnings['company_name'], 
                       earnings['rcept_no'].str[8:11]))
# Better: join by company + date
earnings['disc_date'] = pd.to_datetime(earnings['rcept_dt'])
earnings['prefix3'] = earnings['rcept_no'].str[8:11]
merge_key = earnings[['company_name','disc_date','prefix3']].rename(columns={'company_name':'company'})
df = df.merge(merge_key, on=['company','disc_date'], how='left')

p(f"Total events: {len(df)}")
p(f"With prefix: {df['prefix3'].notna().sum()}")
p(f"Prefix distribution:")
p(str(df['prefix3'].value_counts()))

# ============================================================
# Key insight: For 800-prefix filings (after-hours):
# - Day 0 close is BEFORE the filing → Day 0 return is pre-disclosure
# - Day +1 is the FIRST post-disclosure full trading day
# ============================================================

after_hours = df[df['prefix3'] == '800'].copy()
during_hours = df[df['prefix3'].isin(['900','801','802'])].copy()

p(f"\n=== AFTER-HOURS FILINGS (n={len(after_hours)}) ===")
p("Day 0 return is PRE-disclosure!")

# Redefine windows for after-hours filings
# Pre-disclosure: [-5, 0] (all before public)  
# Post-disclosure: [+1, +5]
p("\n--- Corrected windows (after-hours) ---")
# Pre-disclosure CAR
car_pre_corrected = after_hours[[f'ar_mm_{d}' for d in range(-5, 1)]].sum(axis=1)
t, pv = stats.ttest_1samp(car_pre_corrected.dropna(), 0)
p(f"CAR[-5,0] (true pre-disclosure): {car_pre_corrected.mean():.3f}% (t={t:.2f}, p={pv:.4f})")

car_pre_3 = after_hours[[f'ar_mm_{d}' for d in range(-3, 1)]].sum(axis=1)
t, pv = stats.ttest_1samp(car_pre_3.dropna(), 0)
p(f"CAR[-3,0] (true pre-disclosure): {car_pre_3.mean():.3f}% (t={t:.2f}, p={pv:.4f})")

# Post-disclosure CAR (first available day = +1)
car_post = after_hours[[f'ar_mm_{d}' for d in range(1, 6)]].sum(axis=1)
t, pv = stats.ttest_1samp(car_post.dropna(), 0)
p(f"CAR[+1,+5] (true post-disclosure): {car_post.mean():.3f}% (t={t:.2f}, p={pv:.4f})")

# Day +1 alone (first reaction day)
ar_first = after_hours['ar_mm_1'].dropna()
t, pv = stats.ttest_1samp(ar_first, 0)
p(f"AR[+1] (first reaction): {ar_first.mean():.3f}% (t={t:.2f}, p={pv:.4f})")

# Full window
car_full = after_hours[[f'ar_mm_{d}' for d in range(-5, 6)]].sum(axis=1)
t, pv = stats.ttest_1samp(car_full.dropna(), 0)
p(f"CAR[-5,+5] (full window): {car_full.mean():.3f}% (t={t:.2f}, p={pv:.4f})")

# Percentage pre-disclosure
pct_pre = car_pre_corrected.mean() / car_full.mean() * 100
p(f"\nTrue pre-disclosure share: {pct_pre:.1f}% (was 91% without time correction)")

p(f"\n=== DURING-HOURS FILINGS (n={len(during_hours)}) ===")
if len(during_hours) > 20:
    p("Day 0 includes both pre and post disclosure reaction")
    for w_start, w_end, label in [(-5,-1,'Pre [-5,-1]'), (0,0,'Day 0'), (0,1,'Event [0,+1]'), (1,5,'Post [+1,+5]')]:
        cols = [f'ar_mm_{d}' for d in range(w_start, w_end+1)]
        car = during_hours[cols].sum(axis=1)
        t, pv = stats.ttest_1samp(car.dropna(), 0)
        p(f"  {label}: {car.mean():.3f}% (t={t:.2f}, p={pv:.4f})")

# ============================================================
# Daily AR pattern for after-hours filings
# ============================================================
p(f"\n=== DAILY AR (After-hours only, n={len(after_hours)}) ===")
p(f"{'Day':>5} {'AR(%)':>8} {'t':>8} {'p':>8} {'Note':>20}")
for d in range(-5, 6):
    col = f'ar_mm_{d}'
    vals = after_hours[col].dropna()
    t, pv = stats.ttest_1samp(vals, 0)
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
    note = 'PRE-DISC' if d <= 0 else 'POST-DISC'
    p(f"  {d:+3d}  {vals.mean():>8.3f}  {t:>8.2f}  {pv:>8.4f}  {note:>15} {sig}")

# ============================================================
# Information Leakage Score
# ============================================================
p("\n=== INFORMATION LEAKAGE INDICATORS ===")

# 1. Pre-disclosure direction accuracy
# For after-hours: if CAR[-3,0] correctly predicts direction of eventual CAR[+1,+5]
car_pre_sign = (car_pre_3 > 0).astype(int)
car_post_sign = (car_post > 0).astype(int)
agreement = (car_pre_sign == car_post_sign).mean()
p(f"Direction agreement (pre vs post): {agreement:.3f}")
# If leakage: pre-disc should be correct, post should reverse
# If anticipation: pre and post should agree

# 2. Pre-disclosure proportion with positive AR
pos_pre = (car_pre_corrected > 0).mean()
from scipy.stats import binomtest
bt = binomtest(int((car_pre_corrected > 0).sum()), len(car_pre_corrected), 0.5)
p(f"% positive pre-disclosure CAR: {pos_pre*100:.1f}% (binomial p={bt.pvalue:.4f})")

# 3. Post-disclosure reversal test
# If informed trading: pre goes up, then post should not reverse
# If manipulation: pre goes up, then post reverses
corr_pre_post = car_pre_corrected.corr(car_post)
p(f"Correlation(pre CAR, post CAR): r={corr_pre_post:.3f}")

# 4. Volume-return relationship in pre-disclosure
p("\n=== SAVE SUMMARY ===")
summary = {
    'n_after_hours': int(len(after_hours)),
    'n_during_hours': int(len(during_hours)),
    'pct_after_hours': float(len(after_hours) / len(df) * 100),
    'car_pre_corrected_5_0': float(car_pre_corrected.mean()),
    'car_pre_corrected_3_0': float(car_pre_3.mean()),
    'car_post_1_5': float(car_post.mean()),
    'car_full': float(car_full.mean()),
    'pct_pre_disclosure': float(pct_pre),
    'direction_agreement': float(agreement),
    'pct_positive_pre': float(pos_pre * 100),
    'corr_pre_post': float(corr_pre_post),
}
with open(f'{DATA_DIR}/filing_time_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
p(json.dumps(summary, indent=2))
p("\n✅ Done!")
