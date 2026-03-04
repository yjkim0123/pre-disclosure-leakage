#!/usr/bin/env python3
"""
Step 11: Information Leakage Analysis
- Pre-disclosure abnormal volume & returns
- Does volume_ratio predict CAR? (information leakage proxy)
- Event window analysis: CAR[-5,+5] day-by-day
- Disclosure timing effects (time of day, day of week)
- Industry spillover as alternative to chaebol
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from scipy import stats
import statsmodels.api as sm
import json, warnings, time
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
def p(msg): print(msg, flush=True)

# =====================
# 1. 데이터 로드
# =====================
p("=== 1. Loading Data ===")
earnings = pd.read_pickle(f'{DATA_DIR}/earnings_expanded.pkl')
texts = pd.read_pickle(f'{DATA_DIR}/texts_expanded.pkl')

# Load KOSPI index
kospi = fdr.DataReader('^KS11', '2019-12-01', '2026-03-04')
p(f"KOSPI: {len(kospi)} days")

# Merge
merged = earnings.merge(texts[['rcept_no','text','text_len']], on='rcept_no', how='left')
merged['disc_date'] = pd.to_datetime(merged['rcept_dt'])
p(f"Total disclosures: {len(merged)}, Companies: {merged['company_name'].nunique()}")

# Stock code map from earnings
code_map = dict(zip(merged['company_name'], merged['stock_code']))

# Load stock prices
p("\n=== 2. Loading Stock Prices ===")
price_data = {}
for company, code in code_map.items():
    try:
        pr = fdr.DataReader(code, '2019-12-01', '2026-03-04')
        if pr is not None and len(pr) > 100:
            price_data[company] = pr
    except:
        pass
    time.sleep(0.05)
p(f"Loaded: {len(price_data)} companies")

# =====================
# 2. Extended Event Window CAR[-5, +5]
# =====================
p("\n=== 3. Extended Event Window CAR[-5,+5] ===")

def compute_daily_ar(prices, disc_date, ki, day_offset):
    """Compute abnormal return for a single day relative to disclosure"""
    try:
        td = prices.index
        # Find disclosure date position
        disc_pos = td.searchsorted(disc_date)
        target = disc_pos + day_offset
        prev = target - 1
        if target < 0 or target >= len(td) or prev < 0:
            return np.nan
        r = (prices.iloc[target]['Close'] - prices.iloc[prev]['Close']) / prices.iloc[prev]['Close']
        # Market return
        mtd = ki.index
        mdisc = mtd.searchsorted(td[disc_pos] if disc_pos < len(td) else disc_date)
        mt = mdisc + day_offset
        mp = mt - 1
        if mt < 0 or mt >= len(mtd) or mp < 0:
            return r * 100
        mr = (ki.iloc[mt]['Close'] - ki.iloc[mp]['Close']) / ki.iloc[mp]['Close']
        return (r - mr) * 100
    except:
        return np.nan

def compute_abnormal_volume(prices, disc_date, day_offset, baseline_days=20):
    """Compute abnormal volume ratio for a single day"""
    try:
        td = prices.index
        disc_pos = td.searchsorted(disc_date)
        target = disc_pos + day_offset
        if target < 0 or target >= len(td):
            return np.nan
        vol = prices.iloc[target]['Volume']
        # Baseline: [-25, -6] average
        start = max(0, disc_pos - 25)
        end = max(0, disc_pos - 5)
        if end <= start:
            return np.nan
        baseline = prices.iloc[start:end]['Volume'].mean()
        if baseline <= 0:
            return np.nan
        return vol / baseline
    except:
        return np.nan

# Compute AR and AV for each day in [-5, +5]
results = []
for i, (_, row) in enumerate(merged.iterrows()):
    comp = row['company_name']
    dd = row['disc_date']
    if comp not in price_data:
        continue
    pr = price_data[comp]
    
    record = {'rcept_no': row['rcept_no'], 'company': comp, 'disc_date': dd}
    for d in range(-5, 6):
        record[f'ar_{d}'] = compute_daily_ar(pr, dd, kospi, d)
        record[f'av_{d}'] = compute_abnormal_volume(pr, dd, d)
    results.append(record)
    if (i+1) % 500 == 0:
        p(f"  {i+1}/{len(merged)}")

df = pd.DataFrame(results)
p(f"Event window data: {len(df)} events")

# Average AR by day
p("\n--- Average AR by Day ---")
ar_summary = {}
for d in range(-5, 6):
    col = f'ar_{d}'
    vals = df[col].dropna()
    t, pv = stats.ttest_1samp(vals, 0)
    ar_summary[d] = {'mean': vals.mean(), 't': t, 'p': pv, 'n': len(vals)}
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
    p(f"  Day {d:+d}: AR={vals.mean():>7.3f}%, t={t:>5.2f}, p={pv:.4f} {sig}")

# CAR patterns
p("\n--- Cumulative AR ---")
for window_name, (s, e) in [('Pre [-5,-1]', (-5,-1)), ('Event [0,0]', (0,0)), 
                              ('Post [+1,+5]', (1,5)), ('Full [-5,+5]', (-5,5)),
                              ('Pre [-3,-1]', (-3,-1)), ('Tight [0,+1]', (0,1))]:
    cols = [f'ar_{d}' for d in range(s, e+1)]
    car = df[cols].sum(axis=1).dropna()
    t, pv = stats.ttest_1samp(car, 0)
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
    p(f"  CAR{window_name}: {car.mean():>7.3f}%, t={t:>5.2f}, p={pv:.4f} {sig}")

# =====================
# 3. Abnormal Volume Analysis
# =====================
p("\n=== 4. Abnormal Volume ===")
for d in range(-5, 6):
    col = f'av_{d}'
    vals = df[col].dropna()
    # Test if > 1 (normal level)
    t, pv = stats.ttest_1samp(vals, 1.0)
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
    p(f"  Day {d:+d}: AvgVol={vals.mean():>5.2f}x, t={t:>5.2f}, p={pv:.4f} {sig}")

# =====================
# 4. Pre-disclosure Leakage Test
# =====================
p("\n=== 5. Information Leakage Test ===")

# Pre-disclosure CAR[-3,-1] as leakage proxy
df['car_pre'] = df[['ar_-3','ar_-2','ar_-1']].sum(axis=1)
df['car_event'] = df[['ar_0','ar_1']].sum(axis=1)
df['av_pre'] = df[['av_-3','av_-2','av_-1']].mean(axis=1)

# Correlation: pre-period AR vs event AR (same direction = leakage)
valid = df[['car_pre','car_event','av_pre']].dropna()
corr_ar = stats.pearsonr(valid['car_pre'], valid['car_event'])
p(f"Corr(CAR_pre, CAR_event): r={corr_ar[0]:.4f}, p={corr_ar[1]:.4f}")

# Pre volume predicts event CAR direction?
high_vol = valid[valid['av_pre'] > valid['av_pre'].median()]
low_vol = valid[valid['av_pre'] <= valid['av_pre'].median()]
p(f"High pre-vol → Event CAR: {high_vol['car_event'].mean():.3f}%")
p(f"Low pre-vol → Event CAR: {low_vol['car_event'].mean():.3f}%")
t, pv = stats.ttest_ind(high_vol['car_event'], low_vol['car_event'])
p(f"t-test: t={t:.3f}, p={pv:.4f}")

# =====================
# 5. Disclosure Timing Effects
# =====================
p("\n=== 6. Disclosure Timing ===")

# Day of week
df['dow'] = df['disc_date'].dt.dayofweek  # 0=Mon
dow_names = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri'}
p("--- Day of Week ---")
for d in range(5):
    sub = df[df['dow']==d]['car_event'].dropna()
    if len(sub) > 10:
        t, pv = stats.ttest_1samp(sub, 0)
        p(f"  {dow_names[d]}: CAR={sub.mean():>7.3f}%, n={len(sub)}, t={t:.2f}, p={pv:.4f}")

# Month effect
p("--- Quarter ---")
df['quarter'] = df['disc_date'].dt.quarter
for q in [1,2,3,4]:
    sub = df[df['quarter']==q]['car_event'].dropna()
    if len(sub) > 10:
        t, pv = stats.ttest_1samp(sub, 0)
        p(f"  Q{q}: CAR={sub.mean():>7.3f}%, n={len(sub)}, t={t:.2f}, p={pv:.4f}")

# Year trend
p("--- Year ---")
df['year'] = df['disc_date'].dt.year
for y in sorted(df['year'].unique()):
    sub = df[df['year']==y]['car_event'].dropna()
    if len(sub) > 10:
        t, pv = stats.ttest_1samp(sub, 0)
        p(f"  {y}: CAR={sub.mean():>7.3f}%, n={len(sub)}, t={t:.2f}, p={pv:.4f}")

# =====================
# 6. OLS: What predicts CAR?
# =====================
p("\n=== 7. OLS: Predicting Event CAR ===")

# Add features
df['abs_car_pre'] = df['car_pre'].abs()
df['av_pre_log'] = np.log1p(df['av_pre'].clip(0))

reg_df = df[['car_event','car_pre','av_pre_log','abs_car_pre','dow']].dropna()
reg_df['friday'] = (reg_df['dow']==4).astype(int)
reg_df['monday'] = (reg_df['dow']==0).astype(int)

X = sm.add_constant(reg_df[['car_pre','av_pre_log','friday','monday']])
ols = sm.OLS(reg_df['car_event'], X).fit(cov_type='HC1')
p(ols.summary().as_text())

# =====================
# 7. Leakage Score
# =====================
p("\n=== 8. Leakage Score ===")
# Define leakage: pre-period AR same direction as event AR
df['leakage'] = ((df['car_pre'] > 0) & (df['car_event'] > 0)) | ((df['car_pre'] < 0) & (df['car_event'] < 0))
leak_rate = df['leakage'].mean()
# Under random, expected 50%
binom_p = stats.binomtest(int(df['leakage'].sum()), len(df['leakage'].dropna()), 0.5).pvalue if df['leakage'].notna().sum() > 0 else 1.0
p(f"Leakage rate: {leak_rate:.1%} (random=50%)")
p(f"Binomial test: p={binom_p:.6f}")

# By company size (proxy: number of disclosures)
disc_count = df.groupby('company').size()
big = disc_count[disc_count >= disc_count.median()].index
small = disc_count[disc_count < disc_count.median()].index
big_leak = df[df['company'].isin(big)]['leakage'].mean()
small_leak = df[df['company'].isin(small)]['leakage'].mean()
p(f"Large firms leakage: {big_leak:.1%}")
p(f"Small firms leakage: {small_leak:.1%}")

# =====================
# 8. Save Results
# =====================
p("\n=== Saving ===")
df.to_pickle(f'{DATA_DIR}/leakage_analysis.pkl')
df.to_csv(f'{DATA_DIR}/leakage_analysis.csv', index=False)

summary = {
    'n_events': len(df),
    'n_companies': int(df['company'].nunique()),
    'car_pre_mean': round(df['car_pre'].mean(), 4),
    'car_event_mean': round(df['car_event'].mean(), 4),
    'car_pre_event_corr': round(corr_ar[0], 4),
    'car_pre_event_corr_p': round(corr_ar[1], 4),
    'leakage_rate': round(leak_rate, 4),
    'leakage_binom_p': round(binom_p, 6),
    'ar_by_day': {d: round(v['mean'], 4) for d, v in ar_summary.items()},
}
with open(f'{DATA_DIR}/leakage_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

p(f"\n✅ 완료!")
p(json.dumps(summary, indent=2))
