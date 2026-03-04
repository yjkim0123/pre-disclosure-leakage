#!/usr/bin/env python3
"""
Fama-French 3-Factor Model for abnormal returns
Uses Kenneth French data library for Korean market factors
Alternative: compute market-model AR with estimation window
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
from scipy import stats
import FinanceDataReader as fdr
import time
import json
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'

def p(msg): print(msg, flush=True)

# ============================================================
# Step 1: Load data
# ============================================================
p("Loading data...")
earnings = pd.read_pickle(f'{DATA_DIR}/earnings_expanded.pkl')
earnings['disc_date'] = pd.to_datetime(earnings['rcept_dt'])
p(f"  {len(earnings)} disclosures, {earnings['company_name'].nunique()} companies")

# ============================================================
# Step 2: Get market + individual stock data
# ============================================================
p("Loading price data...")
kospi = fdr.DataReader('^KS11', '2019-06-01', '2026-03-04')
kospi['mkt_ret'] = kospi['Close'].pct_change()

# Load individual stocks
code_map = dict(zip(earnings['company_name'], earnings['stock_code']))
price_data = {}
for company, code in code_map.items():
    try:
        pr = fdr.DataReader(code, '2019-06-01', '2026-03-04')
        if pr is not None and len(pr) > 200:
            pr['ret'] = pr['Close'].pct_change()
            price_data[company] = pr
    except:
        pass
    time.sleep(0.05)
p(f"  {len(price_data)} stocks loaded")

# ============================================================
# Step 3: Get Fama-French factors for Korea
# Try pdr or construct SMB/HML from KOSPI data
# ============================================================
p("Constructing Fama-French factors...")
# Since direct FF Korea factors are hard to get,
# we use the market model with estimation window as primary approach
# AND construct pseudo-SMB/HML from KOSPI sub-indices

# Try to get KOSPI size indices for SMB proxy
try:
    kospi_large = fdr.DataReader('KOSPI Large', '2019-06-01', '2026-03-04')
    kospi_small = fdr.DataReader('KOSPI Small', '2019-06-01', '2026-03-04')
    if kospi_large is not None and kospi_small is not None and len(kospi_large) > 200:
        kospi_large['ret'] = kospi_large['Close'].pct_change()
        kospi_small['ret'] = kospi_small['Close'].pct_change()
        # SMB = Small - Large
        smb = kospi_small['ret'] - kospi_large['ret']
        smb = smb.dropna()
        has_smb = True
        p(f"  SMB proxy constructed ({len(smb)} obs)")
    else:
        has_smb = False
except:
    has_smb = False
    p("  KOSPI size indices not available, using market model only")

# Try value index for HML proxy
try:
    kospi_value = fdr.DataReader('KOSPI Value', '2019-06-01', '2026-03-04')
    kospi_growth = fdr.DataReader('KOSPI Growth', '2019-06-01', '2026-03-04')
    if kospi_value is not None and kospi_growth is not None and len(kospi_value) > 200:
        kospi_value['ret'] = kospi_value['Close'].pct_change()
        kospi_growth['ret'] = kospi_growth['Close'].pct_change()
        hml = kospi_value['ret'] - kospi_growth['ret']
        hml = hml.dropna()
        has_hml = True
        p(f"  HML proxy constructed ({len(hml)} obs)")
    else:
        has_hml = False
except:
    has_hml = False

# ============================================================
# Step 4: Market Model with estimation window [-250, -30]
# ============================================================
p("\nComputing Market Model ARs with estimation window...")

def compute_market_model_ar(prices, disc_date, kospi_df, est_start=-250, est_end=-30, event_window=range(-5,6)):
    """Compute ARs using market model estimated over [-250, -30]"""
    try:
        td = prices.index
        disc_pos = td.searchsorted(disc_date)
        if disc_pos >= len(td): return None
        
        # Estimation window
        est_s = max(0, disc_pos + est_start)
        est_e = max(0, disc_pos + est_end)
        if est_e - est_s < 60:  # need at least 60 obs
            return None
        
        est_dates = td[est_s:est_e]
        stock_est = prices.loc[est_dates, 'ret'].dropna()
        mkt_est = kospi_df.reindex(est_dates)['mkt_ret'].dropna()
        
        common = stock_est.index.intersection(mkt_est.index)
        if len(common) < 60:
            return None
        
        y = stock_est.loc[common].values
        x = mkt_est.loc[common].values
        
        # OLS: R_i = alpha + beta * R_m + epsilon
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        # Event window ARs
        result = {}
        for d in event_window:
            target = disc_pos + d
            if target < 0 or target >= len(td):
                result[d] = np.nan
                continue
            event_date = td[target]
            if event_date not in prices.index or event_date not in kospi_df.index:
                result[d] = np.nan
                continue
            r_stock = prices.loc[event_date, 'ret']
            r_mkt = kospi_df.loc[event_date, 'mkt_ret']
            if pd.isna(r_stock) or pd.isna(r_mkt):
                result[d] = np.nan
                continue
            expected = intercept + slope * r_mkt
            result[d] = (r_stock - expected) * 100  # in percent
        
        result['alpha'] = intercept
        result['beta'] = slope
        return result
    except:
        return None

results = []
for i, (_, row) in enumerate(earnings.iterrows()):
    comp = row['company_name']
    dd = row['disc_date']
    if comp not in price_data: continue
    
    ar = compute_market_model_ar(price_data[comp], dd, kospi)
    if ar is None: continue
    
    rec = {'company': comp, 'disc_date': dd, 'alpha': ar['alpha'], 'beta': ar['beta']}
    for d in range(-5, 6):
        rec[f'ar_mm_{d}'] = ar.get(d, np.nan)
    results.append(rec)

df = pd.DataFrame(results)
df['year'] = df['disc_date'].dt.year
df['quarter'] = df['disc_date'].dt.quarter
p(f"Market model ARs computed: {len(df)} events")

# ============================================================
# Step 5: Results comparison
# ============================================================
p("\n=== MARKET MODEL RESULTS ===")
days = list(range(-5, 6))

p("\n--- Daily ARs ---")
for d in days:
    col = f'ar_mm_{d}'
    vals = df[col].dropna()
    t, pv = stats.ttest_1samp(vals, 0)
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
    p(f"  Day {d:+d}: AR={vals.mean():.3f}%, t={t:.2f}, p={pv:.4f} {sig}")

p("\n--- CAR Windows ---")
windows = [(-5,-1), (-3,-1), (0,0), (0,1), (1,5), (-5,5)]
for w_start, w_end in windows:
    cols = [f'ar_mm_{d}' for d in range(w_start, w_end+1)]
    car = df[cols].sum(axis=1)
    t, pv = stats.ttest_1samp(car.dropna(), 0)
    p(f"  CAR[{w_start:+d},{w_end:+d}]: {car.mean():.3f}%, t={t:.2f}, p={pv:.4f}")

# ============================================================
# Step 6: Compare market-adjusted vs market model
# ============================================================
p("\n=== COMPARISON: Market-Adjusted vs Market Model ===")

# Market-adjusted (simple)
results_ma = []
for i, (_, row) in enumerate(earnings.iterrows()):
    comp = row['company_name']
    dd = row['disc_date']
    if comp not in price_data: continue
    pr = price_data[comp]
    td = pr.index
    disc_pos = td.searchsorted(dd)
    
    rec = {'company': comp, 'disc_date': dd}
    for d in range(-5, 6):
        target = disc_pos + d
        if target < 1 or target >= len(td):
            rec[f'ar_ma_{d}'] = np.nan
            continue
        r = pr.iloc[target]['ret']
        event_date = td[target]
        if event_date in kospi.index:
            mr = kospi.loc[event_date, 'mkt_ret']
        else:
            mr = 0
        rec[f'ar_ma_{d}'] = (r - mr) * 100 if not pd.isna(r) else np.nan
    results_ma.append(rec)
df_ma = pd.DataFrame(results_ma)

p(f"Market-adjusted: {len(df_ma)} events, Market model: {len(df)} events")
p("\n{'Window':<15} {'MA CAR':>10} {'MM CAR':>10} {'Diff':>10}")
for w_start, w_end in windows:
    ma_cols = [f'ar_ma_{d}' for d in range(w_start, w_end+1)]
    mm_cols = [f'ar_mm_{d}' for d in range(w_start, w_end+1)]
    ma_car = df_ma[ma_cols].sum(axis=1).mean()
    mm_car = df[mm_cols].sum(axis=1).mean()
    p(f"  CAR[{w_start:+d},{w_end:+d}]:  {ma_car:>8.3f}%  {mm_car:>8.3f}%  {ma_car-mm_car:>8.3f}%")

# ============================================================
# Step 7: Cross-sectional regression
# ============================================================
p("\n=== CROSS-SECTIONAL ANALYSIS ===")
import statsmodels.api as sm

# Market cap proxy: use average volume as size proxy
size_proxy = {}
for comp, pr in price_data.items():
    size_proxy[comp] = pr['Volume'].mean()
df['size_proxy'] = df['company'].map(size_proxy)
df['log_size'] = np.log(df['size_proxy'] + 1)

# Chaebol mapping
chaebol_map = dict(zip(earnings['company_name'], earnings['chaebol']))
df['chaebol'] = df['company'].map(chaebol_map)
df['is_samsung'] = (df['chaebol'] == 'Samsung').astype(int)

# Beta from estimation
df['car_pre'] = df[[f'ar_mm_{d}' for d in range(-3, 0)]].sum(axis=1)
df['car_event'] = df[[f'ar_mm_{d}' for d in range(0, 2)]].sum(axis=1)
df['car_full'] = df[[f'ar_mm_{d}' for d in range(-5, 6)]].sum(axis=1)

# OLS: CAR_pre = f(log_size, beta, year_dummies)
X_vars = ['log_size', 'beta']
X = df[X_vars].dropna()
y = df.loc[X.index, 'car_pre'].dropna()
common_idx = X.index.intersection(y.index)
X = sm.add_constant(X.loc[common_idx])
y = y.loc[common_idx]

model = sm.OLS(y, X).fit(cov_type='HC1')
p("\nOLS: CAR[-3,-1] ~ log_size + beta")
p(model.summary().tables[1].as_text())

# Large vs Small
median_size = df['log_size'].median()
large = df[df['log_size'] >= median_size]['car_pre']
small = df[df['log_size'] < median_size]['car_pre']
t_size, p_size = stats.ttest_ind(large.dropna(), small.dropna())
p(f"\nLarge firms CAR_pre: {large.mean():.3f}%, Small firms: {small.mean():.3f}%, diff p={p_size:.4f}")

# Samsung vs Others
sam = df[df['is_samsung']==1]['car_pre']
oth = df[df['is_samsung']==0]['car_pre']
t_sam, p_sam = stats.ttest_ind(sam.dropna(), oth.dropna())
p(f"Samsung CAR_pre: {sam.mean():.3f}%, Others: {oth.mean():.3f}%, diff p={p_sam:.4f}")

# By chaebol group
p("\nCAR[-3,-1] by chaebol group:")
for g in sorted(df['chaebol'].unique()):
    sub = df[df['chaebol']==g]['car_pre'].dropna()
    if len(sub) < 10: continue
    t, pv = stats.ttest_1samp(sub, 0)
    p(f"  {g:20s}: {sub.mean():+.3f}% (n={len(sub)}, t={t:.2f}, p={pv:.3f})")

# ============================================================
# Step 8: Disclosure time analysis
# ============================================================
p("\n=== DISCLOSURE TIME ANALYSIS ===")
# DART filings have time info in rcept_dt or we can check from texts
# For now, analyze day-of-week patterns
df['dow'] = df['disc_date'].dt.dayofweek  # 0=Mon
dow_names = ['Mon','Tue','Wed','Thu','Fri']
p("\nCAR[-3,-1] by day of week:")
for d in range(5):
    sub = df[df['dow']==d]['car_pre'].dropna()
    if len(sub) < 20:
        continue
    t, pv = stats.ttest_1samp(sub, 0)
    p(f"  {dow_names[d]}: {sub.mean():+.3f}% (n={len(sub)}, t={t:.2f}, p={pv:.3f})")

# ============================================================
# Step 9: Save results
# ============================================================
p("\nSaving results...")
df.to_pickle(f'{DATA_DIR}/leakage_market_model.pkl')

summary = {
    'n_events': len(df),
    'n_companies': df['company'].nunique(),
    'model': 'market_model_[-250,-30]',
    'car_pre_5_1': float(df[[f'ar_mm_{d}' for d in range(-5, 0)]].sum(axis=1).mean()),
    'car_pre_3_1': float(df['car_pre'].mean()),
    'car_event_0_1': float(df['car_event'].mean()),
    'car_full': float(df['car_full'].mean()),
    'avg_beta': float(df['beta'].mean()),
    'avg_alpha': float(df['alpha'].mean()),
}
with open(f'{DATA_DIR}/market_model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

p(f"\n✅ Done! Summary: {json.dumps(summary, indent=2)}")
