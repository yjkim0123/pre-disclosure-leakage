#!/usr/bin/env python3
"""Final fixes: CAR[+1,+1] surprise test, selection bias check, power analysis, clustering"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binomtest
import json, warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
def p(msg): print(msg, flush=True)

df = pd.read_pickle(f'{DATA_DIR}/leakage_with_surprise.pkl')
p(f"Loaded: {len(df)} events")

# Compute needed CARs
df['car_pre3'] = df[[f'ar_mm_{d}' for d in range(-3, 0)]].sum(axis=1)
df['car_pre5'] = df[[f'ar_mm_{d}' for d in range(-5, 0)]].sum(axis=1)
df['car_01'] = df[['ar_mm_0','ar_mm_1']].sum(axis=1)
df['car_day1'] = df['ar_mm_1']  # Day +1 only
df['car_15'] = df[[f'ar_mm_{d}' for d in range(1, 6)]].sum(axis=1)

# ============================================================
# 1. CAR[+1,+1] by surprise direction
# ============================================================
p("\n=== CAR[+1,+1] BY SURPRISE DIRECTION ===")
valid = df[df['surprise_pos'].notna()].copy()
pos = valid[valid['surprise_pos']==1]
neg = valid[valid['surprise_pos']==0]

for metric, label in [('car_day1','CAR[+1,+1]'), ('car_01','CAR[0,+1]'), ('car_15','CAR[+1,+5]')]:
    pm, nm = pos[metric].mean(), neg[metric].mean()
    t, pv = stats.ttest_ind(pos[metric].dropna(), neg[metric].dropna())
    p(f"  {label}: Pos={pm:+.3f}%, Neg={nm:+.3f}%, Diff={pm-nm:+.3f}%, p={pv:.4f}")

# Direction prediction with Day+1 only
valid['day1_correct'] = ((valid['car_day1'] > 0) & (valid['surprise_pos']==1)) | \
                        ((valid['car_day1'] < 0) & (valid['surprise_pos']==0))
acc = valid['day1_correct'].mean()
bt = binomtest(int(valid['day1_correct'].sum()), len(valid), 0.5)
p(f"  Day+1 direction accuracy: {acc*100:.1f}% (p={bt.pvalue:.4f})")

# ============================================================
# 2. Selection bias: parseable vs non-parseable
# ============================================================
p("\n=== SELECTION BIAS CHECK ===")
df['has_surprise'] = df['surprise_pos'].notna().astype(int)
parse = df[df['has_surprise']==1]
noparse = df[df['has_surprise']==0]

p(f"Parseable: {len(parse)}, Non-parseable: {len(noparse)}")
for metric, label in [('car_pre3','CAR[-3,-1]'), ('car_pre5','CAR[-5,-1]'), ('car_01','CAR[0,+1]')]:
    pm, nm = parse[metric].mean(), noparse[metric].mean()
    t, pv = stats.ttest_ind(parse[metric].dropna(), noparse[metric].dropna())
    p(f"  {label}: Parse={pm:+.3f}%, NoParse={nm:+.3f}%, p={pv:.4f}")

# Size and beta comparison
for var, label in [('log_size','Log Size'), ('beta','Beta')]:
    if var in df.columns:
        pm, nm = parse[var].mean(), noparse[var].mean()
        t, pv = stats.ttest_ind(parse[var].dropna(), noparse[var].dropna())
        p(f"  {label}: Parse={pm:.3f}, NoParse={nm:.3f}, p={pv:.4f}")

# Year distribution
p("  Year distribution:")
for y in sorted(df['year'].unique()):
    n_p = (parse['year']==y).sum()
    n_np = (noparse['year']==y).sum()
    p(f"    {y}: Parse={n_p}, NoParse={n_np}")

# ============================================================
# 3. Power analysis
# ============================================================
p("\n=== POWER ANALYSIS ===")
from scipy.stats import norm
n = len(valid)  # 584
alpha = 0.05
z_alpha = norm.ppf(1 - alpha/2)

for true_acc in [0.52, 0.53, 0.55, 0.60]:
    # Power = P(reject H0 | true accuracy = true_acc)
    se = np.sqrt(0.5 * 0.5 / n)  # SE under H0
    se_alt = np.sqrt(true_acc * (1-true_acc) / n)
    z = (true_acc - 0.5) / se
    power = 1 - norm.cdf(z_alpha - (true_acc - 0.5)/se) + norm.cdf(-z_alpha - (true_acc - 0.5)/se)
    p(f"  True accuracy {true_acc*100:.0f}%: power = {power*100:.1f}%")

# ============================================================
# 4. Firm-clustered standard errors
# ============================================================
p("\n=== FIRM-CLUSTERED STANDARD ERRORS ===")
for metric, label in [('car_pre5','CAR[-5,-1]'), ('car_pre3','CAR[-3,-1]'), ('car_01','CAR[0,+1]')]:
    vals = df[[metric, 'company']].dropna()
    y = vals[metric]
    X = sm.add_constant(pd.DataFrame({'const': np.ones(len(y))}))
    
    # Standard
    t_std, p_std = stats.ttest_1samp(y, 0)
    
    # Firm-clustered
    try:
        model = sm.OLS(y.values, np.ones((len(y),1))).fit(cov_type='cluster', 
                cov_kwds={'groups': vals['company'].values})
        t_cl = model.tvalues[0]
        p_cl = model.pvalues[0]
    except:
        t_cl, p_cl = np.nan, np.nan
    
    # Month-clustered
    df_temp = df[[metric, 'disc_date']].dropna()
    df_temp['month'] = df_temp['disc_date'].dt.to_period('M').astype(str)
    try:
        model2 = sm.OLS(df_temp[metric].values, np.ones((len(df_temp),1))).fit(
                 cov_type='cluster', cov_kwds={'groups': df_temp['month'].values})
        t_mo = model2.tvalues[0]
        p_mo = model2.pvalues[0]
    except:
        t_mo, p_mo = np.nan, np.nan
    
    p(f"  {label}: Standard t={t_std:.2f}(p={p_std:.4f}) | Firm-cluster t={t_cl:.2f}(p={p_cl:.4f}) | Month-cluster t={t_mo:.2f}(p={p_mo:.4f})")

# ============================================================
# 5. Volume table
# ============================================================
p("\n=== ABNORMAL VOLUME TABLE ===")
# Need to recompute from original data
earnings = pd.read_pickle(f'{DATA_DIR}/earnings_expanded.pkl')
earnings['disc_date'] = pd.to_datetime(earnings['rcept_dt'])
import FinanceDataReader as fdr
import time

code_map = dict(zip(earnings['company_name'], earnings['stock_code']))
price_data = {}
for comp, code in code_map.items():
    try:
        pr = fdr.DataReader(code, '2019-06-01', '2026-03-04')
        if pr is not None and len(pr) > 200:
            price_data[comp] = pr
    except: pass
    time.sleep(0.02)

def compute_av(prices, disc_date, day_offset):
    try:
        td = prices.index
        dp = td.searchsorted(disc_date)
        target = dp + day_offset
        if target < 0 or target >= len(td): return np.nan
        vol = prices.iloc[target]['Volume']
        s, e = max(0, dp-25), max(0, dp-5)
        if e <= s: return np.nan
        base = prices.iloc[s:e]['Volume'].mean()
        return vol / base if base > 0 else np.nan
    except: return np.nan

vol_results = []
for _, row in earnings.iterrows():
    comp = row['company_name']
    dd = row['disc_date']
    if comp not in price_data: continue
    rec = {}
    for d in range(-5, 6):
        rec[f'av_{d}'] = compute_av(price_data[comp], dd, d)
    vol_results.append(rec)
vdf = pd.DataFrame(vol_results)

p(f"{'Day':>5} {'AV':>8} {'t':>8} {'p':>8}")
for d in range(-5, 6):
    col = f'av_{d}'
    vals = vdf[col].dropna()
    t, pv = stats.ttest_1samp(vals, 1.0)
    sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''
    p(f"  {d:+3d}  {vals.mean():>8.2f}  {t:>8.2f}  {pv:>8.4f} {sig}")

# ============================================================
# 6. Save everything
# ============================================================
p("\nSaving final results...")
final = {
    'day1_pos_surprise': float(pos['car_day1'].mean()),
    'day1_neg_surprise': float(neg['car_day1'].mean()),
    'day1_diff_p': float(stats.ttest_ind(pos['car_day1'].dropna(), neg['car_day1'].dropna())[1]),
    'day1_direction_acc': float(acc),
    'day1_direction_p': float(bt.pvalue),
    'selection_bias_car_pre3_p': float(stats.ttest_ind(parse['car_pre3'].dropna(), noparse['car_pre3'].dropna())[1]),
    'n_parseable': int(len(parse)),
    'n_nonparseable': int(len(noparse)),
}
with open(f'{DATA_DIR}/final_fixes.json', 'w') as f:
    json.dump(final, f, indent=2)
p(json.dumps(final, indent=2))
p("\n✅ All done!")
