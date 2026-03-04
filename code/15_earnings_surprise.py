#!/usr/bin/env python3
"""
Earnings Surprise Analysis:
1. Compute earnings surprise (YoY change in operating profit from disclosure text)
2. Test if pre-disclosure drift predicts surprise direction
3. Clustered standard errors
4. Non-parametric tests (sign test, Wilcoxon)
5. Standardized abnormal returns
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binomtest, wilcoxon, mannwhitneyu
import json, re, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'

def p(msg): print(msg, flush=True)

# ============================================================
# Step 1: Load data
# ============================================================
p("Loading data...")
mm = pd.read_pickle(f'{DATA_DIR}/leakage_market_model.pkl')
earnings = pd.read_pickle(f'{DATA_DIR}/earnings_expanded.pkl')
texts = pd.read_pickle(f'{DATA_DIR}/texts_expanded.pkl')

# Merge texts
earnings['disc_date'] = pd.to_datetime(earnings['rcept_dt'])
merged = earnings.merge(texts[['rcept_no','text','text_len']], on='rcept_no', how='left')
p(f"  Merged: {len(merged)} rows, {merged['text'].notna().sum()} with text")

# ============================================================
# Step 2: Parse earnings surprise from disclosure text
# ============================================================
p("\nParsing earnings surprise from texts...")

def parse_yoy_profit(text):
    """Extract YoY operating profit change from DART disclosure text"""
    if pd.isna(text) or not isinstance(text, str):
        return None
    
    # Look for 영업이익 (operating profit) patterns
    # Common pattern: 당기 vs 전기 comparison
    # Also: 증감률, 증감비율, YoY
    
    # Try to find operating profit values
    # Pattern 1: 영업이익 ... 당기 XXX ... 전기 XXX
    # Pattern 2: 증감률 or 증감비율 XX%
    
    # Extract numbers near 영업이익
    profit_pattern = r'영업이익[^\n]*?([0-9,.]+)'
    matches = re.findall(profit_pattern, text)
    
    # Try 증감률 pattern
    change_pattern = r'증감[률비율]*[^0-9]*?([+-]?[0-9,.]+)\s*%'
    change_matches = re.findall(change_pattern, text)
    
    # Try to find two consecutive profit numbers (당기, 전기)
    profit_section = re.search(r'영업이익.*?(?:당기|금기).*?([0-9,.]+).*?(?:전기|전년).*?([0-9,.]+)', text, re.DOTALL)
    
    if profit_section:
        try:
            current = float(profit_section.group(1).replace(',', ''))
            previous = float(profit_section.group(2).replace(',', ''))
            if previous != 0:
                return (current - previous) / abs(previous) * 100
        except:
            pass
    
    if change_matches:
        try:
            return float(change_matches[0].replace(',', ''))
        except:
            pass
    
    return None

# Parse from dataset_final which has parsed metrics
try:
    ds = pd.read_pickle(f'{DATA_DIR}/dataset_final_v5.pkl')
    p(f"  dataset_final_v5 loaded: {len(ds)} rows")
    p(f"  Columns: {ds.columns.tolist()}")
    has_ds = True
except:
    has_ds = False
    p("  dataset_final_v5 not found")

if has_ds:
    # Check what metrics are available
    metric_cols = [c for c in ds.columns if any(x in c.lower() for x in ['profit', 'revenue', 'yoy', 'surprise', 'change'])]
    p(f"  Metric columns: {metric_cols}")
    
    # Use op_profit_yoy if available
    if 'op_profit_yoy' in ds.columns:
        p(f"  op_profit_yoy available: {ds['op_profit_yoy'].notna().sum()} non-null")
        p(f"  Sample: {ds['op_profit_yoy'].dropna().head(10).tolist()}")

# ============================================================
# Step 3: Construct surprise from YoY operating profit change
# ============================================================
p("\nConstructing earnings surprise...")

if has_ds and 'op_profit_yoy' in ds.columns:
    # Merge surprise into market model results
    # Need to match by company + date
    ds['disc_date'] = pd.to_datetime(ds['rcept_dt']) if 'rcept_dt' in ds.columns else pd.to_datetime(ds['disc_date']) if 'disc_date' in ds.columns else None
    
    if 'company_name' in ds.columns:
        surprise_map = ds[['company_name', 'disc_date', 'op_profit_yoy']].rename(columns={'company_name': 'company'})
    elif 'company' in ds.columns:
        surprise_map = ds[['company', 'disc_date', 'op_profit_yoy']]
    else:
        p("  Can't find company column in dataset_final")
        surprise_map = None
    
    if surprise_map is not None:
        mm2 = mm.merge(surprise_map, on=['company', 'disc_date'], how='left')
        p(f"  Matched: {mm2['op_profit_yoy'].notna().sum()} / {len(mm2)}")
        
        # Define surprise: positive = YoY increase, negative = YoY decrease
        mm2['surprise_pos'] = (mm2['op_profit_yoy'] > 0).astype(float)
        mm2.loc[mm2['op_profit_yoy'].isna(), 'surprise_pos'] = np.nan
        
        n_pos = (mm2['surprise_pos'] == 1).sum()
        n_neg = (mm2['surprise_pos'] == 0).sum()
        n_na = mm2['surprise_pos'].isna().sum()
        p(f"  Positive surprise: {n_pos}, Negative: {n_neg}, NA: {n_na}")
else:
    # Fallback: parse from text
    p("  Using text parsing fallback...")
    merged['yoy_change'] = merged['text'].apply(parse_yoy_profit)
    p(f"  Parsed YoY from text: {merged['yoy_change'].notna().sum()} / {len(merged)}")
    
    surprise_map = merged[['company_name', 'disc_date', 'yoy_change']].rename(
        columns={'company_name': 'company', 'yoy_change': 'op_profit_yoy'})
    mm2 = mm.merge(surprise_map, on=['company', 'disc_date'], how='left')
    mm2['surprise_pos'] = (mm2['op_profit_yoy'] > 0).astype(float)
    mm2.loc[mm2['op_profit_yoy'].isna(), 'surprise_pos'] = np.nan

# ============================================================
# Step 4: Test if pre-disclosure drift predicts surprise direction
# ============================================================
p("\n=== EARNINGS SURPRISE ANALYSIS ===")

# CARs
mm2['car_pre'] = mm2[[f'ar_mm_{d}' for d in range(-3, 0)]].sum(axis=1)
mm2['car_pre5'] = mm2[[f'ar_mm_{d}' for d in range(-5, 0)]].sum(axis=1)
mm2['car_event'] = mm2[[f'ar_mm_{d}' for d in range(0, 2)]].sum(axis=1)
mm2['car_post'] = mm2[[f'ar_mm_{d}' for d in range(1, 6)]].sum(axis=1)

valid = mm2[mm2['surprise_pos'].notna()].copy()
p(f"\nValid events with surprise: {len(valid)}")

if len(valid) > 50:
    pos = valid[valid['surprise_pos'] == 1]
    neg = valid[valid['surprise_pos'] == 0]
    
    p(f"\n--- By Surprise Direction ---")
    p(f"{'Metric':<25} {'Pos Surprise':>15} {'Neg Surprise':>15} {'Diff':>10} {'p-value':>10}")
    
    for metric, label in [('car_pre', 'CAR[-3,-1]'), ('car_pre5', 'CAR[-5,-1]'), 
                          ('car_event', 'CAR[0,+1]'), ('car_post', 'CAR[+1,+5]')]:
        pos_m = pos[metric].mean()
        neg_m = neg[metric].mean()
        t, pv = stats.ttest_ind(pos[metric].dropna(), neg[metric].dropna())
        p(f"  {label:<23} {pos_m:>+13.3f}% {neg_m:>+13.3f}% {pos_m-neg_m:>+8.3f}% {pv:>8.4f}")
    
    # KEY TEST: Does pre-disclosure drift predict surprise direction?
    # If leakage: positive surprise → positive pre-drift, negative → negative
    p(f"\n--- Direction Prediction Test ---")
    valid['pre_correct'] = ((valid['car_pre'] > 0) & (valid['surprise_pos'] == 1)) | \
                           ((valid['car_pre'] < 0) & (valid['surprise_pos'] == 0))
    accuracy = valid['pre_correct'].mean()
    bt = binomtest(int(valid['pre_correct'].sum()), len(valid), 0.5)
    p(f"  Pre-disclosure drift predicts surprise direction: {accuracy*100:.1f}% (binomial p={bt.pvalue:.4f})")
    p(f"  If p<0.05, evidence of informed trading / leakage")
    p(f"  If p>0.05, drift may be noise / anticipation")
    
    # Correlation between surprise magnitude and pre-disclosure drift
    corr = valid['op_profit_yoy'].corr(valid['car_pre'])
    p(f"\n  Correlation(YoY profit change, CAR[-3,-1]): r={corr:.4f}")
    
    # Rank correlation (more robust)
    rho, rho_p = stats.spearmanr(valid['op_profit_yoy'].dropna(), valid.loc[valid['op_profit_yoy'].notna(), 'car_pre'])
    p(f"  Spearman rho: {rho:.4f}, p={rho_p:.4f}")

# ============================================================
# Step 5: Clustered Standard Errors
# ============================================================
p("\n=== CLUSTERED STANDARD ERRORS ===")
import statsmodels.api as sm

# Cluster by month (to capture same-month reporting)
mm2['month_cluster'] = mm2['disc_date'].dt.to_period('M').astype(str)
mm2['chaebol'] = mm2['company'].map(dict(zip(earnings['company_name'], earnings['chaebol'])))

for metric, label in [('car_pre5', 'CAR[-5,-1]'), ('car_pre', 'CAR[-3,-1]'), ('car_event', 'CAR[0,+1]')]:
    vals = mm2[metric].dropna()
    
    # Standard t-test
    t_std, p_std = stats.ttest_1samp(vals, 0)
    
    # Clustered by month
    y = mm2[[metric, 'month_cluster']].dropna()
    X = sm.add_constant(np.ones(len(y)))
    try:
        model = sm.OLS(y[metric], X).fit(cov_type='cluster', cov_kwds={'groups': y['month_cluster']})
        t_cl = model.tvalues[0]
        p_cl = model.pvalues[0]
    except:
        t_cl, p_cl = np.nan, np.nan
    
    p(f"  {label}: Standard t={t_std:.2f}(p={p_std:.4f}) | Clustered t={t_cl:.2f}(p={p_cl:.4f})")

# ============================================================
# Step 6: Non-Parametric Tests
# ============================================================
p("\n=== NON-PARAMETRIC TESTS ===")

for metric, label in [('car_pre5', 'CAR[-5,-1]'), ('car_pre', 'CAR[-3,-1]'), ('car_event', 'CAR[0,+1]')]:
    vals = mm2[metric].dropna()
    
    # Sign test
    n_pos = (vals > 0).sum()
    n_total = len(vals)
    bt = binomtest(n_pos, n_total, 0.5)
    
    # Wilcoxon signed-rank
    try:
        w_stat, w_p = wilcoxon(vals)
    except:
        w_stat, w_p = np.nan, np.nan
    
    p(f"  {label}: Sign test {n_pos}/{n_total}={n_pos/n_total*100:.1f}% (p={bt.pvalue:.4f}) | Wilcoxon p={w_p:.4f}")

# ============================================================
# Step 7: Standardized Abnormal Returns
# ============================================================
p("\n=== STANDARDIZED ABNORMAL RETURNS ===")

# Use estimation period std to standardize
# We have alpha and beta from market model, need estimation period residual std
# For now, use cross-sectional approach: divide by cross-sectional std of AR

for d in range(-5, 6):
    col = f'ar_mm_{d}'
    vals = mm2[col].dropna()
    sar = vals / vals.std()
    t_sar = sar.mean() * np.sqrt(len(sar))
    p_sar = 2 * (1 - stats.norm.cdf(abs(t_sar)))
    p(f"  Day {d:+d}: SAR mean={sar.mean():.4f}, t={t_sar:.2f}, p={p_sar:.4f}")

# ============================================================
# Step 8: Save
# ============================================================
p("\nSaving results...")
summary = {
    'n_with_surprise': int(len(valid)) if len(valid) > 0 else 0,
    'n_pos_surprise': int((valid['surprise_pos']==1).sum()) if len(valid) > 0 else 0,
    'n_neg_surprise': int((valid['surprise_pos']==0).sum()) if len(valid) > 0 else 0,
}

if len(valid) > 50:
    summary.update({
        'pos_surprise_car_pre': float(pos['car_pre'].mean()),
        'neg_surprise_car_pre': float(neg['car_pre'].mean()),
        'direction_accuracy': float(accuracy),
        'direction_pvalue': float(bt.pvalue),
        'corr_yoy_car': float(corr),
        'spearman_rho': float(rho),
        'spearman_p': float(rho_p),
    })

with open(f'{DATA_DIR}/surprise_analysis.json', 'w') as f:
    json.dump(summary, f, indent=2)

mm2.to_pickle(f'{DATA_DIR}/leakage_with_surprise.pkl')
p(f"\n✅ Done! {json.dumps(summary, indent=2)}")
