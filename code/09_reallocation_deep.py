#!/usr/bin/env python3
"""
Deep dive: Intra-Chaebol Resource Reallocation Hypothesis
When one member reports bad earnings, do peers benefit?
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
def p(msg): print(msg, flush=True)

# =====================
# 1. Load spillover data
# =====================
p("=== 1. 데이터 로드 ===")
spill = pd.read_pickle(f'{DATA_DIR}/spillover_results.pkl')
df = pd.read_pickle(f'{DATA_DIR}/dataset_final_v3.pkl')

# Merge additional features
spill = spill.merge(
    df[['rcept_no', 'revenue_yoy', 'op_profit_qoq', 'volume_ratio', 'pre_vol_20d',
        'quarter_inferred', 'disc_year', 'op_profit', 'revenue']].drop_duplicates('rcept_no'),
    on='rcept_no', how='left'
)

# Filter: only chaebol members with peer data
chaebol_spill = spill[spill['chaebol_peer_car'].notna() & spill['chaebol'].notna()].copy()
p(f"재벌 spillover 데이터: {len(chaebol_spill)}건")

# Surprise direction
chaebol_spill['surprise_dir'] = np.where(
    chaebol_spill['surprise_yoy'] > 0, 'positive',
    np.where(chaebol_spill['surprise_yoy'] < 0, 'negative', 'zero')
)
chaebol_spill['surprise_dir'] = chaebol_spill['surprise_dir'].where(
    chaebol_spill['surprise_yoy'].notna(), np.nan
)

# =====================
# 2. Core Test: Asymmetric Spillover
# =====================
p("\n" + "="*60)
p("=== 2. ASYMMETRIC SPILLOVER TEST ===")
p("="*60)

pos = chaebol_spill[chaebol_spill['surprise_dir']=='positive']['chaebol_peer_car']
neg = chaebol_spill[chaebol_spill['surprise_dir']=='negative']['chaebol_peer_car']

p(f"\nPositive surprise → peer CAR: mean={pos.mean():.3f}%, median={pos.median():.3f}%, n={len(pos)}")
p(f"Negative surprise → peer CAR: mean={neg.mean():.3f}%, median={neg.median():.3f}%, n={len(neg)}")

# t-test
t, pv = stats.ttest_ind(neg, pos)
p(f"\nt-test (neg vs pos): t={t:.3f}, p={pv:.4f}")

# Mann-Whitney U (non-parametric)
u, pu = stats.mannwhitneyu(neg, pos, alternative='greater')
p(f"Mann-Whitney U (neg > pos): U={u:.0f}, p={pu:.4f}")

# One-sample tests
t_neg, p_neg = stats.ttest_1samp(neg, 0)
t_pos, p_pos = stats.ttest_1samp(pos, 0)
p(f"\nNeg surprise peer CAR ≠ 0: t={t_neg:.2f}, p={p_neg:.4f}")
p(f"Pos surprise peer CAR ≠ 0: t={t_pos:.2f}, p={p_pos:.4f}")

# =====================
# 3. Magnitude Analysis
# =====================
p("\n=== 3. Surprise Magnitude × Peer CAR ===")

chaebol_spill['surprise_mag'] = pd.cut(
    chaebol_spill['surprise_yoy'],
    bins=[-np.inf, -50, -20, 0, 20, 50, np.inf],
    labels=['<-50%', '-50~-20%', '-20~0%', '0~20%', '20~50%', '>50%']
)

p(f"{'Magnitude':<12} {'Peer CAR%':>10} {'t':>6} {'p':>8} {'N':>5}")
p("-" * 45)
for mag in ['<-50%', '-50~-20%', '-20~0%', '0~20%', '20~50%', '>50%']:
    sub = chaebol_spill[chaebol_spill['surprise_mag']==mag]['chaebol_peer_car']
    if len(sub) > 3:
        t, pv = stats.ttest_1samp(sub, 0)
        p(f"  {mag:<12} {sub.mean():>9.3f}% {t:>5.2f} {pv:>7.4f} {len(sub):>4}")

# Monotonicity test (Spearman correlation: surprise → peer CAR)
valid = chaebol_spill.dropna(subset=['surprise_yoy', 'chaebol_peer_car'])
rho, p_rho = stats.spearmanr(valid['surprise_yoy'], valid['chaebol_peer_car'])
p(f"\nSpearman(surprise YoY, peer CAR): ρ={rho:.4f}, p={p_rho:.4f}")

# =====================
# 4. By Chaebol Group
# =====================
p("\n=== 4. By Chaebol Group ===")

for group in ['Samsung', 'Hyundai', 'SK', 'LG']:
    sub = chaebol_spill[chaebol_spill['chaebol']==group]
    sub_neg = sub[sub['surprise_dir']=='negative']['chaebol_peer_car']
    sub_pos = sub[sub['surprise_dir']=='positive']['chaebol_peer_car']
    
    p(f"\n  {group}:")
    if len(sub_neg) > 2:
        p(f"    Neg surprise → peer: {sub_neg.mean():.3f}% (n={len(sub_neg)})")
    if len(sub_pos) > 2:
        p(f"    Pos surprise → peer: {sub_pos.mean():.3f}% (n={len(sub_pos)})")
    if len(sub_neg) > 2 and len(sub_pos) > 2:
        diff = sub_neg.mean() - sub_pos.mean()
        p(f"    Difference: {diff:.3f}%p")

# =====================
# 5. Own CAR vs Peer CAR Relationship
# =====================
p("\n=== 5. Own CAR → Peer CAR (by surprise direction) ===")

for direction in ['negative', 'positive']:
    sub = chaebol_spill[chaebol_spill['surprise_dir']==direction].dropna(
        subset=['own_car', 'chaebol_peer_car']
    )
    if len(sub) > 10:
        r, pv = stats.pearsonr(sub['own_car'], sub['chaebol_peer_car'])
        p(f"  {direction}: own↔peer r={r:.4f}, p={pv:.4f}, n={len(sub)}")

# =====================
# 6. Cross-sectional Regression
# =====================
p("\n=== 6. OLS: Peer CAR Determinants ===")

# Create interaction terms
chaebol_spill['neg_dummy'] = (chaebol_spill['surprise_dir'] == 'negative').astype(int)
chaebol_spill['surprise_yoy_clean'] = chaebol_spill['surprise_yoy'].clip(-200, 500)
chaebol_spill['neg_x_surprise'] = chaebol_spill['neg_dummy'] * chaebol_spill['surprise_yoy_clean'].abs()

# Samsung dummy
chaebol_spill['is_samsung'] = (chaebol_spill['chaebol'] == 'Samsung').astype(int)
chaebol_spill['is_hyundai'] = (chaebol_spill['chaebol'] == 'Hyundai').astype(int)

reg_cols = ['surprise_yoy_clean', 'neg_dummy', 'neg_x_surprise',
            'own_car', 'volume_ratio', 'is_samsung', 'is_hyundai']

reg_df = chaebol_spill[reg_cols + ['chaebol_peer_car']].dropna()
if len(reg_df) > 30:
    X = sm.add_constant(reg_df[reg_cols])
    model = sm.OLS(reg_df['chaebol_peer_car'], X).fit(cov_type='HC1')
    p(model.summary().as_text())

# =====================
# 7. Quarterly Pattern
# =====================
p("\n=== 7. Quarterly Pattern ===")
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = chaebol_spill[chaebol_spill['quarter_inferred']==q]
    neg_sub = sub[sub['surprise_dir']=='negative']['chaebol_peer_car']
    pos_sub = sub[sub['surprise_dir']=='positive']['chaebol_peer_car']
    if len(neg_sub) > 2 and len(pos_sub) > 2:
        p(f"  {q}: neg→peer={neg_sub.mean():.3f}% (n={len(neg_sub)}), "
          f"pos→peer={pos_sub.mean():.3f}% (n={len(pos_sub)}), "
          f"diff={neg_sub.mean()-pos_sub.mean():.3f}%p")

# =====================
# 8. Robustness: Industry Control
# =====================
p("\n=== 8. Robustness: Chaebol vs Industry Spillover ===")

# Compare: when negative surprise, does chaebol peer react differently from industry peer?
neg_events = chaebol_spill[chaebol_spill['surprise_dir']=='negative']
neg_chaebol = neg_events['chaebol_peer_car'].dropna()
neg_industry = neg_events['industry_peer_car'].dropna()
neg_unrelated = neg_events['unrelated_car'].dropna()

p(f"  Negative surprise events:")
p(f"    Chaebol peer: {neg_chaebol.mean():.3f}% (n={len(neg_chaebol)})")
p(f"    Industry peer: {neg_industry.mean():.3f}% (n={len(neg_industry)})")
p(f"    Unrelated: {neg_unrelated.mean():.3f}% (n={len(neg_unrelated)})")

if len(neg_chaebol) > 5 and len(neg_industry) > 5:
    t, pv = stats.ttest_ind(neg_chaebol, neg_industry)
    p(f"    Chaebol vs Industry: t={t:.2f}, p={pv:.4f}")

pos_events = chaebol_spill[chaebol_spill['surprise_dir']=='positive']
pos_chaebol = pos_events['chaebol_peer_car'].dropna()
pos_industry = pos_events['industry_peer_car'].dropna()

p(f"\n  Positive surprise events:")
p(f"    Chaebol peer: {pos_chaebol.mean():.3f}% (n={len(pos_chaebol)})")
p(f"    Industry peer: {pos_industry.mean():.3f}% (n={len(pos_industry)})")

# =====================
# 9. Bootstrap Confidence Intervals
# =====================
p("\n=== 9. Bootstrap CI (Neg-Pos Difference) ===")
np.random.seed(42)
n_boot = 10000
boot_diffs = []
for _ in range(n_boot):
    neg_sample = np.random.choice(neg.values, size=len(neg), replace=True)
    pos_sample = np.random.choice(pos.values, size=len(pos), replace=True)
    boot_diffs.append(neg_sample.mean() - pos_sample.mean())

boot_diffs = np.array(boot_diffs)
ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
p(f"  Neg-Pos difference: {np.mean(boot_diffs):.3f}%p")
p(f"  95% Bootstrap CI: [{ci_low:.3f}, {ci_high:.3f}]")
p(f"  0 in CI: {'Yes ❌' if ci_low <= 0 <= ci_high else 'No ✅'}")

# =====================
# 10. Summary
# =====================
p("\n" + "="*60)
p("=== SUMMARY ===")
p("="*60)

summary = {
    'n_chaebol_events': len(chaebol_spill),
    'neg_surprise_peer_car': round(neg.mean(), 4),
    'pos_surprise_peer_car': round(pos.mean(), 4),
    'neg_pos_diff': round(neg.mean() - pos.mean(), 4),
    'ttest_pval': round(pv, 4),
    'mannwhitney_pval': round(pu, 4),
    'spearman_rho': round(rho, 4),
    'bootstrap_ci': [round(ci_low, 4), round(ci_high, 4)],
    'own_peer_corr': round(
        stats.pearsonr(
            chaebol_spill.dropna(subset=['own_car','chaebol_peer_car'])['own_car'],
            chaebol_spill.dropna(subset=['own_car','chaebol_peer_car'])['chaebol_peer_car']
        )[0], 4
    ),
}

with open(f'{DATA_DIR}/reallocation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

p(json.dumps(summary, indent=2))
p("\n✅ 완료!")
