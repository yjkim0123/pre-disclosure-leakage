#!/usr/bin/env python3
"""
Chaebol Information Spillover Analysis
When Samsung Electronics reports earnings, do Samsung SDI/SDS/etc. move?
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GroupKFold
import json, time, warnings
warnings.filterwarnings('ignore')

DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
def p(msg): print(msg, flush=True)

# =====================
# 1. Chaebol Group Mapping
# =====================
p("=== 1. 재벌 그룹 매핑 ===")

CHAEBOL_GROUPS = {
    'Samsung': ['삼성전자', '삼성SDI', '삼성전기', '삼성물산', '삼성에스디에스', '삼성생명'],
    'SK': ['SK하이닉스', 'SK이노베이션', 'SK텔레콤'],
    'Hyundai': ['현대차', '기아', '현대모비스'],
    'LG': ['LG', 'LG화학', 'LG전자'],
    'Financial_KB': ['KB금융'],
    'Financial_Shinhan': ['신한지주'],
    'Financial_Hana': ['하나금융'],
    'Financial_Woori': ['우리금융'],
}

# Industry mapping
INDUSTRY = {
    '삼성전자': 'Semiconductor/Electronics', 'SK하이닉스': 'Semiconductor/Electronics',
    '삼성전기': 'Semiconductor/Electronics', '삼성에스디에스': 'IT_Services',
    '삼성SDI': 'Battery/Chemical', 'LG화학': 'Battery/Chemical', 'LG': 'Conglomerate',
    '삼성물산': 'Conglomerate', 'POSCO홀딩스': 'Steel/Materials', '롯데케미칼': 'Chemical',
    '현대차': 'Auto', '기아': 'Auto', '현대모비스': 'Auto_Parts',
    '네이버': 'Internet/Platform', '카카오': 'Internet/Platform',
    'SK텔레콤': 'Telecom', 'KT': 'Telecom',
    'SK이노베이션': 'Energy/Refining', 'S-Oil': 'Energy/Refining',
    'LG전자': 'Electronics', '삼성생명': 'Insurance',
    'KB금융': 'Banking', '신한지주': 'Banking', '하나금융': 'Banking', '우리금융': 'Banking',
    '셀트리온': 'Bio/Pharma', '아모레퍼시픽': 'Consumer',
    'KT&G': 'Consumer', '대한항공': 'Airlines',
}

# Company → chaebol mapping
company_to_chaebol = {}
for group, members in CHAEBOL_GROUPS.items():
    for m in members:
        company_to_chaebol[m] = group

# =====================
# 2. Load Data
# =====================
p("\n=== 2. 데이터 로드 ===")
df = pd.read_pickle(f'{DATA_DIR}/dataset_final_v3.pkl')
df['chaebol'] = df['company_name'].map(company_to_chaebol)
df['industry'] = df['company_name'].map(INDUSTRY)
df['disc_date'] = pd.to_datetime(df['rcept_dt'])

p(f"총 공시: {len(df)}건")
p(f"재벌 소속: {df['chaebol'].notna().sum()}건")
p(f"\n재벌별 공시 수:")
for g in ['Samsung', 'SK', 'Hyundai', 'LG']:
    sub = df[df['chaebol']==g]
    p(f"  {g}: {len(sub)}건, 기업 {sub['company_name'].nunique()}개 ({', '.join(sub['company_name'].unique())})")

# =====================
# 3. Compute Peer CARs (Spillover)
# =====================
p("\n=== 3. Peer CAR 계산 ===")

# Get stock code mapping
code_map = df.groupby('company_name')['stock_code'].first().to_dict()
all_companies = df['company_name'].unique()

# Load all stock prices
p("주가 데이터 로드...")
price_data = {}
for company in all_companies:
    code = code_map[company]
    try:
        prices = fdr.DataReader(code, '2019-12-01', '2026-03-04')
        price_data[company] = prices
    except:
        p(f"  {company}: 주가 로드 실패")
    time.sleep(0.1)
p(f"주가 로드: {len(price_data)}/{len(all_companies)} 기업")

# KOSPI index
try:
    kospi = fdr.DataReader('^KS11', '2019-12-01', '2026-03-04')
    p(f"KOSPI: {len(kospi)} days")
except:
    kospi = None
    p("KOSPI 실패")

def compute_car(prices, disc_date, kospi_data=None, window=1):
    """Compute market-adjusted CAR for a given stock around disc_date"""
    try:
        td = prices.index
        post = td[td >= disc_date]
        pre = td[td < disc_date]
        if len(post) < window + 1 or len(pre) < 1:
            return np.nan
        pc = prices.loc[pre[-1], 'Close']
        end_idx = min(window, len(post) - 1)
        ret = (prices.loc[post[end_idx], 'Close'] - pc) / pc
        
        # Market adjustment
        if kospi_data is not None:
            mp = kospi_data.index[kospi_data.index >= disc_date]
            mpr = kospi_data.index[kospi_data.index < disc_date]
            if len(mp) > end_idx and len(mpr) >= 1:
                mpc = kospi_data.loc[mpr[-1], 'Close']
                mret = (kospi_data.loc[mp[end_idx], 'Close'] - mpc) / mpc
                ret = ret - mret
        return ret * 100
    except:
        return np.nan

# For each disclosure, compute:
# 1. Own CAR (already have this)
# 2. Same-chaebol peers CAR
# 3. Same-industry non-chaebol peers CAR
# 4. Unrelated firms CAR

spillover_results = []

for idx, row in df.iterrows():
    discloser = row['company_name']
    disc_date = row['disc_date']
    chaebol = row.get('chaebol')
    industry = row.get('industry')
    
    # Own CAR
    own_car = row.get('mcar_1d', np.nan)
    
    # Chaebol peers
    chaebol_cars = []
    if chaebol and chaebol in CHAEBOL_GROUPS:
        for peer in CHAEBOL_GROUPS[chaebol]:
            if peer != discloser and peer in price_data:
                car = compute_car(price_data[peer], disc_date, kospi, window=1)
                if not np.isnan(car):
                    chaebol_cars.append(car)
    
    # Same industry, different chaebol
    industry_cars = []
    if industry:
        for comp, ind in INDUSTRY.items():
            if ind == industry and comp != discloser and comp in price_data:
                comp_chaebol = company_to_chaebol.get(comp)
                if comp_chaebol != chaebol:  # different chaebol or no chaebol
                    car = compute_car(price_data[comp], disc_date, kospi, window=1)
                    if not np.isnan(car):
                        industry_cars.append(car)
    
    # Unrelated (different chaebol AND different industry)
    unrelated_cars = []
    for comp in all_companies:
        if comp == discloser or comp not in price_data:
            continue
        comp_chaebol = company_to_chaebol.get(comp)
        comp_industry = INDUSTRY.get(comp)
        if comp_chaebol != chaebol and comp_industry != industry:
            car = compute_car(price_data[comp], disc_date, kospi, window=1)
            if not np.isnan(car):
                unrelated_cars.append(car)
    
    spillover_results.append({
        'rcept_no': row['rcept_no'],
        'discloser': discloser,
        'chaebol': chaebol,
        'industry': industry,
        'own_car': own_car,
        'chaebol_peer_car': np.mean(chaebol_cars) if chaebol_cars else np.nan,
        'chaebol_peer_n': len(chaebol_cars),
        'industry_peer_car': np.mean(industry_cars) if industry_cars else np.nan,
        'industry_peer_n': len(industry_cars),
        'unrelated_car': np.mean(unrelated_cars) if unrelated_cars else np.nan,
        'unrelated_n': len(unrelated_cars),
        'surprise_yoy': row.get('op_profit_yoy', np.nan),
    })
    
    if (idx + 1) % 100 == 0:
        p(f"  {idx+1}/{len(df)}")

spill_df = pd.DataFrame(spillover_results)
p(f"\n계산 완료: {len(spill_df)}건")

# =====================
# 4. Main Results
# =====================
p("\n" + "="*60)
p("=== 4. MAIN RESULTS: Spillover Analysis ===")
p("="*60)

# 4a. Overall spillover by type
p("\n--- Overall Mean CAR by Relationship ---")
p(f"{'Type':<25} {'Mean CAR%':>10} {'t-stat':>8} {'p-value':>8} {'N':>6}")
p("-" * 60)

for col, label in [
    ('own_car', 'Own (discloser)'),
    ('chaebol_peer_car', 'Chaebol Peer'),
    ('industry_peer_car', 'Industry Peer'),
    ('unrelated_car', 'Unrelated'),
]:
    vals = spill_df[col].dropna()
    if len(vals) > 2:
        t, pv = stats.ttest_1samp(vals, 0)
        p(f"  {label:<25} {vals.mean():>9.3f}% {t:>7.2f} {pv:>8.4f} {len(vals):>5}")

# 4b. Chaebol spillover vs others (paired test)
both = spill_df.dropna(subset=['chaebol_peer_car', 'unrelated_car'])
if len(both) > 10:
    t, pv = stats.ttest_rel(both['chaebol_peer_car'], both['unrelated_car'])
    diff = both['chaebol_peer_car'].mean() - both['unrelated_car'].mean()
    p(f"\n  Chaebol - Unrelated: {diff:.3f}%p (paired t={t:.2f}, p={pv:.4f})")

both2 = spill_df.dropna(subset=['chaebol_peer_car', 'industry_peer_car'])
if len(both2) > 10:
    t, pv = stats.ttest_rel(both2['chaebol_peer_car'], both2['industry_peer_car'])
    diff = both2['chaebol_peer_car'].mean() - both2['industry_peer_car'].mean()
    p(f"  Chaebol - Industry: {diff:.3f}%p (paired t={t:.2f}, p={pv:.4f})")

# 4c. By chaebol group
p(f"\n--- Spillover by Chaebol Group ---")
for group in ['Samsung', 'Hyundai', 'LG', 'SK']:
    sub = spill_df[spill_df['chaebol']==group]
    peer_cars = sub['chaebol_peer_car'].dropna()
    if len(peer_cars) > 5:
        t, pv = stats.ttest_1samp(peer_cars, 0)
        p(f"  {group:<12}: peer CAR={peer_cars.mean():.3f}%, t={t:.2f}, p={pv:.3f}, n={len(peer_cars)}")

# 4d. Asymmetry: positive vs negative surprise spillover
p(f"\n--- Spillover by Surprise Direction ---")
pos_spill = spill_df[spill_df['surprise_yoy'] > 0]['chaebol_peer_car'].dropna()
neg_spill = spill_df[spill_df['surprise_yoy'] < 0]['chaebol_peer_car'].dropna()
if len(pos_spill) > 5 and len(neg_spill) > 5:
    p(f"  Positive surprise → peer CAR: {pos_spill.mean():.3f}% (n={len(pos_spill)})")
    p(f"  Negative surprise → peer CAR: {neg_spill.mean():.3f}% (n={len(neg_spill)})")
    t, pv = stats.ttest_ind(pos_spill, neg_spill)
    p(f"  Difference: t={t:.2f}, p={pv:.4f}")

# 4e. Flagship effect (Samsung Electronics, Hyundai Motor)
p(f"\n--- Flagship Effect ---")
flagships = ['삼성전자', '현대차', 'SK하이닉스']
for flagship in flagships:
    sub = spill_df[spill_df['discloser']==flagship]
    peer = sub['chaebol_peer_car'].dropna()
    if len(peer) > 3:
        t, pv = stats.ttest_1samp(peer, 0)
        p(f"  {flagship} reports → peer CAR: {peer.mean():.3f}%, t={t:.2f}, p={pv:.3f}, n={len(peer)}")

# Non-flagship
non_flagship = spill_df[~spill_df['discloser'].isin(flagships)]
nf_peer = non_flagship['chaebol_peer_car'].dropna()
if len(nf_peer) > 5:
    t, pv = stats.ttest_1samp(nf_peer, 0)
    p(f"  Non-flagship → peer CAR: {nf_peer.mean():.3f}%, t={t:.2f}, p={pv:.3f}, n={len(nf_peer)}")

# 4f. Correlation: own CAR ↔ peer CAR
both3 = spill_df.dropna(subset=['own_car', 'chaebol_peer_car'])
if len(both3) > 10:
    r, pv = stats.pearsonr(both3['own_car'], both3['chaebol_peer_car'])
    p(f"\n  Own CAR ↔ Chaebol Peer CAR correlation: r={r:.4f}, p={pv:.4f}")

# 4g. Multi-window analysis
p(f"\n--- Multi-window Peer CAR ---")
for window in [1, 3, 5]:
    # Recompute for different windows
    cars_w = []
    for _, row in spill_df.iterrows():
        discloser = row['discloser']
        chaebol = row['chaebol']
        if not chaebol or chaebol not in CHAEBOL_GROUPS:
            continue
        disc_date = df[df['rcept_no']==row['rcept_no']]['disc_date'].iloc[0]
        peer_cars = []
        for peer in CHAEBOL_GROUPS[chaebol]:
            if peer != discloser and peer in price_data:
                car = compute_car(price_data[peer], disc_date, kospi, window=window)
                if not np.isnan(car):
                    peer_cars.append(car)
        if peer_cars:
            cars_w.append(np.mean(peer_cars))
    
    if cars_w:
        vals = np.array(cars_w)
        t, pv = stats.ttest_1samp(vals, 0)
        p(f"  Window [0,{window}]: peer CAR={vals.mean():.3f}%, t={t:.2f}, p={pv:.4f}, n={len(vals)}")

# =====================
# 5. Save
# =====================
spill_df.to_pickle(f'{DATA_DIR}/spillover_results.pkl')
spill_df.to_csv(f'{DATA_DIR}/spillover_results.csv', index=False, encoding='utf-8-sig')

summary = {
    'n_disclosures': len(spill_df),
    'chaebol_peer_car_mean': round(spill_df['chaebol_peer_car'].dropna().mean(), 4),
    'industry_peer_car_mean': round(spill_df['industry_peer_car'].dropna().mean(), 4),
    'unrelated_car_mean': round(spill_df['unrelated_car'].dropna().mean(), 4),
    'chaebol_vs_unrelated_diff': round(
        spill_df['chaebol_peer_car'].dropna().mean() - spill_df['unrelated_car'].dropna().mean(), 4
    ) if spill_df['chaebol_peer_car'].notna().any() else None,
}
with open(f'{DATA_DIR}/spillover_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

p(f"\n✅ 완료!")
p(json.dumps(summary, indent=2))
