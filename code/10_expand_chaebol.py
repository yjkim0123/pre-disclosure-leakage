#!/usr/bin/env python3
"""
Step 10: KOSPI 전체 재벌 계열사 확대 — Spillover 분석
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import OpenDartReader
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from scipy import stats
import statsmodels.api as sm
import time, re, json, warnings
warnings.filterwarnings('ignore')

DART_API_KEY = '345beacedad4863cfdbbbb7f565ff85e0b3cb495'
dart = OpenDartReader(DART_API_KEY)
DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
def p(msg): print(msg, flush=True)

# =====================
# 1. 주요 재벌 그룹 계열사 리스트 (KOSPI 상장)
# =====================
p("=== 1. 재벌 계열사 매핑 ===")

# Major chaebols and their KOSPI-listed affiliates
CHAEBOL_FULL = {
    'Samsung': [
        '삼성전자', '삼성SDI', '삼성전기', '삼성물산', '삼성에스디에스',
        '삼성생명', '삼성화재', '삼성증권', '호텔신라', '삼성엔지니어링',
        '제일기획', '에스원',
    ],
    'SK': [
        'SK하이닉스', 'SK이노베이션', 'SK텔레콤', 'SK', 'SK네트웍스',
        'SK바이오팜', 'SK바이오사이언스', 'SKC',
    ],
    'Hyundai_Motor': [
        '현대차', '기아', '현대모비스', '현대위아', '현대오토에버',
        '현대제철', '현대건설', '현대글로비스',
    ],
    'LG': [
        'LG', 'LG화학', 'LG전자', 'LG이노텍', 'LG디스플레이',
        'LG유플러스', 'LG생활건강', 'LG에너지솔루션',
    ],
    'Lotte': [
        '롯데케미칼', '롯데쇼핑', '롯데지주', '롯데정밀화학', '롯데칠성음료',
    ],
    'Hanwha': [
        '한화', '한화솔루션', '한화에어로스페이스', '한화생명', '한화오션',
    ],
    'POSCO': [
        'POSCO홀딩스', '포스코퓨처엠', '포스코인터내셔널', '포스코DX',
    ],
    'HD_Hyundai': [
        'HD현대', 'HD한국조선해양', 'HD현대중공업', 'HD현대일렉트릭',
        '현대미포조선',
    ],
    'KT': ['KT', 'KT&G'],  # KT&G is separate but often grouped
    'Doosan': ['두산', '두산에너빌리티', '두산밥캣', '두산로보틱스'],
    'GS': ['GS', 'GS건설', 'GS칼텍스'],
    'CJ': ['CJ', 'CJ제일제당', 'CJ대한통운', 'CJ ENM'],
}

# Flatten to company → chaebol mapping
company_to_chaebol = {}
for group, members in CHAEBOL_FULL.items():
    for m in members:
        company_to_chaebol[m] = group

all_chaebol_companies = list(company_to_chaebol.keys())
p(f"전체 재벌 계열사: {len(all_chaebol_companies)}개, {len(CHAEBOL_FULL)}개 그룹")

# =====================
# 2. KOSPI에서 종목코드 찾기
# =====================
p("\n=== 2. 종목코드 매핑 ===")
kospi = fdr.StockListing('KOSPI')
p(f"KOSPI 전체: {len(kospi)}종목")

# Match by name
code_map = {}
not_found = []
for company in all_chaebol_companies:
    matches = kospi[kospi['Name'] == company]
    if len(matches) > 0:
        code_map[company] = matches.iloc[0]['Code']
    else:
        # Try partial match
        partial = kospi[kospi['Name'].str.contains(company, na=False)]
        if len(partial) > 0:
            code_map[company] = partial.iloc[0]['Code']
            p(f"  Partial match: {company} → {partial.iloc[0]['Name']} ({partial.iloc[0]['Code']})")
        else:
            not_found.append(company)

p(f"매칭 성공: {len(code_map)}/{len(all_chaebol_companies)}")
if not_found:
    p(f"미매칭: {not_found}")

# =====================
# 3. DART 공시 수집 (확대)
# =====================
p("\n=== 3. DART 공시 수집 ===")

# Check if we already have expanded data
expanded_path = f'{DATA_DIR}/earnings_expanded.pkl'
try:
    all_earnings = pd.read_pickle(expanded_path)
    p(f"기존 확대 데이터 로드: {len(all_earnings)}건")
except:
    all_earnings_list = []
    for company, code in code_map.items():
        try:
            df = dart.list(code, start='2020-01-01', end='2026-03-04')
            if df is not None and len(df) > 0:
                earnings = df[df['report_nm'].str.contains('영업.*실적.*공정공시', regex=True, na=False)]
                orig = earnings[~earnings['report_nm'].str.contains('기재정정', na=False)].copy()
                if len(orig) > 0:
                    orig['stock_code'] = code
                    orig['company_name'] = company
                    orig['chaebol'] = company_to_chaebol[company]
                    all_earnings_list.append(orig)
                    p(f"  {company} ({company_to_chaebol[company]}): {len(orig)}건")
                else:
                    p(f"  {company}: 영업실적 공시 없음")
            else:
                p(f"  {company}: 공시 없음")
        except Exception as e:
            p(f"  {company}: ERROR {e}")
        time.sleep(0.5)
    
    if all_earnings_list:
        all_earnings = pd.concat(all_earnings_list, ignore_index=True)
        all_earnings.to_pickle(expanded_path)
        p(f"\n총 수집: {len(all_earnings)}건, {all_earnings['company_name'].nunique()}개 기업")
    else:
        p("수집 실패!")
        exit(1)

# Filter and stats
p(f"\n--- 확대 데이터 통계 ---")
p(f"총 공시: {len(all_earnings)}건")
p(f"기업: {all_earnings['company_name'].nunique()}개")
p(f"재벌그룹: {all_earnings['chaebol'].nunique()}개")
p(f"\n그룹별:")
for g in sorted(all_earnings['chaebol'].unique()):
    sub = all_earnings[all_earnings['chaebol']==g]
    p(f"  {g}: {len(sub)}건, {sub['company_name'].nunique()}개 기업")

# =====================
# 4. 텍스트 수집 + 파싱 (새 기업만)
# =====================
p("\n=== 4. 텍스트 수집 + 파싱 ===")

# Already have texts for original 30 companies
try:
    existing_texts = pd.read_pickle(f'{DATA_DIR}/texts_all.pkl')
    existing_rcepts = set(existing_texts['rcept_no'])
    p(f"기존 텍스트: {len(existing_texts)}건")
except:
    existing_rcepts = set()

new_rcepts = all_earnings[~all_earnings['rcept_no'].isin(existing_rcepts)]
p(f"새로 수집할 텍스트: {len(new_rcepts)}건")

if len(new_rcepts) > 0:
    new_texts = []
    for i, (idx, row) in enumerate(new_rcepts.iterrows()):
        try:
            doc = dart.document(row['rcept_no'])
            if doc:
                soup = BeautifulSoup(doc, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                new_texts.append({'rcept_no': row['rcept_no'], 'text': text, 'text_len': len(text)})
            else:
                new_texts.append({'rcept_no': row['rcept_no'], 'text': '', 'text_len': 0})
        except:
            new_texts.append({'rcept_no': row['rcept_no'], 'text': '', 'text_len': 0})
        if (i+1) % 50 == 0:
            p(f"  {i+1}/{len(new_rcepts)}")
        time.sleep(0.3)
    
    new_texts_df = pd.DataFrame(new_texts)
    if len(existing_rcepts) > 0:
        all_texts = pd.concat([existing_texts, new_texts_df], ignore_index=True)
    else:
        all_texts = new_texts_df
    all_texts.to_pickle(f'{DATA_DIR}/texts_expanded.pkl')
    p(f"텍스트 수집 완료: {len(all_texts)}건")
else:
    all_texts = existing_texts

# Parse earnings numbers
def extract_section(text, keyword):
    pat = re.escape(keyword) + r'\s+당해실적\s+(.*?)누계실적'
    m = re.search(pat, str(text), re.DOTALL)
    if not m: return None, None
    chunk = m.group(1)
    nums = [float(n.replace(',','')) for n in re.findall(r'(-?[\d,]+\.?\d*)', chunk) if n.replace(',','').replace('.','').replace('-','').isdigit()]
    if len(nums) >= 5: return nums[0], nums[4]  # current, yoy
    elif len(nums) >= 4: return nums[0], nums[3]
    elif len(nums) >= 1: return nums[0], None
    return None, None

merged = all_earnings.merge(all_texts[['rcept_no','text','text_len']], on='rcept_no', how='left')
parsed = merged['text'].apply(lambda t: extract_section(str(t), '영업이익') if t else (None, None))
merged['op_profit'] = parsed.apply(lambda x: x[0])
merged['op_profit_yoy'] = parsed.apply(lambda x: x[1])

# Clean YoY
merged.loc[merged['op_profit_yoy'].abs() > 1000, 'op_profit_yoy'] = np.nan

p(f"\n영업이익 추출: {merged['op_profit'].notna().sum()}/{len(merged)}")
p(f"영업이익 YoY: {merged['op_profit_yoy'].notna().sum()}/{len(merged)}")

# =====================
# 5. 주가 + CAR
# =====================
p("\n=== 5. 주가 수집 ===")
merged['disc_date'] = pd.to_datetime(merged['rcept_dt'])

# KOSPI index
try:
    kospi_idx = fdr.DataReader('^KS11', '2019-12-01', '2026-03-04')
    p(f"KOSPI: {len(kospi_idx)} days")
except:
    kospi_idx = None

# Load all stock prices
price_data = {}
for company in merged['company_name'].unique():
    code = code_map.get(company)
    if not code: continue
    try:
        prices = fdr.DataReader(code, '2019-12-01', '2026-03-04')
        if len(prices) > 100:
            price_data[company] = prices
    except:
        pass
    time.sleep(0.1)
p(f"주가 로드: {len(price_data)} 기업")

def compute_car(prices, disc_date, kospi_data=None, window=1):
    try:
        td = prices.index
        post = td[td >= disc_date]
        pre = td[td < disc_date]
        if len(post) < window + 1 or len(pre) < 1: return np.nan
        pc = prices.loc[pre[-1], 'Close']
        ret = (prices.loc[post[min(window, len(post)-1)], 'Close'] - pc) / pc
        if kospi_data is not None:
            mp = kospi_data.index[kospi_data.index >= disc_date]
            mpr = kospi_data.index[kospi_data.index < disc_date]
            if len(mp) > window and len(mpr) >= 1:
                mpc = kospi_data.loc[mpr[-1], 'Close']
                mret = (kospi_data.loc[mp[min(window, len(mp)-1)], 'Close'] - mpc) / mpc
                ret -= mret
        return ret * 100
    except:
        return np.nan

# =====================
# 6. Spillover 계산 (확대)
# =====================
p("\n=== 6. Spillover 계산 ===")

spillover = []
total = len(merged)
for i, (idx, row) in enumerate(merged.iterrows()):
    discloser = row['company_name']
    chaebol = row['chaebol']
    disc_date = row['disc_date']
    
    # Own CAR
    own_car = compute_car(price_data[discloser], disc_date, kospi_idx) if discloser in price_data else np.nan
    
    # Chaebol peers
    peer_cars = []
    if chaebol in CHAEBOL_FULL:
        for peer in CHAEBOL_FULL[chaebol]:
            if peer != discloser and peer in price_data:
                car = compute_car(price_data[peer], disc_date, kospi_idx)
                if not np.isnan(car):
                    peer_cars.append(car)
    
    spillover.append({
        'rcept_no': row['rcept_no'],
        'discloser': discloser,
        'chaebol': chaebol,
        'disc_date': disc_date,
        'own_car': own_car,
        'peer_car_mean': np.mean(peer_cars) if peer_cars else np.nan,
        'peer_car_median': np.median(peer_cars) if peer_cars else np.nan,
        'peer_n': len(peer_cars),
        'surprise_yoy': row.get('op_profit_yoy', np.nan),
    })
    
    if (i+1) % 200 == 0:
        p(f"  {i+1}/{total}")

sp = pd.DataFrame(spillover)
sp.to_pickle(f'{DATA_DIR}/spillover_expanded.pkl')
p(f"\nSpillover 계산 완료: {len(sp)}건")
p(f"Peer CAR 있는 건: {sp['peer_car_mean'].notna().sum()}")

# =====================
# 7. 결과 분석
# =====================
p("\n" + "="*60)
p("=== 7. EXPANDED RESULTS ===")
p("="*60)

valid = sp[sp['peer_car_mean'].notna() & sp['surprise_yoy'].notna()].copy()
valid['surprise_dir'] = np.where(valid['surprise_yoy'] > 0, 'positive', 'negative')

p(f"\n유효 데이터: {len(valid)}건 (기업 {valid['discloser'].nunique()}개, 그룹 {valid['chaebol'].nunique()}개)")

# Core asymmetry test
pos = valid[valid['surprise_dir']=='positive']['peer_car_mean']
neg = valid[valid['surprise_dir']=='negative']['peer_car_mean']

p(f"\n--- Core Asymmetry Test ---")
p(f"Positive surprise → peer: {pos.mean():.3f}% (n={len(pos)})")
p(f"Negative surprise → peer: {neg.mean():.3f}% (n={len(neg)})")
t, pv = stats.ttest_ind(neg, pos)
u, pu = stats.mannwhitneyu(neg, pos, alternative='greater')
p(f"t-test: t={t:.3f}, p={pv:.4f}")
p(f"Mann-Whitney: U={u:.0f}, p={pu:.4f}")

# One-sample
t_n, p_n = stats.ttest_1samp(neg, 0)
t_p, p_p = stats.ttest_1samp(pos, 0)
p(f"Neg peer ≠ 0: t={t_n:.2f}, p={p_n:.4f}")
p(f"Pos peer ≠ 0: t={t_p:.2f}, p={p_p:.4f}")

# By chaebol
p(f"\n--- By Chaebol Group ---")
for g in sorted(valid['chaebol'].unique()):
    sub = valid[valid['chaebol']==g]
    sub_neg = sub[sub['surprise_dir']=='negative']['peer_car_mean']
    sub_pos = sub[sub['surprise_dir']=='positive']['peer_car_mean']
    if len(sub_neg) > 2 and len(sub_pos) > 2:
        diff = sub_neg.mean() - sub_pos.mean()
        p(f"  {g:<18}: neg={sub_neg.mean():>6.3f}% (n={len(sub_neg):>3}), "
          f"pos={sub_pos.mean():>6.3f}% (n={len(sub_pos):>3}), diff={diff:>6.3f}%p")

# Magnitude
p(f"\n--- Magnitude Bins ---")
valid['mag'] = pd.cut(valid['surprise_yoy'], bins=[-np.inf,-50,-20,0,20,50,np.inf],
                       labels=['<-50','-50~-20','-20~0','0~20','20~50','>50'])
for m in ['<-50','-50~-20','-20~0','0~20','20~50','>50']:
    sub = valid[valid['mag']==m]['peer_car_mean']
    if len(sub) > 3:
        t, pv = stats.ttest_1samp(sub, 0)
        p(f"  {m:<10}: {sub.mean():>7.3f}%, t={t:>5.2f}, p={pv:.4f}, n={len(sub)}")

# OLS
p(f"\n--- OLS Regression ---")
valid['neg_dummy'] = (valid['surprise_dir']=='negative').astype(int)
valid['surprise_clean'] = valid['surprise_yoy'].clip(-200, 500)
valid['neg_x_abs'] = valid['neg_dummy'] * valid['surprise_clean'].abs()

reg_cols = ['surprise_clean', 'neg_dummy', 'own_car', 'peer_n']
reg_df = valid[reg_cols + ['peer_car_mean']].dropna()
if len(reg_df) > 30:
    X = sm.add_constant(reg_df[reg_cols])
    ols = sm.OLS(reg_df['peer_car_mean'], X).fit(cov_type='HC1')
    p(ols.summary().as_text())

# Bootstrap
p(f"\n--- Bootstrap CI ---")
np.random.seed(42)
diffs = []
for _ in range(10000):
    ns = np.random.choice(neg.values, len(neg), replace=True)
    ps = np.random.choice(pos.values, len(pos), replace=True)
    diffs.append(ns.mean() - ps.mean())
diffs = np.array(diffs)
ci = np.percentile(diffs, [2.5, 97.5])
p(f"  Diff: {diffs.mean():.3f}%p, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
p(f"  0 in CI: {'Yes ❌' if ci[0] <= 0 <= ci[1] else 'No ✅'}")

# Save
summary = {
    'n_total': len(sp),
    'n_valid': len(valid),
    'n_companies': int(valid['discloser'].nunique()),
    'n_groups': int(valid['chaebol'].nunique()),
    'neg_peer_car': round(neg.mean(), 4),
    'pos_peer_car': round(pos.mean(), 4),
    'diff': round(neg.mean() - pos.mean(), 4),
    'ttest_p': round(pv, 4),
    'mannwhitney_p': round(pu, 4),
    'bootstrap_ci': [round(ci[0], 4), round(ci[1], 4)],
}
with open(f'{DATA_DIR}/spillover_expanded_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

p(f"\n✅ 완료!")
p(json.dumps(summary, indent=2))
