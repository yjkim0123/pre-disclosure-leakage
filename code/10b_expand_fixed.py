#!/usr/bin/env python3
"""
Step 10b: 확대 재벌 spillover — 종목코드 수동 매핑
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
# 1. 전체 종목코드 매핑
# =====================
CODE_MAP = {
    # Samsung (12)
    '삼성전자': '005930', '삼성SDI': '006400', '삼성전기': '009150',
    '삼성물산': '028260', '삼성에스디에스': '018260', '삼성생명': '032830',
    '삼성화재': '000810', '삼성증권': '016360', '호텔신라': '008770',
    '삼성엔지니어링': '028050', '제일기획': '030000', '에스원': '012750',
    # SK (8)
    'SK하이닉스': '000660', 'SK이노베이션': '096770', 'SK텔레콤': '017670',
    'SK': '034730', 'SK네트웍스': '001740', 'SK바이오팜': '326030',
    'SK바이오사이언스': '302440', 'SKC': '011790',
    # Hyundai Motor (8)
    '현대차': '005380', '기아': '000270', '현대모비스': '012330',
    '현대위아': '011210', '현대오토에버': '307950', '현대제철': '004020',
    '현대건설': '000720', '현대글로비스': '086280',
    # LG (8)
    'LG': '003550', 'LG화학': '051910', 'LG전자': '066570',
    'LG이노텍': '011070', 'LG디스플레이': '034220', 'LG유플러스': '032640',
    'LG생활건강': '051900', 'LG에너지솔루션': '373220',
    # Lotte (5)
    '롯데케미칼': '011170', '롯데쇼핑': '023530', '롯데지주': '004990',
    '롯데정밀화학': '004000', '롯데칠성음료': '005300',
    # Hanwha (5)
    '한화': '000880', '한화솔루션': '009830', '한화에어로스페이스': '012450',
    '한화생명': '088350', '한화오션': '042660',
    # POSCO (4)
    'POSCO홀딩스': '005490', '포스코퓨처엠': '003670',
    '포스코인터내셔널': '047050', '포스코DX': '022100',
    # HD Hyundai (5)
    'HD현대': '267250', 'HD한국조선해양': '009540', 'HD현대중공업': '329180',
    'HD현대일렉트릭': '267260', '현대미포조선': '010620',
    # Doosan (4)
    '두산': '000150', '두산에너빌리티': '034020', '두산밥캣': '241560',
    '두산로보틱스': '454910',
    # GS (2)
    'GS': '078930', 'GS건설': '006360',
    # CJ (4)
    'CJ': '001040', 'CJ제일제당': '097950', 'CJ대한통운': '000120', 'CJ ENM': '035760',
    # KT (2)
    'KT': '030200', 'KT&G': '033780',
}

CHAEBOL = {
    'Samsung': ['삼성전자','삼성SDI','삼성전기','삼성물산','삼성에스디에스','삼성생명',
                '삼성화재','삼성증권','호텔신라','삼성엔지니어링','제일기획','에스원'],
    'SK': ['SK하이닉스','SK이노베이션','SK텔레콤','SK','SK네트웍스','SK바이오팜','SK바이오사이언스','SKC'],
    'Hyundai_Motor': ['현대차','기아','현대모비스','현대위아','현대오토에버','현대제철','현대건설','현대글로비스'],
    'LG': ['LG','LG화학','LG전자','LG이노텍','LG디스플레이','LG유플러스','LG생활건강','LG에너지솔루션'],
    'Lotte': ['롯데케미칼','롯데쇼핑','롯데지주','롯데정밀화학','롯데칠성음료'],
    'Hanwha': ['한화','한화솔루션','한화에어로스페이스','한화생명','한화오션'],
    'POSCO': ['POSCO홀딩스','포스코퓨처엠','포스코인터내셔널','포스코DX'],
    'HD_Hyundai': ['HD현대','HD한국조선해양','HD현대중공업','HD현대일렉트릭','현대미포조선'],
    'Doosan': ['두산','두산에너빌리티','두산밥캣','두산로보틱스'],
    'GS': ['GS','GS건설'],
    'CJ': ['CJ','CJ제일제당','CJ대한통운','CJ ENM'],
    'KT': ['KT','KT&G'],
}

comp2chaebol = {}
for g, ms in CHAEBOL.items():
    for m in ms:
        comp2chaebol[m] = g

p(f"재벌 {len(CHAEBOL)}개 그룹, {len(CODE_MAP)}개 기업")

# =====================
# 2. DART 공시 수집
# =====================
p("\n=== 2. DART 공시 수집 ===")
expanded_path = f'{DATA_DIR}/earnings_expanded.pkl'
try:
    all_e = pd.read_pickle(expanded_path)
    p(f"기존 데이터: {len(all_e)}건, {all_e['company_name'].nunique()}기업")
    existing = set(all_e['company_name'].unique())
    new_companies = [c for c in CODE_MAP if c not in existing]
except:
    all_e = None
    new_companies = list(CODE_MAP.keys())

if new_companies:
    p(f"새로 수집: {len(new_companies)}개 기업")
    new_list = []
    for company in new_companies:
        code = CODE_MAP[company]
        try:
            df = dart.list(code, start='2020-01-01', end='2026-03-04')
            if df is not None and len(df) > 0:
                e = df[df['report_nm'].str.contains('영업.*실적.*공정공시', regex=True, na=False)]
                orig = e[~e['report_nm'].str.contains('기재정정', na=False)].copy()
                if len(orig) > 0:
                    orig['stock_code'] = code
                    orig['company_name'] = company
                    orig['chaebol'] = comp2chaebol.get(company)
                    new_list.append(orig)
                    p(f"  {company}: {len(orig)}건")
            time.sleep(0.5)
        except Exception as ex:
            p(f"  {company}: ERROR {ex}")
            time.sleep(1)
    
    if new_list:
        new_df = pd.concat(new_list, ignore_index=True)
        if all_e is not None:
            all_e = pd.concat([all_e, new_df], ignore_index=True)
        else:
            all_e = new_df
        all_e.to_pickle(expanded_path)

if 'chaebol' not in all_e.columns:
    all_e['chaebol'] = all_e['company_name'].map(comp2chaebol)

p(f"\n총: {len(all_e)}건, {all_e['company_name'].nunique()}기업, {all_e['chaebol'].nunique()}그룹")
for g in sorted(all_e['chaebol'].dropna().unique()):
    sub = all_e[all_e['chaebol']==g]
    p(f"  {g}: {len(sub)}건, {sub['company_name'].nunique()}기업")

# =====================
# 3. 텍스트 + 파싱 (새 것만)
# =====================
p("\n=== 3. 텍스트 수집 ===")
try:
    existing_texts = pd.read_pickle(f'{DATA_DIR}/texts_expanded.pkl')
except:
    try:
        existing_texts = pd.read_pickle(f'{DATA_DIR}/texts_all.pkl')
    except:
        existing_texts = pd.DataFrame(columns=['rcept_no','text','text_len'])

done_rcepts = set(existing_texts['rcept_no'])
new_rcepts = all_e[~all_e['rcept_no'].isin(done_rcepts)]
p(f"기존 텍스트: {len(done_rcepts)}, 새로 수집: {len(new_rcepts)}")

if len(new_rcepts) > 0:
    new_t = []
    for i, (_, row) in enumerate(new_rcepts.iterrows()):
        try:
            doc = dart.document(row['rcept_no'])
            if doc:
                soup = BeautifulSoup(doc, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                new_t.append({'rcept_no': row['rcept_no'], 'text': text, 'text_len': len(text)})
            else:
                new_t.append({'rcept_no': row['rcept_no'], 'text': '', 'text_len': 0})
        except:
            new_t.append({'rcept_no': row['rcept_no'], 'text': '', 'text_len': 0})
        if (i+1) % 50 == 0:
            p(f"  {i+1}/{len(new_rcepts)}")
        time.sleep(0.3)
    
    new_t_df = pd.DataFrame(new_t)
    all_texts = pd.concat([existing_texts, new_t_df], ignore_index=True)
    all_texts.to_pickle(f'{DATA_DIR}/texts_expanded.pkl')
    p(f"텍스트 총: {len(all_texts)}건")
else:
    all_texts = existing_texts

# Parse op_profit_yoy
def parse_yoy(text):
    if not text or len(str(text)) < 50: return np.nan
    pat = r'영업이익\s+당해실적\s+(.*?)누계실적'
    m = re.search(pat, str(text), re.DOTALL)
    if not m: return np.nan
    nums = [float(n.replace(',','')) for n in re.findall(r'(-?[\d,]+\.?\d*)', m.group(1)) 
            if n.replace(',','').replace('.','').replace('-','').isdigit()]
    yoy = nums[4] if len(nums) >= 5 else (nums[3] if len(nums) >= 4 else np.nan)
    return yoy if yoy is not None and abs(yoy) <= 1000 else np.nan

merged = all_e.merge(all_texts[['rcept_no','text','text_len']], on='rcept_no', how='left')
merged['op_profit_yoy'] = merged['text'].apply(parse_yoy)
merged['disc_date'] = pd.to_datetime(merged['rcept_dt'])
p(f"YoY 추출: {merged['op_profit_yoy'].notna().sum()}/{len(merged)}")

# =====================
# 4. 주가 + Spillover
# =====================
p("\n=== 4. 주가 + Spillover ===")
try:
    kospi_idx = fdr.DataReader('^KS11', '2019-12-01', '2026-03-04')
    p(f"KOSPI: {len(kospi_idx)} days")
except:
    kospi_idx = None

price_data = {}
for company, code in CODE_MAP.items():
    try:
        pr = fdr.DataReader(code, '2019-12-01', '2026-03-04')
        if pr is not None and len(pr) > 100:
            price_data[company] = pr
    except:
        pass
    time.sleep(0.1)
p(f"주가 로드: {len(price_data)}/{len(CODE_MAP)}")

def compute_car(prices, dd, ki=None, w=1):
    try:
        td = prices.index
        post, pre = td[td>=dd], td[td<dd]
        if len(post)<w+1 or len(pre)<1: return np.nan
        pc = prices.loc[pre[-1],'Close']
        r = (prices.loc[post[min(w,len(post)-1)],'Close']-pc)/pc
        if ki is not None:
            mp,mpr = ki.index[ki.index>=dd], ki.index[ki.index<dd]
            if len(mp)>w and len(mpr)>=1:
                mpc = ki.loc[mpr[-1],'Close']
                r -= (ki.loc[mp[min(w,len(mp)-1)],'Close']-mpc)/mpc
        return r*100
    except:
        return np.nan

spill = []
for i, (_, row) in enumerate(merged.iterrows()):
    disc = row['company_name']
    chaebol = row.get('chaebol')
    dd = row['disc_date']
    own = compute_car(price_data[disc], dd, kospi_idx) if disc in price_data else np.nan
    peers = []
    if chaebol and chaebol in CHAEBOL:
        for peer in CHAEBOL[chaebol]:
            if peer != disc and peer in price_data:
                c = compute_car(price_data[peer], dd, kospi_idx)
                if not np.isnan(c): peers.append(c)
    spill.append({
        'rcept_no': row['rcept_no'], 'discloser': disc, 'chaebol': chaebol,
        'own_car': own,
        'peer_car': np.mean(peers) if peers else np.nan,
        'peer_n': len(peers),
        'surprise_yoy': row.get('op_profit_yoy', np.nan),
    })
    if (i+1) % 300 == 0: p(f"  {i+1}/{len(merged)}")

sp = pd.DataFrame(spill)
sp.to_pickle(f'{DATA_DIR}/spillover_expanded.pkl')

# =====================
# 5. 결과
# =====================
p("\n" + "="*60)
p("=== EXPANDED RESULTS ===")
p("="*60)

valid = sp[sp['peer_car'].notna() & sp['surprise_yoy'].notna()].copy()
valid['dir'] = np.where(valid['surprise_yoy']>0, 'pos', 'neg')
p(f"유효: {len(valid)}건, {valid['discloser'].nunique()}기업, {valid['chaebol'].nunique()}그룹")

pos = valid[valid['dir']=='pos']['peer_car']
neg = valid[valid['dir']=='neg']['peer_car']

p(f"\nPos surprise → peer: {pos.mean():.3f}% (n={len(pos)})")
p(f"Neg surprise → peer: {neg.mean():.3f}% (n={len(neg)})")
t, pv = stats.ttest_ind(neg, pos)
u, pu = stats.mannwhitneyu(neg, pos, alternative='greater')
p(f"t-test: t={t:.3f}, p={pv:.4f}")
p(f"Mann-Whitney: p={pu:.4f}")

# By group
p(f"\n--- By Group ---")
for g in sorted(valid['chaebol'].unique()):
    sub = valid[valid['chaebol']==g]
    sn = sub[sub['dir']=='neg']['peer_car']
    sp2 = sub[sub['dir']=='pos']['peer_car']
    if len(sn)>2 and len(sp2)>2:
        p(f"  {g:<18}: neg={sn.mean():>6.3f}%(n={len(sn):>3}), pos={sp2.mean():>6.3f}%(n={len(sp2):>3}), diff={sn.mean()-sp2.mean():>6.3f}%p")

# Magnitude
p(f"\n--- Magnitude ---")
valid['mag'] = pd.cut(valid['surprise_yoy'], bins=[-np.inf,-50,-20,0,20,50,np.inf],
                       labels=['<-50','-50~-20','-20~0','0~20','20~50','>50'])
for m in ['<-50','-50~-20','-20~0','0~20','20~50','>50']:
    sub = valid[valid['mag']==m]['peer_car']
    if len(sub)>3:
        t2,p2 = stats.ttest_1samp(sub,0)
        p(f"  {m:<10}: {sub.mean():>7.3f}%, t={t2:>5.2f}, p={p2:.4f}, n={len(sub)}")

# OLS
p(f"\n--- OLS ---")
valid['neg_d'] = (valid['dir']=='neg').astype(int)
valid['surp'] = valid['surprise_yoy'].clip(-200,500)
reg = valid[['surp','neg_d','own_car','peer_n','peer_car']].dropna()
if len(reg)>30:
    X = sm.add_constant(reg[['surp','neg_d','own_car','peer_n']])
    ols = sm.OLS(reg['peer_car'], X).fit(cov_type='HC1')
    p(ols.summary().as_text())

# Bootstrap
np.random.seed(42)
diffs = [np.random.choice(neg.values,len(neg),True).mean()-np.random.choice(pos.values,len(pos),True).mean() for _ in range(10000)]
ci = np.percentile(diffs,[2.5,97.5])
p(f"\nBootstrap: diff={np.mean(diffs):.3f}%p, 95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
p(f"0 in CI: {'Yes ❌' if ci[0]<=0<=ci[1] else 'No ✅'}")

summary = {
    'n': len(valid), 'companies': int(valid['discloser'].nunique()),
    'groups': int(valid['chaebol'].nunique()),
    'neg_peer': round(neg.mean(),4), 'pos_peer': round(pos.mean(),4),
    'diff': round(neg.mean()-pos.mean(),4),
    'ttest_p': round(pv,4), 'mw_p': round(pu,4),
    'ci': [round(ci[0],4), round(ci[1],4)],
}
with open(f'{DATA_DIR}/spillover_expanded_summary.json','w') as f:
    json.dump(summary,f,indent=2)
p(f"\n✅ 완료!")
p(json.dumps(summary,indent=2))
