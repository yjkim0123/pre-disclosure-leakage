#!/usr/bin/env python3
"""
Build final dataset: Parse all texts + Stock CAR + Feature engineering
"""
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import re, time
from datetime import timedelta

DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
def p(msg): print(msg, flush=True)

# =====================
# 1. Load data
# =====================
p("=== 1. 데이터 로드 ===")
earnings = pd.read_pickle(f'{DATA_DIR}/earnings_30.pkl')
texts = pd.read_pickle(f'{DATA_DIR}/texts_all.pkl')

e = earnings[earnings['report_nm'].str.contains('영업.*실적.*공정공시', regex=True, na=False)]
orig = e[~e['report_nm'].str.contains('기재정정', na=False)].copy().reset_index(drop=True)
p(f"공시 {len(orig)}건, 텍스트 {len(texts)}건")

merged = orig.merge(texts, on='rcept_no', how='left')
p(f"Merged: {len(merged)}건, 텍스트 있는 건: {merged['text_len'].notna().sum()}")

# =====================
# 2. Robust number parsing
# =====================
p("\n=== 2. 숫자 파싱 ===")

def extract_section(text, keyword):
    """Extract numbers from 'keyword 당해실적 ... 누계실적' section"""
    pat = re.escape(keyword) + r'\s+당해실적\s+(.*?)누계실적'
    m = re.search(pat, text, re.DOTALL)
    if not m:
        return None, None, None, None, None
    chunk = m.group(1)

    # Check for turnaround markers
    has_qoq_turn = bool(re.search(r'(흑자전환|적자전환|적자지속)', chunk.split('전년동기')[0] if '전년동기' in chunk else chunk[:len(chunk)//2]))
    has_yoy_turn = False

    # Extract all number tokens (handle -N, N.N, N,NNN)
    nums = []
    for n in re.findall(r'(-?[\d,]+\.?\d*)\s*%?', chunk):
        try:
            nums.append(float(n.replace(',', '')))
        except:
            pass

    # Turnaround type
    turn = None
    if '흑자전환' in chunk:
        turn = 'profit_turn'
    elif '적자전환' in chunk:
        turn = 'loss_turn'
    elif '적자지속' in chunk:
        turn = 'loss_cont'

    if len(nums) >= 5:
        # Normal: [current, prev_q, qoq%, prev_year, yoy%]
        return nums[0], nums[2], nums[3], nums[4], turn
    elif len(nums) >= 4:
        # One marker replaced a percentage
        # Most common: QoQ is 흑자/적자전환 → [current, prev_q, prev_year, yoy%]
        return nums[0], None, nums[2], nums[3], turn
    elif len(nums) >= 3:
        # Both QoQ and YoY are markers
        return nums[0], None, None, None, turn
    elif len(nums) >= 1:
        return nums[0], None, None, None, turn
    return None, None, None, None, turn

def parse_all(text):
    if not text or len(str(text)) < 50:
        return {}
    text = str(text)
    result = {}

    # 매출액
    rev, rev_qoq, rev_py, rev_yoy, rev_turn = extract_section(text, '매출액')
    if rev is not None: result['revenue'] = rev
    if rev_qoq is not None: result['revenue_qoq'] = rev_qoq
    if rev_py is not None: result['revenue_prev_year'] = rev_py
    if rev_yoy is not None: result['revenue_yoy'] = rev_yoy
    if rev_turn: result['revenue_turn'] = rev_turn

    # 영업이익
    op, op_qoq, op_py, op_yoy, op_turn = extract_section(text, '영업이익')
    if op is not None: result['op_profit'] = op
    if op_qoq is not None: result['op_profit_qoq'] = op_qoq
    if op_py is not None: result['op_profit_prev_year'] = op_py
    if op_yoy is not None: result['op_profit_yoy'] = op_yoy
    if op_turn: result['op_turn'] = op_turn

    # 당기순이익
    ni, ni_qoq, ni_py, ni_yoy, ni_turn = extract_section(text, '당기순이익')
    if ni is not None: result['net_income'] = ni
    if ni_qoq is not None: result['net_income_qoq'] = ni_qoq
    if ni_yoy is not None: result['net_income_yoy'] = ni_yoy
    if ni_turn: result['ni_turn'] = ni_turn

    # 실적기간 → 분기
    pm = re.search(r'당기실적\s+(\d{4})-(\d{2})-\d{2}\s*~\s*\d{4}-(\d{2})-(\d{2})', text)
    if pm:
        result['period_year'] = int(pm.group(1))
        q_map = {3: 'Q1', 6: 'Q2', 9: 'Q3', 12: 'Q4'}
        result['quarter'] = q_map.get(int(pm.group(3)), f'M{pm.group(3)}')

    # 단위 (억원 vs 백만원)
    if '단위 : 백만원' in text or '단위: 백만원' in text:
        result['unit'] = 'million_won'
    elif '단위 : 억원' in text or '단위: 억원' in text:
        result['unit'] = 'hundred_million_won'

    # 연결 vs 별도
    if '연결재무제표' in text[:200]:
        result['consolidated'] = True
    else:
        result['consolidated'] = False

    # 잠정 vs 확정
    result['is_preliminary'] = '잠정' in text[:300]

    # 텍스트 길이
    result['text_length'] = len(text)

    return result

parsed = merged['text'].apply(parse_all)
parsed_df = pd.json_normalize(parsed)
for col in parsed_df.columns:
    merged[col] = parsed_df[col].values

# Stats
for col in ['revenue', 'op_profit', 'net_income', 'revenue_yoy', 'op_profit_yoy', 'net_income_yoy', 'quarter']:
    if col in merged.columns:
        n = merged[col].notna().sum()
        p(f"  {col}: {n}/{len(merged)} ({n/len(merged)*100:.1f}%)")

# =====================
# 3. 주가 + CAR
# =====================
p("\n=== 3. 주가 수집 + CAR ===")

# KOSPI index
try:
    kospi = fdr.DataReader('KS11', '2019-12-01', '2026-03-04')
    p(f"KOSPI 지수: {len(kospi)} days")
except:
    kospi = None
    p("KOSPI 지수 실패")

car_list = []
for code in merged['stock_code'].unique():
    sub = merged[merged['stock_code'] == code]
    name = sub.iloc[0]['company_name']
    try:
        prices = fdr.DataReader(code, '2019-12-01', '2026-03-04')
        matched = 0
        for _, row in sub.iterrows():
            try:
                dd = pd.to_datetime(row['rcept_dt'])
                td = prices.index
                post = td[td >= dd]
                pre = td[td < dd]
                if len(post) < 5 or len(pre) < 1:
                    continue
                pc = prices.loc[pre[-1], 'Close']
                r1 = (prices.loc[post[0], 'Close'] - pc) / pc
                r3 = (prices.loc[post[min(2, len(post)-1)], 'Close'] - pc) / pc
                r5 = (prices.loc[post[min(4, len(post)-1)], 'Close'] - pc) / pc

                # Market-adjusted
                c1, c3, c5 = r1, r3, r5
                if kospi is not None:
                    try:
                        mp = kospi.index[kospi.index >= dd]
                        mpr = kospi.index[kospi.index < dd]
                        if len(mp) >= 5 and len(mpr) >= 1:
                            mpc = kospi.loc[mpr[-1], 'Close']
                            m1 = (kospi.loc[mp[0], 'Close'] - mpc) / mpc
                            m3 = (kospi.loc[mp[min(2, len(mp)-1)], 'Close'] - mpc) / mpc
                            m5 = (kospi.loc[mp[min(4, len(mp)-1)], 'Close'] - mpc) / mpc
                            c1, c3, c5 = r1-m1, r3-m3, r5-m5
                    except:
                        pass

                vol20 = prices.loc[pre[-20:], 'Close'].pct_change().std() if len(pre) >= 20 else np.nan

                # Trading volume change
                if 'Volume' in prices.columns and len(pre) >= 20:
                    avg_vol = prices.loc[pre[-20:], 'Volume'].mean()
                    disc_vol = prices.loc[post[0], 'Volume'] if len(post) > 0 else np.nan
                    vol_ratio = disc_vol / avg_vol if avg_vol > 0 else np.nan
                else:
                    vol_ratio = np.nan

                car_list.append({
                    'rcept_no': row['rcept_no'],
                    'ret_1d': round(r1*100, 4),
                    'ret_3d': round(r3*100, 4),
                    'ret_5d': round(r5*100, 4),
                    'car_1d': round(c1*100, 4),
                    'car_3d': round(c3*100, 4),
                    'car_5d': round(c5*100, 4),
                    'pre_vol_20d': round(vol20*100, 4) if not np.isnan(vol20) else np.nan,
                    'volume_ratio': round(vol_ratio, 2) if not np.isnan(vol_ratio) else np.nan,
                    'pre_close': pc,
                })
                matched += 1
            except:
                pass
        p(f"  {name}: {matched}/{len(sub)}")
    except Exception as ex:
        p(f"  {name}: ERROR {ex}")
    time.sleep(0.2)

car_df = pd.DataFrame(car_list)
p(f"\nCAR 매칭: {len(car_df)}/{len(merged)}")

final = merged.merge(car_df, on='rcept_no', how='left')

# =====================
# 4. Feature engineering
# =====================
p("\n=== 4. Feature engineering ===")

# Date features
final['disc_date'] = pd.to_datetime(final['rcept_dt'])
final['disc_year'] = final['disc_date'].dt.year
final['disc_month'] = final['disc_date'].dt.month
final['disc_dow'] = final['disc_date'].dt.dayofweek  # 0=Mon

# Surprise direction & magnitude
if 'op_profit_yoy' in final.columns:
    final['surprise_dir'] = np.where(final['op_profit_yoy'] > 0, 'positive',
                            np.where(final['op_profit_yoy'] < 0, 'negative', 'zero'))
    final['surprise_mag'] = pd.cut(
        final['op_profit_yoy'],
        bins=[-np.inf, -50, -20, 0, 20, 50, np.inf],
        labels=['large_neg', 'med_neg', 'small_neg', 'small_pos', 'med_pos', 'large_pos']
    )

# Earnings surprise relative to previous quarter
if 'op_profit' in final.columns and 'op_profit_qoq' in final.columns:
    final['beat_qoq'] = (final['op_profit_qoq'] > 0).astype(int)

# Revenue-profit divergence (revenue up but profit down, or vice versa)
if 'revenue_yoy' in final.columns and 'op_profit_yoy' in final.columns:
    final['rev_profit_diverge'] = (
        (final['revenue_yoy'] > 0) & (final['op_profit_yoy'] < 0) |
        (final['revenue_yoy'] < 0) & (final['op_profit_yoy'] > 0)
    ).astype(int)

# Normalize revenue to 억원 for comparability
if 'revenue' in final.columns and 'unit' in final.columns:
    final['revenue_eok'] = final.apply(
        lambda r: r['revenue'] / 100 if r.get('unit') == 'million_won' else r['revenue'],
        axis=1
    )

# =====================
# 5. Save + Stats
# =====================
final.to_pickle(f'{DATA_DIR}/dataset_final.pkl')
final.to_csv(f'{DATA_DIR}/dataset_final.csv', index=False, encoding='utf-8-sig')

p(f"\n{'='*60}")
p(f"=== 최종 데이터셋 ===")
p(f"{'='*60}")
p(f"총 {len(final)}건, 기업 {final['company_name'].nunique()}개")
p(f"기간: {final['rcept_dt'].min()} ~ {final['rcept_dt'].max()}")
p(f"컬럼 ({len(final.columns)}): {sorted(final.columns.tolist())}")

p(f"\n--- 핵심 통계 ---")
for col in ['revenue', 'op_profit', 'net_income', 'revenue_yoy', 'op_profit_yoy',
            'car_1d', 'car_3d', 'car_5d', 'volume_ratio']:
    if col in final.columns:
        s = final[col].dropna()
        if len(s) > 0:
            p(f"  {col}: n={len(s)}, mean={s.mean():.2f}, std={s.std():.2f}, "
              f"min={s.min():.2f}, max={s.max():.2f}")

# CAR by surprise
if 'car_1d' in final.columns and 'surprise_dir' in final.columns:
    p(f"\n--- CAR[0,1] by Surprise Direction ---")
    for d in ['positive', 'negative']:
        sub = final[final['surprise_dir']==d]['car_1d'].dropna()
        p(f"  {d}: mean={sub.mean():.3f}%, median={sub.median():.3f}%, n={len(sub)}")

    p(f"\n--- CAR[0,1] by Surprise Magnitude ---")
    if 'surprise_mag' in final.columns:
        for cat in ['large_neg','med_neg','small_neg','small_pos','med_pos','large_pos']:
            sub = final[final['surprise_mag']==cat]['car_1d'].dropna()
            if len(sub) > 0:
                p(f"  {cat}: mean={sub.mean():.3f}%, n={len(sub)}")

# Revenue-Profit divergence effect
if 'rev_profit_diverge' in final.columns and 'car_1d' in final.columns:
    p(f"\n--- Revenue-Profit Divergence ---")
    div = final[final['rev_profit_diverge']==1]['car_1d'].dropna()
    nodiv = final[final['rev_profit_diverge']==0]['car_1d'].dropna()
    p(f"  Diverge: mean={div.mean():.3f}%, n={len(div)}")
    p(f"  Aligned: mean={nodiv.mean():.3f}%, n={len(nodiv)}")

# Quarter distribution
if 'quarter' in final.columns:
    p(f"\n--- 분기 분포 ---")
    p(str(final['quarter'].value_counts().sort_index()))

# Turnaround effects
if 'op_turn' in final.columns and 'car_1d' in final.columns:
    p(f"\n--- 흑자/적자 전환 효과 ---")
    for turn in ['profit_turn', 'loss_turn', 'loss_cont']:
        sub = final[final['op_turn']==turn]['car_1d'].dropna()
        if len(sub) > 0:
            p(f"  {turn}: mean CAR={sub.mean():.3f}%, n={len(sub)}")

p(f"\n✅ 저장 완료!")
p(f"  {DATA_DIR}/dataset_final.pkl")
p(f"  {DATA_DIR}/dataset_final.csv")
