#!/usr/bin/env python3
"""
Resume: 나머지 텍스트 수집 + 전체 합치기 + 파싱 + 주가 CAR + 저장
"""
import OpenDartReader
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time, os, re, sys
from datetime import timedelta

DART_API_KEY = '345beacedad4863cfdbbbb7f565ff85e0b3cb495'
dart = OpenDartReader(DART_API_KEY)
DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'

# Flush prints immediately
def p(msg):
    print(msg, flush=True)

# =====================
# 1. 이미 수집된 텍스트 합치기
# =====================
p("=== 1. 기존 텍스트 합치기 ===")
texts_600 = pd.read_pickle(f'{DATA_DIR}/texts_partial_600.pkl')
texts_rem = pd.read_pickle(f'{DATA_DIR}/texts_remaining_500.pkl')
done_texts = pd.concat([texts_600, texts_rem], ignore_index=True)
done_rcept = set(done_texts['rcept_no'])
p(f"수집 완료: {len(done_texts)}건")

# 원본 공시 목록
earnings = pd.read_pickle(f'{DATA_DIR}/earnings_30.pkl')
e = earnings[earnings['report_nm'].str.contains('영업.*실적.*공정공시', regex=True, na=False)]
orig = e[~e['report_nm'].str.contains('기재정정', na=False)].copy().reset_index(drop=True)
p(f"전체 대상: {len(orig)}건")

remaining = orig[~orig['rcept_no'].isin(done_rcept)]
p(f"남은 건: {len(remaining)}건")

# =====================
# 2. 나머지 수집
# =====================
if len(remaining) > 0:
    p(f"\n=== 2. 나머지 {len(remaining)}건 수집 ===")
    new_texts = []
    for i, (idx, row) in enumerate(remaining.iterrows()):
        rcept_no = row['rcept_no']
        try:
            doc = dart.document(rcept_no)
            if doc:
                soup = BeautifulSoup(doc, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
                new_texts.append({'rcept_no': rcept_no, 'text': text, 'text_len': len(text)})
            else:
                new_texts.append({'rcept_no': rcept_no, 'text': '', 'text_len': 0})
        except Exception as ex:
            new_texts.append({'rcept_no': rcept_no, 'text': f'ERROR: {ex}', 'text_len': 0})
        if (i+1) % 20 == 0:
            p(f"  {i+1}/{len(remaining)}")
        time.sleep(0.3)

    new_df = pd.DataFrame(new_texts)
    done_texts = pd.concat([done_texts, new_df], ignore_index=True)
    p(f"전체 텍스트: {len(done_texts)}건")

# 저장
done_texts.to_pickle(f'{DATA_DIR}/texts_all.pkl')
p(f"texts_all.pkl 저장 완료")

# =====================
# 3. 숫자 파싱
# =====================
p("\n=== 3. 숫자 파싱 ===")

def parse_earnings(text):
    result = {}
    if not text or len(str(text)) < 50:
        return result
    text = str(text)
    try:
        # 매출액 당해실적
        for pat in [r'매출액\s*(?:\()?당해실적(?:\))?\s*([-\d,\.]+)',
                     r'매출액\s+당해실적\s+([-\d,\.]+)']:
            m = re.search(pat, text)
            if m:
                result['revenue'] = float(m.group(1).replace(',', ''))
                break

        # 영업이익 당해실적
        for pat in [r'영업이익\s*(?:\()?당해실적(?:\))?\s*([-\d,\.]+)',
                     r'영업이익\s+당해실적\s+([-\d,\.]+)']:
            m = re.search(pat, text)
            if m:
                result['op_profit'] = float(m.group(1).replace(',', ''))
                break

        # 당기순이익 당해실적
        for pat in [r'당기순이익\s*(?:\()?당해실적(?:\))?\s*([-\d,\.]+)',
                     r'순이익\s+당해실적\s+([-\d,\.]+)']:
            m = re.search(pat, text)
            if m:
                result['net_income'] = float(m.group(1).replace(',', ''))
                break

        # 전년동기대비 증감율 — look for the table-like structure
        # "매출액 당해실적 XX 누계실적 XX 영업이익 ... 전년동기대비 증감율(%)"
        # In the text structure: after "전년동기대비" there's typically a percentage
        
        # Revenue YoY: find after 매출액 section
        rev_section = re.search(r'매출액.*?전년동기.*?증감율.*?([-\d,\.]+)', text[:len(text)//2], re.DOTALL)
        if rev_section:
            try:
                val = float(rev_section.group(1).replace(',', ''))
                if -500 < val < 10000:
                    result['revenue_yoy'] = val
            except:
                pass

        # Operating profit YoY
        op_section = re.search(r'영업이익.*?전년동기.*?증감율.*?([-\d,\.]+)', text, re.DOTALL)
        if op_section:
            try:
                val = float(op_section.group(1).replace(',', ''))
                if -500 < val < 50000:
                    result['op_profit_yoy'] = val
            except:
                pass

        # 실적기간
        pm = re.search(r'당기실적\s+(\d{4})-(\d{2})-\d{2}\s*~\s*\d{4}-(\d{2})-(\d{2})', text)
        if pm:
            result['period_year'] = int(pm.group(1))
            q_map = {3: 'Q1', 6: 'Q2', 9: 'Q3', 12: 'Q4'}
            result['quarter'] = q_map.get(int(pm.group(3)), f'M{pm.group(3)}')

        # 흑자/적자 전환
        if '흑자전환' in text:
            result['turnaround'] = 'profit'
        elif '적자전환' in text:
            result['turnaround'] = 'loss'
        elif '적자지속' in text:
            result['turnaround'] = 'loss_cont'

        # 공시 텍스트 길이 (sentiment proxy)
        result['text_length'] = len(text)

    except:
        pass
    return result

merged = orig.merge(done_texts, on='rcept_no', how='left')
parsed = merged['text'].apply(parse_earnings)
parsed_df = pd.json_normalize(parsed)
for col in parsed_df.columns:
    merged[col] = parsed_df[col].values

p(f"매출액: {merged['revenue'].notna().sum()}/{len(merged)}")
p(f"영업이익: {merged['op_profit'].notna().sum()}/{len(merged)}")
p(f"순이익: {merged.get('net_income', pd.Series()).notna().sum()}/{len(merged)}")
p(f"매출 YoY: {merged['revenue_yoy'].notna().sum()}/{len(merged)}")
p(f"영업이익 YoY: {merged['op_profit_yoy'].notna().sum()}/{len(merged)}")
p(f"분기: {merged['quarter'].notna().sum()}/{len(merged)}")

# =====================
# 4. 주가 + CAR
# =====================
p("\n=== 4. 주가 수집 + CAR ===")

# KOSPI index
try:
    kospi = fdr.DataReader('KS11', '2019-12-01', '2026-03-04')
    kospi['mkt_ret'] = kospi['Close'].pct_change()
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

                car_list.append({
                    'rcept_no': row['rcept_no'],
                    'ret_1d': round(r1*100, 4), 'ret_3d': round(r3*100, 4), 'ret_5d': round(r5*100, 4),
                    'car_1d': round(c1*100, 4), 'car_3d': round(c3*100, 4), 'car_5d': round(c5*100, 4),
                    'pre_vol_20d': round(vol20*100, 4) if not np.isnan(vol20) else np.nan,
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
p(f"\nCAR 매칭: {len(car_df)}건")

final = merged.merge(car_df, on='rcept_no', how='left')

# =====================
# 5. Feature engineering
# =====================
p("\n=== 5. Feature engineering ===")

# Surprise direction
final['surprise_pos'] = (final['op_profit_yoy'] > 0).astype(int) if 'op_profit_yoy' in final.columns else 0
final['surprise_neg'] = (final['op_profit_yoy'] < 0).astype(int) if 'op_profit_yoy' in final.columns else 0

# Magnitude bins
if 'op_profit_yoy' in final.columns:
    final['surprise_magnitude'] = pd.cut(
        final['op_profit_yoy'],
        bins=[-np.inf, -50, -20, 0, 20, 50, np.inf],
        labels=['large_neg', 'med_neg', 'small_neg', 'small_pos', 'med_pos', 'large_pos']
    )

# 연도, 월
final['disc_year'] = pd.to_datetime(final['rcept_dt']).dt.year
final['disc_month'] = pd.to_datetime(final['rcept_dt']).dt.month

# =====================
# 6. 저장 + 통계
# =====================
final.to_pickle(f'{DATA_DIR}/dataset_final.pkl')
final.to_csv(f'{DATA_DIR}/dataset_final.csv', index=False, encoding='utf-8-sig')

p(f"\n{'='*50}")
p(f"=== 최종 데이터셋 ===")
p(f"{'='*50}")
p(f"총 {len(final)}건, 기업 {final['company_name'].nunique()}개")
p(f"컬럼 ({len(final.columns)}): {final.columns.tolist()}")

p(f"\n--- 핵심 통계 ---")
for col in ['revenue', 'op_profit', 'revenue_yoy', 'op_profit_yoy', 'car_1d', 'car_3d', 'car_5d']:
    if col in final.columns:
        s = final[col].dropna()
        p(f"  {col}: n={len(s)}, mean={s.mean():.2f}, std={s.std():.2f}")

# CAR by surprise
if 'car_1d' in final.columns and 'op_profit_yoy' in final.columns:
    p(f"\n--- CAR by Surprise Direction ---")
    pos = final[final['surprise_pos']==1]['car_1d'].dropna()
    neg = final[final['surprise_neg']==1]['car_1d'].dropna()
    p(f"  긍정(YoY>0) CAR[0,1]: mean={pos.mean():.3f}%, n={len(pos)}")
    p(f"  부정(YoY<0) CAR[0,1]: mean={neg.mean():.3f}%, n={len(neg)}")

    if 'surprise_magnitude' in final.columns:
        p(f"\n--- CAR by Surprise Magnitude ---")
        for cat in ['large_neg','med_neg','small_neg','small_pos','med_pos','large_pos']:
            sub = final[final['surprise_magnitude']==cat]['car_1d'].dropna()
            if len(sub) > 0:
                p(f"  {cat}: mean={sub.mean():.3f}%, n={len(sub)}")

p(f"\n✅ 저장 완료: {DATA_DIR}/dataset_final.pkl")
p(f"✅ 저장 완료: {DATA_DIR}/dataset_final.csv")
