"""
Step 3: 나머지 597건 텍스트 수집 + 전체 파싱 + 주가 매칭
"""
import OpenDartReader
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time, os, re, json
from datetime import datetime, timedelta

DART_API_KEY = '345beacedad4863cfdbbbb7f565ff85e0b3cb495'
dart = OpenDartReader(DART_API_KEY)
DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'

# ============================
# 1. 나머지 텍스트 수집
# ============================
print("=== 1. 데이터 로드 ===")
earnings = pd.read_pickle(f'{DATA_DIR}/earnings_30.pkl')
e = earnings[earnings['report_nm'].str.contains('영업.*실적.*공정공시', regex=True, na=False)]
orig = e[~e['report_nm'].str.contains('기재정정', na=False)].copy().reset_index(drop=True)
print(f"영업실적 공정공시 (정정 제외): {len(orig)}건")

# 이미 수집한 텍스트
texts_done = pd.read_pickle(f'{DATA_DIR}/texts_partial_600.pkl')
done_rcept = set(texts_done['rcept_no'])
print(f"이미 수집: {len(done_rcept)}건")

# 남은 건
remaining = orig[~orig['rcept_no'].isin(done_rcept)]
print(f"남은 건: {len(remaining)}건")

print("\n=== 2. 나머지 텍스트 수집 ===")
new_texts = []
for i, (idx, row) in enumerate(remaining.iterrows()):
    rcept_no = row['rcept_no']
    try:
        doc = dart.document(rcept_no)
        if doc:
            soup = BeautifulSoup(doc, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            new_texts.append({
                'rcept_no': rcept_no,
                'text': text,
                'text_len': len(text)
            })
        else:
            new_texts.append({'rcept_no': rcept_no, 'text': '', 'text_len': 0})
    except Exception as e:
        new_texts.append({'rcept_no': rcept_no, 'text': f'ERROR: {e}', 'text_len': 0})

    if (i + 1) % 100 == 0:
        # 중간 저장
        partial = pd.DataFrame(new_texts)
        partial.to_pickle(f'{DATA_DIR}/texts_remaining_{i+1}.pkl')
        print(f"  {i+1}/{len(remaining)} 완료 (중간 저장)")

    time.sleep(0.3)

new_texts_df = pd.DataFrame(new_texts)
print(f"\n신규 수집: {len(new_texts_df)}건, 성공: {(new_texts_df['text_len'] > 0).sum()}건")

# 전체 합치기
all_texts = pd.concat([texts_done, new_texts_df], ignore_index=True)
all_texts.to_pickle(f'{DATA_DIR}/texts_all.pkl')
print(f"전체 텍스트: {len(all_texts)}건")

# ============================
# 3. 숫자 파싱 (개선된 버전)
# ============================
print("\n=== 3. 숫자 파싱 ===")

def parse_earnings_v2(text):
    """개선된 실적 숫자 파싱"""
    result = {}
    if not text or len(text) < 50:
        return result

    try:
        # 매출액 당해실적 (숫자)
        # Pattern: "매출액 당해실적 938,374" 또는 "매출액(당해실적) 938,374"
        rev_patterns = [
            r'매출액\s*(?:\()?당해실적(?:\))?\s*([-\d,\.]+)',
            r'매출액\s+당해실적\s+([-\d,\.]+)',
        ]
        for pat in rev_patterns:
            m = re.search(pat, text)
            if m:
                val = m.group(1).replace(',', '')
                if val.replace('.', '').replace('-', '').isdigit():
                    result['revenue'] = float(val)
                break

        # 영업이익 당해실적
        op_patterns = [
            r'영업이익\s*(?:\()?당해실적(?:\))?\s*([-\d,\.]+)',
            r'영업이익\s+당해실적\s+([-\d,\.]+)',
        ]
        for pat in op_patterns:
            m = re.search(pat, text)
            if m:
                val = m.group(1).replace(',', '')
                if val.replace('.', '').replace('-', '').isdigit():
                    result['op_profit'] = float(val)
                break

        # 전기대비 증감율 (매출액)
        # Pattern: 매출액 ... 증감율(%) ... 숫자
        rev_yoy_patterns = [
            r'매출액.*?전년동기대비\s*(?:증감율\(%\))?\s*([-\d,\.]+)',
            r'전년동기대비.*?매출액.*?([-\d,\.]+)\s*%?',
        ]
        for pat in rev_yoy_patterns:
            m = re.search(pat, text, re.DOTALL)
            if m:
                val = m.group(1).replace(',', '')
                try:
                    fval = float(val)
                    if -500 < fval < 10000:  # reasonable YoY range
                        result['revenue_yoy'] = fval
                except:
                    pass
                break

        # 전기대비 증감율 (영업이익)
        op_yoy_patterns = [
            r'영업이익.*?전년동기대비\s*(?:증감율\(%\))?\s*([-\d,\.]+)',
        ]
        for pat in op_yoy_patterns:
            m = re.search(pat, text, re.DOTALL)
            if m:
                val = m.group(1).replace(',', '')
                try:
                    fval = float(val)
                    if -500 < fval < 50000:
                        result['op_profit_yoy'] = fval
                except:
                    pass
                break

        # 실적기간 추출 (분기 식별)
        period_match = re.search(r'당기실적\s+(\d{4})-(\d{2})-\d{2}\s*~\s*(\d{4})-(\d{2})-(\d{2})', text)
        if period_match:
            y, m_start, y2, m_end, d_end = period_match.groups()
            result['period_year'] = int(y)
            result['period_month_start'] = int(m_start)
            result['period_month_end'] = int(m_end)
            # 분기 식별
            q_map = {3: 'Q1', 6: 'Q2', 9: 'Q3', 12: 'Q4'}
            result['quarter'] = q_map.get(int(m_end), f'M{m_end}')

        # 흑자/적자 전환
        if '흑자전환' in text:
            result['turnaround'] = 'profit'
        elif '적자전환' in text:
            result['turnaround'] = 'loss'
        elif '적자지속' in text:
            result['turnaround'] = 'loss_cont'

    except Exception as e:
        result['parse_error'] = str(e)

    return result

# 텍스트와 공시 데이터 합치기
merged = orig.merge(all_texts, on='rcept_no', how='left')

# 파싱
parsed = merged['text'].apply(parse_earnings_v2)
parsed_df = pd.json_normalize(parsed)
for col in parsed_df.columns:
    merged[col] = parsed_df[col].values

print(f"매출액 추출: {merged['revenue'].notna().sum()}/{len(merged)}")
print(f"영업이익 추출: {merged['op_profit'].notna().sum()}/{len(merged)}")
print(f"매출 YoY: {merged['revenue_yoy'].notna().sum()}/{len(merged)}")
print(f"영업이익 YoY: {merged['op_profit_yoy'].notna().sum()}/{len(merged)}")
print(f"분기 식별: {merged['quarter'].notna().sum()}/{len(merged)}")

# ============================
# 4. 주가 데이터 + CAR 계산
# ============================
print("\n=== 4. 주가 데이터 수집 + CAR ===")

# KOSPI 인덱스 (시장 수익률 차감용)
print("KOSPI 지수 수집...")
try:
    kospi_idx = fdr.DataReader('KS11', '2019-12-01', '2026-03-04')
    kospi_idx['market_ret'] = kospi_idx['Close'].pct_change()
    print(f"  KOSPI 인덱스: {len(kospi_idx)} trading days")
except Exception as e:
    print(f"  KOSPI 인덱스 실패: {e}")
    kospi_idx = None

# 종목별 주가 수집
unique_codes = merged['stock_code'].unique()
print(f"\n종목 {len(unique_codes)}개 주가 수집...")

car_results = []
for code in unique_codes:
    subset = merged[merged['stock_code'] == code]
    company = subset.iloc[0]['company_name']
    try:
        prices = fdr.DataReader(code, '2019-12-01', '2026-03-04')
        prices['stock_ret'] = prices['Close'].pct_change()

        for _, row in subset.iterrows():
            try:
                disc_date = pd.to_datetime(row['rcept_dt'])
                trading_days = prices.index
                post_days = trading_days[trading_days >= disc_date]
                pre_days = trading_days[trading_days < disc_date]

                if len(post_days) < 5 or len(pre_days) < 1:
                    continue

                pre_close = prices.loc[pre_days[-1], 'Close']

                # Raw returns
                ret_1d = (prices.loc[post_days[0], 'Close'] - pre_close) / pre_close
                ret_3d = (prices.loc[post_days[min(2, len(post_days)-1)], 'Close'] - pre_close) / pre_close
                ret_5d = (prices.loc[post_days[min(4, len(post_days)-1)], 'Close'] - pre_close) / pre_close

                # Market-adjusted returns (CAR)
                car_1d = ret_1d
                car_3d = ret_3d
                car_5d = ret_5d
                if kospi_idx is not None:
                    try:
                        m_post = kospi_idx.index[kospi_idx.index >= disc_date]
                        m_pre = kospi_idx.index[kospi_idx.index < disc_date]
                        if len(m_post) >= 5 and len(m_pre) >= 1:
                            m_pre_close = kospi_idx.loc[m_pre[-1], 'Close']
                            mkt_1d = (kospi_idx.loc[m_post[0], 'Close'] - m_pre_close) / m_pre_close
                            mkt_3d = (kospi_idx.loc[m_post[min(2, len(m_post)-1)], 'Close'] - m_pre_close) / m_pre_close
                            mkt_5d = (kospi_idx.loc[m_post[min(4, len(m_post)-1)], 'Close'] - m_pre_close) / m_pre_close
                            car_1d = ret_1d - mkt_1d
                            car_3d = ret_3d - mkt_3d
                            car_5d = ret_5d - mkt_5d
                    except:
                        pass

                # Pre-event volatility (20-day)
                if len(pre_days) >= 20:
                    pre_20 = prices.loc[pre_days[-20:], 'stock_ret'].std()
                else:
                    pre_20 = np.nan

                car_results.append({
                    'rcept_no': row['rcept_no'],
                    'ret_1d': round(ret_1d * 100, 4),
                    'ret_3d': round(ret_3d * 100, 4),
                    'ret_5d': round(ret_5d * 100, 4),
                    'car_1d': round(car_1d * 100, 4),
                    'car_3d': round(car_3d * 100, 4),
                    'car_5d': round(car_5d * 100, 4),
                    'pre_vol_20d': round(pre_20 * 100, 4) if not np.isnan(pre_20) else np.nan,
                    'pre_close': pre_close,
                })
            except:
                pass

        print(f"  {company}: {len([r for r in car_results if r['rcept_no'] in set(subset['rcept_no'])])}건 매칭")
    except Exception as e:
        print(f"  {company}: ERROR - {e}")
    time.sleep(0.2)

car_df = pd.DataFrame(car_results)
print(f"\nCAR 계산 완료: {len(car_df)}건")

# 최종 합치기
final = merged.merge(car_df, on='rcept_no', how='left')

# ============================
# 5. 저장
# ============================
final.to_pickle(f'{DATA_DIR}/dataset_final.pkl')
final.to_csv(f'{DATA_DIR}/dataset_final.csv', index=False, encoding='utf-8-sig')
print(f"\n=== 최종 데이터셋 ===")
print(f"총 {len(final)}건, 컬럼: {len(final.columns)}개")
print(f"Columns: {final.columns.tolist()}")

# 기초 통계
print(f"\n--- 기초 통계 ---")
for col in ['revenue', 'op_profit', 'revenue_yoy', 'op_profit_yoy', 'car_1d', 'car_3d', 'car_5d']:
    if col in final.columns:
        s = final[col].dropna()
        print(f"  {col}: n={len(s)}, mean={s.mean():.2f}, std={s.std():.2f}, min={s.min():.2f}, max={s.max():.2f}")

print(f"\n분기별 분포:")
if 'quarter' in final.columns:
    print(final['quarter'].value_counts())

# Surprise indicator
if 'op_profit_yoy' in final.columns:
    final['surprise_pos'] = (final['op_profit_yoy'] > 0).astype(int)
    final['surprise_neg'] = (final['op_profit_yoy'] < 0).astype(int)
    print(f"\n영업이익 서프라이즈:")
    print(f"  긍정 (YoY>0): {final['surprise_pos'].sum()}")
    print(f"  부정 (YoY<0): {final['surprise_neg'].sum()}")

    # CAR by surprise
    if 'car_1d' in final.columns:
        pos = final[final['surprise_pos'] == 1]['car_1d'].dropna()
        neg = final[final['surprise_neg'] == 1]['car_1d'].dropna()
        print(f"\n  긍정 서프라이즈 CAR[0,1]: mean={pos.mean():.3f}%, n={len(pos)}")
        print(f"  부정 서프라이즈 CAR[0,1]: mean={neg.mean():.3f}%, n={len(neg)}")

final.to_pickle(f'{DATA_DIR}/dataset_final.pkl')
print("\n✅ 완료!")
