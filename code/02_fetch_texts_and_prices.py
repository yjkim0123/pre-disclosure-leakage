"""
Step 2: 영업실적 공정공시 원문 수집 + 주가 데이터 매칭
"""
import OpenDartReader
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time, os, json, re
from datetime import datetime, timedelta

DART_API_KEY = '345beacedad4863cfdbbbb7f565ff85e0b3cb495'
dart = OpenDartReader(DART_API_KEY)
DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'

# 1. 영업실적 공정공시 로드
df = pd.read_pickle(f'{DATA_DIR}/earnings_30.pkl')
earnings = df[df['report_nm'].str.contains('영업.*실적.*공정공시', regex=True, na=False)].copy()
# 기재정정 제외 (원본만)
earnings_orig = earnings[~earnings['report_nm'].str.contains('기재정정', na=False)].copy()
print(f"영업실적 공정공시 (정정 제외): {len(earnings_orig)}건")

# 2. 원문 텍스트 수집
print("\n=== 원문 수집 ===")
texts = []
for i, (idx, row) in enumerate(earnings_orig.iterrows()):
    rcept_no = row['rcept_no']
    try:
        doc = dart.document(rcept_no)
        if doc:
            soup = BeautifulSoup(doc, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            texts.append({
                'rcept_no': rcept_no,
                'text': text,
                'text_len': len(text)
            })
        else:
            texts.append({'rcept_no': rcept_no, 'text': '', 'text_len': 0})
    except Exception as e:
        texts.append({'rcept_no': rcept_no, 'text': f'ERROR: {e}', 'text_len': 0})
    
    if (i+1) % 50 == 0:
        print(f"  {i+1}/{len(earnings_orig)} 완료")
    time.sleep(0.3)

texts_df = pd.DataFrame(texts)
earnings_orig = earnings_orig.merge(texts_df, on='rcept_no', how='left')
print(f"\n원문 수집 완료: {(earnings_orig['text_len'] > 0).sum()}/{len(earnings_orig)}건 성공")

# 3. 숫자 파싱 (매출액, 영업이익, 증감율)
def parse_earnings_numbers(text):
    """공시 텍스트에서 실적 숫자 추출"""
    result = {}
    try:
        # 매출액 추출 (조 단위)
        rev_match = re.search(r'매출액\(당해실적\)\s*([\d,.]+)', text)
        if rev_match:
            result['revenue'] = float(rev_match.group(1).replace(',', ''))
        
        # 영업이익 추출
        op_match = re.search(r'영업이익\(당해실적\)\s*([\d,.]+)', text)
        if op_match:
            result['operating_profit'] = float(op_match.group(1).replace(',', ''))
        
        # 전년동기대비 매출 증감율
        rev_yoy = re.search(r'전년동기대비.*매출액\(당해실적\)\s*([-\d,.]+)', text, re.DOTALL)
        if rev_yoy:
            result['revenue_yoy'] = float(rev_yoy.group(1).replace(',', ''))
        
        # 전년동기대비 영업이익 증감율
        op_yoy = re.search(r'전년동기대비.*영업이익\(당해실적\)\s*([-\d,.]+)', text, re.DOTALL)
        if op_yoy:
            result['op_profit_yoy'] = float(op_yoy.group(1).replace(',', ''))
    except:
        pass
    return result

print("\n=== 숫자 파싱 ===")
parsed = earnings_orig['text'].apply(parse_earnings_numbers)
for col in ['revenue', 'operating_profit', 'revenue_yoy', 'op_profit_yoy']:
    earnings_orig[col] = parsed.apply(lambda x: x.get(col))

print(f"매출액 추출: {earnings_orig['revenue'].notna().sum()}건")
print(f"영업이익 추출: {earnings_orig['operating_profit'].notna().sum()}건")
print(f"매출 YoY: {earnings_orig['revenue_yoy'].notna().sum()}건")
print(f"영업이익 YoY: {earnings_orig['op_profit_yoy'].notna().sum()}건")

# 4. 주가 데이터 매칭 (공시일 전후 수익률)
print("\n=== 주가 데이터 수집 ===")
price_cache = {}

def get_stock_returns(stock_code, disc_date_str, window=3):
    """공시일 전후 주가 수익률 계산"""
    try:
        disc_date = pd.to_datetime(disc_date_str)
        start = disc_date - timedelta(days=30)
        end = disc_date + timedelta(days=30)
        
        cache_key = f"{stock_code}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        if cache_key not in price_cache:
            prices = fdr.DataReader(stock_code, start, end)
            price_cache[cache_key] = prices
        else:
            prices = price_cache[cache_key]
        
        if len(prices) < 5:
            return {}
        
        # 공시일 이후 가장 가까운 거래일 찾기
        trading_days = prices.index
        post_days = trading_days[trading_days >= disc_date]
        pre_days = trading_days[trading_days < disc_date]
        
        if len(post_days) < window or len(pre_days) < 1:
            return {}
        
        close = prices['Close']
        pre_close = close[pre_days[-1]]  # 공시 전일 종가
        
        # CAR (Cumulative Abnormal Return) - 간단히 raw return 사용
        ret_1d = (close[post_days[0]] - pre_close) / pre_close * 100
        ret_3d = (close[post_days[min(2, len(post_days)-1)]] - pre_close) / pre_close * 100
        
        return {
            'ret_1d': round(ret_1d, 4),
            'ret_3d': round(ret_3d, 4),
            'pre_close': pre_close
        }
    except:
        return {}

returns_data = []
unique_codes = earnings_orig['stock_code'].unique()
for code in unique_codes:
    subset = earnings_orig[earnings_orig['stock_code'] == code]
    try:
        # 전체 기간 주가 한번에
        prices = fdr.DataReader(code, '2020-01-01', '2026-03-04')
        for _, row in subset.iterrows():
            try:
                disc_date = pd.to_datetime(row['rcept_dt'])
                trading_days = prices.index
                post_days = trading_days[trading_days >= disc_date]
                pre_days = trading_days[trading_days < disc_date]
                
                if len(post_days) >= 3 and len(pre_days) >= 1:
                    close = prices['Close']
                    pre_close = close[pre_days[-1]]
                    ret_1d = (close[post_days[0]] - pre_close) / pre_close * 100
                    ret_3d = (close[post_days[min(2, len(post_days)-1)]] - pre_close) / pre_close * 100
                    returns_data.append({
                        'rcept_no': row['rcept_no'],
                        'ret_1d': round(ret_1d, 4),
                        'ret_3d': round(ret_3d, 4)
                    })
            except:
                pass
        print(f"  {row['company_name']}: 주가 매칭 완료")
    except Exception as e:
        print(f"  {code}: ERROR {e}")
    time.sleep(0.2)

returns_df = pd.DataFrame(returns_data)
if len(returns_df) > 0:
    earnings_orig = earnings_orig.merge(returns_df, on='rcept_no', how='left')
    print(f"\n주가 매칭: {earnings_orig['ret_1d'].notna().sum()}/{len(earnings_orig)}건")

# 5. 저장
earnings_orig.to_pickle(f'{DATA_DIR}/earnings_with_text_prices.pkl')
print(f"\n=== 최종 저장 ===")
print(f"총 {len(earnings_orig)}건")
print(f"컬럼: {earnings_orig.columns.tolist()}")
print(f"\n기초 통계:")
for col in ['ret_1d', 'ret_3d', 'revenue_yoy', 'op_profit_yoy']:
    if col in earnings_orig.columns:
        s = earnings_orig[col].dropna()
        print(f"  {col}: mean={s.mean():.2f}, std={s.std():.2f}, n={len(s)}")
