"""
DART 실적 공시 수집 + 주가 반응 분석 파이프라인
Step 1: KOSPI 상장사 실적 공시 수집
"""
import OpenDartReader
import FinanceDataReader as fdr
import pandas as pd
import time
import json
import os
from datetime import datetime, timedelta

DART_API_KEY = '345beacedad4863cfdbbbb7f565ff85e0b3cb495'
dart = OpenDartReader(DART_API_KEY)

DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
os.makedirs(DATA_DIR, exist_ok=True)

# 1. KOSPI 상장사 리스트
print("=== KOSPI 상장사 리스트 수집 ===")
kospi = fdr.StockListing('KOSPI')
print(f"KOSPI 종목 수: {len(kospi)}")
print(kospi[['Code', 'Name', 'Market']].head(10))

# 시가총액 상위 100개 기업 선택
kospi_top = kospi.nlargest(100, 'Marcap') if 'Marcap' in kospi.columns else kospi.head(100)
print(f"\n상위 100개 기업 선택")

# 2. 실적 공시 수집 (영업실적 잠정/공정공시)
print("\n=== 실적 공시 수집 (2020-2025) ===")
all_earnings = []
errors = []

for idx, row in kospi_top.iterrows():
    code = row['Code']
    name = row['Name']
    try:
        # 5년치 공시
        df = dart.list(code, start='2020-01-01', end='2026-03-04')
        if df is not None and len(df) > 0:
            # 실적 관련 공시 필터
            earnings_keywords = ['영업', '실적', '매출', '손익', '재무', '배당']
            mask = df['report_nm'].apply(
                lambda x: any(k in str(x) for k in earnings_keywords)
            )
            earnings = df[mask].copy()
            earnings['stock_code'] = code
            earnings['company_name'] = name
            all_earnings.append(earnings)
            print(f"  [{idx+1:3d}/100] {name}: 전체 {len(df)}건, 실적 관련 {len(earnings)}건")
        else:
            print(f"  [{idx+1:3d}/100] {name}: 공시 없음")
    except Exception as e:
        errors.append({'code': code, 'name': name, 'error': str(e)})
        print(f"  [{idx+1:3d}/100] {name}: ERROR - {e}")
    
    time.sleep(0.5)  # API rate limit

# 결합
if all_earnings:
    combined = pd.concat(all_earnings, ignore_index=True)
    print(f"\n=== 수집 완료 ===")
    print(f"총 실적 관련 공시: {len(combined)}건")
    print(f"기업 수: {combined['company_name'].nunique()}")
    print(f"\n공시유형 TOP 20:")
    print(combined['report_nm'].str[:25].value_counts().head(20))
    
    # 저장
    combined.to_pickle(f'{DATA_DIR}/earnings_disclosures.pkl')
    combined.to_csv(f'{DATA_DIR}/earnings_disclosures.csv', index=False, encoding='utf-8-sig')
    print(f"\n저장: {DATA_DIR}/earnings_disclosures.pkl")
    print(f"저장: {DATA_DIR}/earnings_disclosures.csv")
else:
    print("수집된 데이터 없음!")

if errors:
    print(f"\n에러 {len(errors)}건:")
    for e in errors[:5]:
        print(f"  {e['name']}: {e['error']}")
