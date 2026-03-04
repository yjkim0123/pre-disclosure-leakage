#!/usr/bin/env python3
"""
Step 5: KOSPI 시장보정 CAR + 텍스트 감성분석
"""
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import re, time, warnings
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
import json

warnings.filterwarnings('ignore')
DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
def p(msg): print(msg, flush=True)

# =====================
# 1. Load
# =====================
p("=== 1. 데이터 로드 ===")
df = pd.read_pickle(f'{DATA_DIR}/dataset_final.pkl')
p(f"데이터: {len(df)}건")

# =====================
# 2. KOSPI 시장보정 재시도
# =====================
p("\n=== 2. KOSPI 시장보정 ===")

# Try multiple tickers for KOSPI
kospi = None
for ticker in ['KS11', 'KOSPI', '1001', '^KS11']:
    try:
        k = fdr.DataReader(ticker, '2019-12-01', '2026-03-04')
        if k is not None and len(k) > 100:
            kospi = k
            p(f"KOSPI 로드 성공: ticker={ticker}, {len(k)} days")
            break
    except Exception as e:
        p(f"  {ticker}: {e}")

# If FDR fails, try Yahoo Finance via pandas_datareader or direct URL
if kospi is None:
    try:
        import yfinance as yf
        kospi = yf.download('^KS11', start='2019-12-01', end='2026-03-04', progress=False)
        if len(kospi) > 100:
            p(f"KOSPI via yfinance: {len(kospi)} days")
    except:
        p("yfinance도 실패")

# Last resort: use ETF KODEX200 as proxy
if kospi is None:
    try:
        kospi = fdr.DataReader('069500', '2019-12-01', '2026-03-04')  # KODEX 200
        if kospi is not None and len(kospi) > 100:
            p(f"KODEX200 프록시 사용: {len(kospi)} days")
        else:
            kospi = None
    except:
        pass

if kospi is not None:
    p(f"시장지수 확보: {len(kospi)} trading days")
    
    # Recalculate CAR with market adjustment
    car_list = []
    for code in df['stock_code'].unique():
        sub = df[df['stock_code'] == code]
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
                    
                    # Market returns
                    mp = kospi.index[kospi.index >= dd]
                    mpr = kospi.index[kospi.index < dd]
                    if len(mp) >= 5 and len(mpr) >= 1:
                        mpc = kospi.loc[mpr[-1], 'Close']
                        m1 = (kospi.loc[mp[0], 'Close'] - mpc) / mpc
                        m3 = (kospi.loc[mp[min(2, len(mp)-1)], 'Close'] - mpc) / mpc
                        m5 = (kospi.loc[mp[min(4, len(mp)-1)], 'Close'] - mpc) / mpc
                        c1, c3, c5 = r1-m1, r3-m3, r5-m5
                    else:
                        c1, c3, c5 = r1, r3, r5
                    
                    car_list.append({
                        'rcept_no': row['rcept_no'],
                        'mcar_1d': round(c1*100, 4),
                        'mcar_3d': round(c3*100, 4),
                        'mcar_5d': round(c5*100, 4),
                    })
                    matched += 1
                except:
                    pass
            p(f"  {name}: {matched}/{len(sub)}")
        except:
            pass
        time.sleep(0.1)
    
    car_df = pd.DataFrame(car_list)
    df = df.merge(car_df, on='rcept_no', how='left')
    p(f"\n시장보정 CAR 매칭: {df['mcar_1d'].notna().sum()}/{len(df)}")
    
    # Compare raw vs market-adjusted
    p(f"\n--- Raw vs Market-Adjusted CAR ---")
    for w in [('car_1d', 'mcar_1d'), ('car_3d', 'mcar_3d'), ('car_5d', 'mcar_5d')]:
        raw = df[w[0]].dropna()
        adj = df[w[1]].dropna()
        p(f"  {w[0]}: raw={raw.mean():.3f}%, adjusted={adj.mean():.3f}%")
else:
    p("⚠️ 시장지수 확보 실패 — raw CAR 사용")
    df['mcar_1d'] = df['car_1d']
    df['mcar_3d'] = df['car_3d']
    df['mcar_5d'] = df['car_5d']

# =====================
# 3. 텍스트 감성분석 (Rule-based + Statistical)
# =====================
p("\n=== 3. 텍스트 감성분석 ===")

# 3a. 긍정/부정 키워드 사전 (한국어 금융)
positive_words = [
    '증가', '상승', '성장', '개선', '호조', '흑자', '흑자전환', '최대', '최고',
    '확대', '신장', '회복', '강화', '호실적', '상향', '돌파', '역대',
    '기록', '달성', '순증', '순이익', '호재', '반등', '급증'
]
negative_words = [
    '감소', '하락', '악화', '부진', '적자', '적자전환', '적자지속', '축소',
    '위축', '둔화', '하향', '손실', '감익', '역성장', '감액', '차질',
    '리스크', '우려', '저조', '급감', '폭락'
]

def sentiment_score(text):
    """Simple lexicon-based sentiment"""
    if not text or len(str(text)) < 50:
        return 0, 0, 0
    text = str(text)
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    total = pos_count + neg_count
    if total == 0:
        return 0, 0, 0
    score = (pos_count - neg_count) / total  # [-1, 1]
    return score, pos_count, neg_count

df['sentiment'], df['pos_words'], df['neg_words'] = zip(*df['text'].apply(sentiment_score))
df['sentiment_label'] = pd.cut(df['sentiment'], bins=[-1.01, -0.33, 0.33, 1.01],
                                labels=['negative', 'neutral', 'positive'])

p(f"감성 분포:")
p(str(df['sentiment_label'].value_counts()))
p(f"\n감성 통계: mean={df['sentiment'].mean():.3f}, std={df['sentiment'].std():.3f}")

# 3b. Sentiment vs CAR
target_car = 'mcar_1d'
p(f"\n--- Sentiment vs {target_car} ---")
for label in ['negative', 'neutral', 'positive']:
    sub = df[df['sentiment_label']==label][target_car].dropna()
    if len(sub) > 0:
        p(f"  {label}: mean={sub.mean():.3f}%, n={len(sub)}")

# 3c. Text complexity features
df['avg_sentence_len'] = df['text'].apply(
    lambda x: np.mean([len(s) for s in str(x).split('.') if len(s) > 5]) if x else 0
)
df['num_count'] = df['text'].apply(
    lambda x: len(re.findall(r'\d+', str(x))) if x else 0
)

# =====================
# 4. Enhanced Feature Set + ML
# =====================
p("\n=== 4. Enhanced ML ===")

# Fix YoY outliers
for col in ['revenue_yoy', 'op_profit_yoy', 'net_income_yoy']:
    if col in df.columns:
        df.loc[df[col].abs() > 1000, col] = np.nan

# Quarter inference
def infer_quarter(row):
    month = int(str(row['rcept_dt'])[4:6])
    if month in [1, 2, 3]: return 'Q4'
    elif month in [4, 5, 6]: return 'Q1'
    elif month in [7, 8, 9]: return 'Q2'
    else: return 'Q3'

df['quarter_inferred'] = df.apply(infer_quarter, axis=1)

# Enhanced features
feature_cols = []
for col in ['op_profit_yoy', 'revenue_yoy', 'net_income_yoy',
            'op_profit_qoq', 'revenue_qoq',
            'volume_ratio', 'pre_vol_20d', 'text_length',
            'sentiment', 'pos_words', 'neg_words',
            'avg_sentence_len', 'num_count']:
    if col in df.columns:
        feature_cols.append(col)

# Dummies
df['is_consolidated'] = df['consolidated'].fillna(False).astype(int)
df['is_preliminary'] = df.get('is_preliminary', pd.Series(0, index=df.index)).fillna(0).astype(int)
df['has_turnaround'] = df['op_turn'].notna().astype(int) if 'op_turn' in df.columns else 0
df['rev_profit_div'] = df['rev_profit_diverge'].fillna(0).astype(int) if 'rev_profit_diverge' in df.columns else 0

for q in ['Q1', 'Q2', 'Q3']:
    df[f'is_{q.lower()}'] = (df['quarter_inferred'] == q).astype(int)

feature_cols += ['is_consolidated', 'is_preliminary', 'has_turnaround', 'rev_profit_div',
                 'is_q1', 'is_q2', 'is_q3']

# Model data
model_df = df[feature_cols + [target_car, 'stock_code']].dropna(subset=feature_cols + [target_car])
X = model_df[feature_cols].values
y = model_df[target_car].values
groups = model_df['stock_code'].values

p(f"Enhanced model: {len(model_df)} samples, {len(feature_cols)} features")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gkf = GroupKFold(n_splits=5)

# Regression
p(f"\n--- Regression (predict {target_car}) ---")
p(f"{'Model':<12} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
p("-" * 40)

results = {}
for name, model in [
    ('OLS', LinearRegression()),
    ('Lasso', Lasso(alpha=0.01)),
    ('RF', RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42)),
    ('GBM', GradientBoostingRegressor(n_estimators=300, max_depth=4, min_samples_leaf=10, learning_rate=0.05, random_state=42)),
]:
    use_X = X_scaled if name in ['OLS', 'Lasso'] else X
    r2 = cross_val_score(model, use_X, y, cv=gkf, groups=groups, scoring='r2').mean()
    mae = -cross_val_score(model, use_X, y, cv=gkf, groups=groups, scoring='neg_mean_absolute_error').mean()
    rmse = np.sqrt(-cross_val_score(model, use_X, y, cv=gkf, groups=groups, scoring='neg_mean_squared_error').mean())
    results[name] = {'r2': r2, 'mae': mae, 'rmse': rmse}
    p(f"  {name:<12} {r2:>7.4f} {mae:>7.3f} {rmse:>7.3f}")

best = max(results, key=lambda k: results[k]['r2'])
p(f"\nBest: {best} (R²={results[best]['r2']:.4f})")

# Feature importance
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
p(f"\nRF Feature Importance:")
for feat, val in imp.head(12).items():
    p(f"  {feat}: {val:.4f}")

# Classification
p(f"\n--- Classification ---")
y_cls = (y > 0).astype(int)
for name, model in [
    ('Logistic', LogisticRegression(max_iter=1000, random_state=42)),
    ('RF', RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42)),
]:
    use_X = X_scaled if name == 'Logistic' else X
    acc = cross_val_score(model, use_X, y_cls, cv=gkf, groups=groups, scoring='accuracy').mean()
    f1 = cross_val_score(model, use_X, y_cls, cv=gkf, groups=groups, scoring='f1').mean()
    auc = cross_val_score(model, use_X, y_cls, cv=gkf, groups=groups, scoring='roc_auc').mean()
    p(f"  {name:<12} acc={acc:.4f}  f1={f1:.4f}  auc={auc:.4f}")

# =====================
# 5. Cross-sectional Regression (OLS with controls)
# =====================
p("\n=== 5. Cross-sectional Regression ===")
import statsmodels.api as sm

# Prepare for OLS with controls
reg_cols = ['op_profit_yoy', 'revenue_yoy', 'volume_ratio', 'pre_vol_20d',
            'sentiment', 'is_consolidated', 'has_turnaround']
reg_df = df[reg_cols + [target_car]].dropna()

if len(reg_df) > 50:
    Xr = sm.add_constant(reg_df[reg_cols])
    model_ols = sm.OLS(reg_df[target_car], Xr).fit(cov_type='HC1')  # robust SE
    p(model_ols.summary().as_text())
else:
    p("회귀분석 데이터 부족")

# =====================
# 6. 최종 저장
# =====================
df.to_pickle(f'{DATA_DIR}/dataset_final_v3.pkl')
df.to_csv(f'{DATA_DIR}/dataset_final_v3.csv', index=False, encoding='utf-8-sig')

# Summary
summary = {
    'n': len(df),
    'n_companies': int(df['company_name'].nunique()),
    'market_adjusted': kospi is not None,
    'mcar_1d_mean': round(df['mcar_1d'].dropna().mean(), 3),
    'mcar_1d_tstat': round(stats.ttest_1samp(df['mcar_1d'].dropna(), 0)[0], 2),
    'pos_vs_neg_pval': round(stats.ttest_ind(
        df[df['op_profit_yoy']>0]['mcar_1d'].dropna(),
        df[df['op_profit_yoy']<0]['mcar_1d'].dropna()
    )[1], 4) if 'op_profit_yoy' in df.columns else None,
    'best_ml_r2': round(results[best]['r2'], 4),
    'sentiment_corr': round(df[['sentiment', 'mcar_1d']].dropna().corr().iloc[0,1], 4),
}
with open(f'{DATA_DIR}/analysis_v2.json', 'w') as f:
    json.dump(summary, f, indent=2)

p(f"\n✅ 완료! {DATA_DIR}/dataset_final_v3.pkl")
p(json.dumps(summary, indent=2))
