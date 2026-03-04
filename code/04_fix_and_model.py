#!/usr/bin/env python3
"""
Step 4: Fix data issues + ML modeling + Analysis
"""
import pandas as pd
import numpy as np
import re, warnings
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

warnings.filterwarnings('ignore')
DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
def p(msg): print(msg, flush=True)

# =====================
# 1. Load + Fix
# =====================
p("=== 1. 데이터 로드 + 수정 ===")
df = pd.read_pickle(f'{DATA_DIR}/dataset_final.pkl')
p(f"원본: {len(df)}건")

# Fix 1: quarter detection — use rcept_dt month as proxy
# 실적 공시 타이밍: Q1→4월, Q2→7월, Q3→10월, Q4→1-2월
def infer_quarter(row):
    month = int(str(row['rcept_dt'])[4:6])
    if month in [1, 2, 3]:
        return 'Q4'  # 전년도 Q4 실적
    elif month in [4, 5, 6]:
        return 'Q1'
    elif month in [7, 8, 9]:
        return 'Q2'
    elif month in [10, 11, 12]:
        return 'Q3'
    return None

df['quarter_inferred'] = df.apply(infer_quarter, axis=1)
p(f"분기 추론: {df['quarter_inferred'].value_counts().to_dict()}")

# Fix 2: YoY — cap to reasonable percentage range
# Actual YoY should be between -100% and ~10000% (극단적 경우)
# Values outside this are absolute numbers misidentified as percentages
for col in ['revenue_yoy', 'op_profit_yoy', 'net_income_yoy']:
    if col in df.columns:
        before = df[col].notna().sum()
        # If absolute value > 1000, likely an absolute number not percentage
        df.loc[df[col].abs() > 1000, col] = np.nan
        after = df[col].notna().sum()
        if before != after:
            p(f"  {col}: {before}→{after} (removed {before-after} outliers)")

# Fix 3: revenue_eok 재계산 (unit 기반)
# 억원 단위 통일
if 'unit' in df.columns:
    p(f"단위 분포: {df['unit'].value_counts().to_dict()}")

# Stats after fix
p(f"\n--- 수정 후 통계 ---")
for col in ['revenue_yoy', 'op_profit_yoy', 'car_1d', 'car_3d']:
    if col in df.columns:
        s = df[col].dropna()
        p(f"  {col}: n={len(s)}, mean={s.mean():.3f}, std={s.std():.2f}, "
          f"[{s.quantile(0.01):.1f}, {s.quantile(0.99):.1f}]")

# =====================
# 2. Event Study Analysis
# =====================
p("\n=== 2. Event Study ===")

# 2a. Overall CAR significance
for window in ['car_1d', 'car_3d', 'car_5d']:
    if window in df.columns:
        vals = df[window].dropna()
        t_stat, p_val = stats.ttest_1samp(vals, 0)
        p(f"  {window}: mean={vals.mean():.3f}%, t={t_stat:.2f}, p={p_val:.4f}")

# 2b. CAR by surprise direction (t-test)
if 'op_profit_yoy' in df.columns and 'car_1d' in df.columns:
    pos = df[df['op_profit_yoy'] > 0]['car_1d'].dropna()
    neg = df[df['op_profit_yoy'] < 0]['car_1d'].dropna()
    t, p_val = stats.ttest_ind(pos, neg)
    p(f"\n  Positive vs Negative surprise:")
    p(f"    Positive: {pos.mean():.3f}% (n={len(pos)})")
    p(f"    Negative: {neg.mean():.3f}% (n={len(neg)})")
    p(f"    t-test: t={t:.2f}, p={p_val:.4f}")

# 2c. CAR by magnitude (ANOVA)
if 'surprise_mag' in df.columns and 'car_1d' in df.columns:
    groups = []
    labels = []
    for cat in ['large_neg','med_neg','small_neg','small_pos','med_pos','large_pos']:
        sub = df[df['surprise_mag']==cat]['car_1d'].dropna()
        if len(sub) > 5:
            groups.append(sub)
            labels.append(cat)
    if len(groups) >= 3:
        f_stat, p_val = stats.f_oneway(*groups)
        p(f"\n  ANOVA across magnitude groups: F={f_stat:.2f}, p={p_val:.4f}")

# 2d. Turnaround effect
if 'op_turn' in df.columns:
    p(f"\n  Turnaround effects:")
    for turn in ['profit_turn', 'loss_turn']:
        sub = df[df['op_turn']==turn]['car_1d'].dropna()
        if len(sub) > 3:
            t_stat, p_val = stats.ttest_1samp(sub, 0)
            p(f"    {turn}: mean={sub.mean():.3f}%, t={t_stat:.2f}, p={p_val:.4f}, n={len(sub)}")

# 2e. Volume reaction
if 'volume_ratio' in df.columns:
    p(f"\n  Volume reaction:")
    for d in ['positive', 'negative']:
        if 'surprise_dir' in df.columns:
            sub = df[df['surprise_dir']==d]['volume_ratio'].dropna()
            p(f"    {d}: mean vol ratio={sub.mean():.2f}x, n={len(sub)}")

# 2f. Quarter seasonality
p(f"\n  CAR by quarter:")
for q in ['Q1', 'Q2', 'Q3', 'Q4']:
    sub = df[df['quarter_inferred']==q]['car_1d'].dropna()
    p(f"    {q}: mean={sub.mean():.3f}%, n={len(sub)}")

# =====================
# 3. Regression: Predict CAR
# =====================
p("\n=== 3. ML: CAR Prediction ===")

# Feature matrix
feature_cols = []
for col in ['op_profit_yoy', 'revenue_yoy', 'net_income_yoy',
            'op_profit_qoq', 'revenue_qoq',
            'volume_ratio', 'pre_vol_20d', 'text_length']:
    if col in df.columns:
        feature_cols.append(col)

# Dummy features
df['is_q4'] = (df['quarter_inferred'] == 'Q4').astype(int)
df['is_consolidated'] = df['consolidated'].fillna(False).astype(int) if 'consolidated' in df.columns else 0
df['is_preliminary'] = df['is_preliminary'].fillna(False).astype(int) if 'is_preliminary' in df.columns else 0
df['has_turnaround'] = df['op_turn'].notna().astype(int) if 'op_turn' in df.columns else 0
df['rev_profit_div'] = df['rev_profit_diverge'].fillna(0).astype(int) if 'rev_profit_diverge' in df.columns else 0

feature_cols += ['is_q4', 'is_consolidated', 'is_preliminary', 'has_turnaround', 'rev_profit_div']

# Quarter dummies
for q in ['Q1', 'Q2', 'Q3']:
    df[f'is_{q.lower()}'] = (df['quarter_inferred'] == q).astype(int)
    feature_cols.append(f'is_{q.lower()}')

# Target
target = 'car_1d'

# Prepare data (drop NaN rows)
model_df = df[feature_cols + [target, 'company_name', 'stock_code']].dropna(subset=feature_cols + [target])
X = model_df[feature_cols].values
y = model_df[target].values
groups = model_df['stock_code'].values  # for GroupKFold

p(f"Model data: {len(model_df)} samples, {len(feature_cols)} features")
p(f"Features: {feature_cols}")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3a. Regression models (GroupKFold to prevent leakage)
gkf = GroupKFold(n_splits=5)

models_reg = {
    'OLS': LinearRegression(),
    'Lasso': Lasso(alpha=0.01),
    'RF': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'GBM': GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
}

p(f"\n--- Regression (predict CAR[0,1]) ---")
p(f"{'Model':<12} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
p("-" * 40)

best_r2 = -999
best_model_name = ''
for name, model in models_reg.items():
    use_X = X_scaled if name in ['OLS', 'Lasso'] else X
    r2_scores = cross_val_score(model, use_X, y, cv=gkf, groups=groups, scoring='r2')
    mae_scores = -cross_val_score(model, use_X, y, cv=gkf, groups=groups, scoring='neg_mean_absolute_error')
    rmse_scores = np.sqrt(-cross_val_score(model, use_X, y, cv=gkf, groups=groups, scoring='neg_mean_squared_error'))
    r2 = r2_scores.mean()
    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
    p(f"  {name:<12} {r2:>7.4f} {mae_scores.mean():>7.3f} {rmse_scores.mean():>7.3f}")

p(f"\nBest: {best_model_name} (R²={best_r2:.4f})")

# 3b. Feature importance (RF)
rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
p(f"\nRF Feature Importance (top 10):")
for feat, imp in importances.head(10).items():
    p(f"  {feat}: {imp:.4f}")

# =====================
# 4. Classification: Predict CAR direction
# =====================
p("\n=== 4. Classification: CAR Direction ===")

# Binary: positive CAR vs negative/zero
model_df['car_positive'] = (model_df[target] > 0).astype(int)
y_cls = model_df['car_positive'].values

p(f"Class balance: positive={y_cls.sum()} ({y_cls.mean()*100:.1f}%), negative={len(y_cls)-y_cls.sum()}")

models_cls = {
    'Logistic': LogisticRegression(max_iter=1000, random_state=42),
    'RF': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
}

p(f"\n{'Model':<12} {'Accuracy':>10} {'F1':>8} {'AUC':>8}")
p("-" * 42)
for name, model in models_cls.items():
    use_X = X_scaled if name == 'Logistic' else X
    acc = cross_val_score(model, use_X, y_cls, cv=gkf, groups=groups, scoring='accuracy').mean()
    f1 = cross_val_score(model, use_X, y_cls, cv=gkf, groups=groups, scoring='f1').mean()
    auc = cross_val_score(model, use_X, y_cls, cv=gkf, groups=groups, scoring='roc_auc').mean()
    p(f"  {name:<12} {acc:>9.4f} {f1:>7.4f} {auc:>7.4f}")

# =====================
# 5. Additional Analysis: Sentiment from text features
# =====================
p("\n=== 5. Text-based Features ===")

# Text length as info proxy
if 'text_length' in df.columns and 'car_1d' in df.columns:
    # Quartile analysis
    df['text_q'] = pd.qcut(df['text_length'].dropna(), 4, labels=['short','medium','long','very_long'])
    p("CAR by text length quartile:")
    for q in ['short','medium','long','very_long']:
        sub = df[df['text_q']==q]['car_1d'].dropna()
        if len(sub) > 0:
            p(f"  {q}: mean={sub.mean():.3f}%, n={len(sub)}")

# Preliminary vs final (잠정 vs 확정)
if 'is_preliminary' in df.columns:
    p(f"\nCAR by disclosure type:")
    for val, label in [(1, 'Preliminary'), (0, 'Final')]:
        sub = df[df['is_preliminary']==val]['car_1d'].dropna()
        if len(sub) > 0:
            p(f"  {label}: mean={sub.mean():.3f}%, n={len(sub)}")

# Consolidated vs standalone
if 'consolidated' in df.columns:
    p(f"\nCAR by statement type:")
    for val, label in [(True, 'Consolidated'), (False, 'Standalone')]:
        sub = df[df['consolidated']==val]['car_1d'].dropna()
        if len(sub) > 0:
            p(f"  {label}: mean={sub.mean():.3f}%, n={len(sub)}")

# =====================
# 6. Multi-window analysis
# =====================
p("\n=== 6. Multi-window CAR Analysis ===")
if 'op_profit_yoy' in df.columns:
    p(f"\nCAR by window and surprise direction:")
    p(f"{'Direction':<12} {'CAR[0,1]':>10} {'CAR[0,3]':>10} {'CAR[0,5]':>10}")
    p("-" * 46)
    for d in ['positive', 'negative']:
        sub = df[df['surprise_dir']==d]
        c1 = sub['car_1d'].dropna().mean()
        c3 = sub['car_3d'].dropna().mean()
        c5 = sub['car_5d'].dropna().mean()
        p(f"  {d:<12} {c1:>9.3f}% {c3:>9.3f}% {c5:>9.3f}%")

# =====================
# 7. Save results
# =====================
results = {
    'n_disclosures': len(df),
    'n_companies': int(df['company_name'].nunique()),
    'period': f"{df['rcept_dt'].min()} ~ {df['rcpt_dt'].max()}" if 'rcpt_dt' in df.columns else f"{df['rcpt_dt'].min() if 'rcpt_dt' in df.columns else df['rcept_dt'].min()} ~ {df['rcept_dt'].max()}",
    'parsing_rate': {
        'revenue': int(df['revenue'].notna().sum()),
        'op_profit': int(df['op_profit'].notna().sum()),
        'revenue_yoy': int(df['revenue_yoy'].notna().sum()),
        'op_profit_yoy': int(df['op_profit_yoy'].notna().sum()),
    },
    'car_overall': {
        'car_1d_mean': round(df['car_1d'].dropna().mean(), 3),
        'car_3d_mean': round(df['car_3d'].dropna().mean(), 3),
    },
    'best_regression_r2': round(best_r2, 4),
    'best_model': best_model_name,
}

with open(f'{DATA_DIR}/analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

df.to_pickle(f'{DATA_DIR}/dataset_final_v2.pkl')
p(f"\n✅ 분석 완료! 결과 저장: {DATA_DIR}/analysis_results.json")
