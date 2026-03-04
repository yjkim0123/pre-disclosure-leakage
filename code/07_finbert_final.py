#!/usr/bin/env python3
"""
Step 7: KR-FinBert-SC 감성분석 + 최종 분석
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re, json, warnings
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

warnings.filterwarnings('ignore')
DATA_DIR = '/Users/yongjun_kim/Documents/project_dart/data'
def p(msg): print(msg, flush=True)

# =====================
# 1. Load
# =====================
p("=== 1. 데이터 로드 ===")
df = pd.read_pickle(f'{DATA_DIR}/dataset_final_v3.pkl')
p(f"데이터: {len(df)}건")

# =====================
# 2. KR-FinBert-SC
# =====================
p("\n=== 2. KR-FinBert-SC 감성분석 ===")
tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-FinBert-SC')
model = AutoModelForSequenceClassification.from_pretrained('snunlp/KR-FinBert-SC')
model.eval()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)
p(f"모델 로드 완료, device={device}")
# Labels: 0=negative, 1=neutral, 2=positive

def finbert_sentiment(text, max_len=256):
    if not text or len(str(text)) < 50:
        return np.nan, np.nan, np.nan, np.nan
    text = str(text)
    # Keep financial summary (skip boilerplate header)
    if len(text) > 1500:
        text = text[:700] + ' ' + text[-700:]
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_len, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()[0]
        # Score: positive - negative
        score = float(probs[2] - probs[0])
        label = int(np.argmax(probs))
        return score, float(probs[0]), float(probs[1]), float(probs[2])
    except:
        return np.nan, np.nan, np.nan, np.nan

scores, p_neg, p_neu, p_pos = [], [], [], []
for i in range(len(df)):
    s, pn, pne, pp = finbert_sentiment(df.iloc[i]['text'])
    scores.append(s)
    p_neg.append(pn)
    p_neu.append(pne)
    p_pos.append(pp)
    if (i+1) % 200 == 0:
        p(f"  {i+1}/{len(df)}")

df['finbert_score'] = scores
df['finbert_neg'] = p_neg
df['finbert_neu'] = p_neu
df['finbert_pos'] = p_pos

valid = df['finbert_score'].notna().sum()
p(f"\nFinBert 결과: {valid}/{len(df)}")
p(f"Score: mean={df['finbert_score'].dropna().mean():.4f}, std={df['finbert_score'].dropna().std():.4f}")

# Label distribution
df['finbert_label'] = df['finbert_score'].apply(
    lambda x: 'positive' if x > 0.3 else ('negative' if x < -0.3 else 'neutral') if not np.isnan(x) else np.nan
)
p(f"분포: {df['finbert_label'].value_counts().to_dict()}")

# =====================
# 3. FinBert vs CAR
# =====================
p("\n=== 3. FinBert Sentiment vs CAR ===")
target = 'mcar_1d'

# By label
for label in ['negative', 'neutral', 'positive']:
    sub = df[df['finbert_label']==label][target].dropna()
    if len(sub) > 0:
        t_stat, pv = stats.ttest_1samp(sub, 0) if len(sub) > 2 else (0, 1)
        p(f"  {label}: CAR={sub.mean():.3f}%, t={t_stat:.2f}, p={pv:.3f}, n={len(sub)}")

# By tercile
df['finbert_tercile'] = pd.qcut(df['finbert_score'].dropna(), 3, labels=['bottom','middle','top'])
p(f"\nBy tercile:")
for t in ['bottom', 'middle', 'top']:
    sub = df[df['finbert_tercile']==t][target].dropna()
    if len(sub) > 0:
        p(f"  {t}: CAR={sub.mean():.3f}%, n={len(sub)}")

# Long-short spread
bottom = df[df['finbert_tercile']=='bottom'][target].dropna()
top = df[df['finbert_tercile']=='top'][target].dropna()
spread = top.mean() - bottom.mean()
t_ls, p_ls = stats.ttest_ind(top, bottom)
p(f"\nLong-Short spread: {spread:.3f}%p (t={t_ls:.2f}, p={p_ls:.3f})")

# Correlation
corr_fb = df[['finbert_score', target]].dropna().corr().iloc[0,1]
corr_lex = df[['sentiment', target]].dropna().corr().iloc[0,1]
p(f"\nCorrelation with {target}:")
p(f"  FinBert: {corr_fb:.4f}")
p(f"  Lexicon: {corr_lex:.4f}")

# Compare FinBert vs Lexicon
p(f"\nFinBert ↔ Lexicon corr: {df[['finbert_score','sentiment']].dropna().corr().iloc[0,1]:.4f}")

# =====================
# 4. Enhanced ML
# =====================
p("\n=== 4. ML with FinBert ===")

feature_cols = ['op_profit_yoy', 'revenue_yoy', 'net_income_yoy',
                'op_profit_qoq', 'revenue_qoq',
                'volume_ratio', 'pre_vol_20d', 'text_length',
                'sentiment', 'pos_words', 'neg_words',
                'finbert_score', 'finbert_neg', 'finbert_pos',
                'is_consolidated', 'is_preliminary', 'has_turnaround', 'rev_profit_div',
                'is_q1', 'is_q2', 'is_q3']
feature_cols = [c for c in feature_cols if c in df.columns]

model_df = df[feature_cols + [target, 'stock_code']].dropna(subset=feature_cols + [target])
X = model_df[feature_cols].values
y = model_df[target].values
groups = model_df['stock_code'].values

p(f"Data: {len(model_df)} samples, {len(feature_cols)} features")

gkf = GroupKFold(n_splits=5)

p(f"\n{'Model':<12} {'R²':>8} {'MAE':>8} {'RMSE':>8}")
p("-" * 40)

best_r2 = -999
for name, reg in [
    ('OLS', LinearRegression()),
    ('RF', RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42)),
    ('GBM', GradientBoostingRegressor(n_estimators=300, max_depth=4, min_samples_leaf=10, learning_rate=0.05, random_state=42)),
]:
    use_X = StandardScaler().fit_transform(X) if name == 'OLS' else X
    r2 = cross_val_score(reg, use_X, y, cv=gkf, groups=groups, scoring='r2').mean()
    mae = -cross_val_score(reg, use_X, y, cv=gkf, groups=groups, scoring='neg_mean_absolute_error').mean()
    rmse = np.sqrt(-cross_val_score(reg, use_X, y, cv=gkf, groups=groups, scoring='neg_mean_squared_error').mean())
    if r2 > best_r2: best_r2 = r2
    p(f"  {name:<12} {r2:>7.4f} {mae:>7.3f} {rmse:>7.3f}")

# Classification
p(f"\n--- Classification ---")
y_cls = (y > 0).astype(int)
for name, cls in [
    ('Logistic', LogisticRegression(max_iter=1000, random_state=42)),
    ('RF', RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42)),
]:
    use_X = StandardScaler().fit_transform(X) if name == 'Logistic' else X
    acc = cross_val_score(cls, use_X, y_cls, cv=gkf, groups=groups, scoring='accuracy').mean()
    auc = cross_val_score(cls, use_X, y_cls, cv=gkf, groups=groups, scoring='roc_auc').mean()
    p(f"  {name:<12} acc={acc:.4f}  auc={auc:.4f}")

# Feature importance
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
p(f"\nFeature Importance:")
for f, v in imp.items():
    p(f"  {f}: {v:.4f}")

# =====================
# 5. OLS with FinBert
# =====================
p("\n=== 5. OLS Regression ===")
reg_cols = ['op_profit_yoy', 'revenue_yoy', 'volume_ratio', 'pre_vol_20d',
            'sentiment', 'finbert_score']
reg_df = df[reg_cols + [target]].dropna()
if len(reg_df) > 50:
    Xr = sm.add_constant(reg_df[reg_cols])
    ols = sm.OLS(reg_df[target], Xr).fit(cov_type='HC1')
    p(ols.summary().as_text())

# =====================
# 6. Save
# =====================
df.to_pickle(f'{DATA_DIR}/dataset_final_v5.pkl')
df.to_csv(f'{DATA_DIR}/dataset_final_v5.csv', index=False, encoding='utf-8-sig')

summary = {
    'n': len(df),
    'finbert_model': 'snunlp/KR-FinBert-SC',
    'finbert_car_corr': round(corr_fb, 4),
    'lexicon_car_corr': round(corr_lex, 4),
    'long_short_spread': round(spread, 3),
    'long_short_pval': round(p_ls, 4),
    'best_ml_r2': round(best_r2, 4),
    'finbert_dist': df['finbert_label'].value_counts().to_dict(),
}
with open(f'{DATA_DIR}/analysis_final.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

p(f"\n✅ 완료!")
p(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
