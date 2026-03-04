#!/usr/bin/env python3
"""
Step 6: KoBERT/KLUE-BERT 감성분석 + 최종 ML 재실행
"""
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
# 2. KLUE-BERT Sentiment
# =====================
p("\n=== 2. KLUE-BERT 감성분석 ===")

# Use KLUE-BERT fine-tuned for sentiment (Korean)
# Try multiple models
model_name = None
tokenizer = None
model = None

for mname in [
    'snunlp/KR-FinBert-SC',        # Korean financial sentiment
    'beomi/KcELECTRA-base-v2022',   # Korean general
    'klue/bert-base',                # KLUE base
]:
    try:
        p(f"  시도: {mname}")
        tokenizer = AutoTokenizer.from_pretrained(mname)
        model = AutoModelForSequenceClassification.from_pretrained(mname)
        model_name = mname
        model.eval()
        p(f"  ✅ 로드 성공: {mname}")
        p(f"  Labels: {model.config.id2label if hasattr(model.config, 'id2label') else 'unknown'}")
        break
    except Exception as e:
        p(f"  ❌ {mname}: {e}")
        tokenizer = None
        model = None

if model is None:
    # Fallback: use multilingual sentiment model
    try:
        mname = 'nlptown/bert-base-multilingual-uncased-sentiment'
        tokenizer = AutoTokenizer.from_pretrained(mname)
        model = AutoModelForSequenceClassification.from_pretrained(mname)
        model_name = mname
        model.eval()
        p(f"  ✅ Fallback 로드: {mname} (1-5 star rating)")
    except Exception as e:
        p(f"  ❌ 모든 모델 실패: {e}")

if model is not None and tokenizer is not None:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    p(f"  Device: {device}")
    model = model.to(device)
    
    def get_bert_sentiment(text, max_len=256):
        """Get BERT sentiment score from disclosure text"""
        if not text or len(str(text)) < 50:
            return np.nan, np.nan
        
        # Extract key financial section (not full text)
        text = str(text)
        # Focus on the numbers section and any commentary
        # Truncate intelligently
        if len(text) > 1000:
            # Keep first 500 + last 500 chars (header + commentary)
            text = text[:500] + ' ' + text[-500:]
        
        try:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, 
                             max_length=max_len, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            if 'sentiment' in model_name.lower() or 'finbert' in model_name.lower():
                # Typically: 0=negative, 1=neutral, 2=positive
                if len(probs) == 3:
                    score = probs[2] - probs[0]  # positive - negative
                    label_idx = np.argmax(probs)
                    return score, label_idx
                elif len(probs) == 2:
                    return probs[1] - probs[0], np.argmax(probs)
            
            if 'nlptown' in model_name:
                # 5-class: 1star to 5star
                score = (probs * np.array([1,2,3,4,5])).sum()
                normalized = (score - 3) / 2  # normalize to [-1, 1]
                return normalized, int(np.argmax(probs))
            
            # Generic: assume last class is positive
            score = probs[-1] - probs[0]
            return score, int(np.argmax(probs))
            
        except Exception as e:
            return np.nan, np.nan
    
    # Process all texts
    p(f"\n  Processing {len(df)} texts...")
    bert_scores = []
    bert_labels = []
    
    batch_size = 1  # Process one at a time for stability
    for i in range(len(df)):
        score, label = get_bert_sentiment(df.iloc[i]['text'])
        bert_scores.append(score)
        bert_labels.append(label)
        
        if (i+1) % 100 == 0:
            valid = sum(1 for s in bert_scores if not np.isnan(s))
            p(f"  {i+1}/{len(df)} (valid: {valid})")
    
    df['bert_sentiment'] = bert_scores
    df['bert_label'] = bert_labels
    
    valid_count = df['bert_sentiment'].notna().sum()
    p(f"\n  BERT 감성 추출: {valid_count}/{len(df)}")
    p(f"  Mean: {df['bert_sentiment'].dropna().mean():.4f}")
    p(f"  Std: {df['bert_sentiment'].dropna().std():.4f}")
    
    # BERT sentiment vs CAR
    p(f"\n--- BERT Sentiment vs CAR ---")
    df['bert_sent_q'] = pd.qcut(df['bert_sentiment'].dropna(), 3, labels=['neg','neutral','pos'])
    for q in ['neg', 'neutral', 'pos']:
        sub = df[df['bert_sent_q']==q]['mcar_1d'].dropna()
        if len(sub) > 0:
            p(f"  {q}: mean CAR={sub.mean():.3f}%, n={len(sub)}")
    
    # Correlation
    corr = df[['bert_sentiment', 'mcar_1d']].dropna().corr().iloc[0,1]
    p(f"\n  BERT sentiment ↔ CAR correlation: {corr:.4f}")
    
    # Compare lexicon vs BERT
    both = df[['sentiment', 'bert_sentiment', 'mcar_1d']].dropna()
    p(f"\n--- Lexicon vs BERT ---")
    p(f"  Lexicon ↔ CAR: {both[['sentiment','mcar_1d']].corr().iloc[0,1]:.4f}")
    p(f"  BERT ↔ CAR: {both[['bert_sentiment','mcar_1d']].corr().iloc[0,1]:.4f}")
    p(f"  Lexicon ↔ BERT: {both[['sentiment','bert_sentiment']].corr().iloc[0,1]:.4f}")

else:
    p("⚠️ BERT 모델 없음 — lexicon 감성만 사용")
    df['bert_sentiment'] = np.nan

# =====================
# 3. Enhanced ML with BERT
# =====================
p("\n=== 3. Enhanced ML (with BERT sentiment) ===")

feature_cols = ['op_profit_yoy', 'revenue_yoy', 'net_income_yoy',
                'op_profit_qoq', 'revenue_qoq',
                'volume_ratio', 'pre_vol_20d', 'text_length',
                'sentiment', 'pos_words', 'neg_words',
                'avg_sentence_len', 'num_count',
                'is_consolidated', 'is_preliminary', 'has_turnaround', 'rev_profit_div',
                'is_q1', 'is_q2', 'is_q3']

# Add BERT if available
if df['bert_sentiment'].notna().sum() > 100:
    feature_cols.append('bert_sentiment')

# Ensure columns exist
feature_cols = [c for c in feature_cols if c in df.columns]

target = 'mcar_1d'
model_df = df[feature_cols + [target, 'stock_code']].dropna(subset=feature_cols + [target])
X = model_df[feature_cols].values
y = model_df[target].values
groups = model_df['stock_code'].values

p(f"Model: {len(model_df)} samples, {len(feature_cols)} features")
p(f"Features: {feature_cols}")

gkf = GroupKFold(n_splits=5)

p(f"\n{'Model':<12} {'R²':>8} {'MAE':>8} {'AUC(cls)':>10}")
p("-" * 42)
for name, reg_model, cls_model in [
    ('RF', RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42),
           RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42)),
    ('GBM', GradientBoostingRegressor(n_estimators=300, max_depth=4, min_samples_leaf=10, learning_rate=0.05, random_state=42),
            None),
]:
    r2 = cross_val_score(reg_model, X, y, cv=gkf, groups=groups, scoring='r2').mean()
    mae = -cross_val_score(reg_model, X, y, cv=gkf, groups=groups, scoring='neg_mean_absolute_error').mean()
    
    y_cls = (y > 0).astype(int)
    if cls_model:
        auc = cross_val_score(cls_model, X, y_cls, cv=gkf, groups=groups, scoring='roc_auc').mean()
    else:
        auc = 0
    p(f"  {name:<12} {r2:>7.4f} {mae:>7.3f} {auc:>9.4f}")

# Feature importance
rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
p(f"\nFeature Importance:")
for f, v in imp.items():
    p(f"  {f}: {v:.4f}")

# =====================
# 4. Final OLS with BERT
# =====================
p("\n=== 4. OLS with BERT ===")
import statsmodels.api as sm

reg_cols = ['op_profit_yoy', 'revenue_yoy', 'volume_ratio', 'pre_vol_20d', 'sentiment']
if df['bert_sentiment'].notna().sum() > 100:
    reg_cols.append('bert_sentiment')

reg_df = df[reg_cols + [target]].dropna()
if len(reg_df) > 50:
    Xr = sm.add_constant(reg_df[reg_cols])
    ols = sm.OLS(reg_df[target], Xr).fit(cov_type='HC1')
    p(ols.summary().as_text())

# =====================
# 5. Save
# =====================
df.to_pickle(f'{DATA_DIR}/dataset_final_v4.pkl')

summary = {
    'model_used': model_name or 'lexicon_only',
    'bert_sentiment_mean': round(df['bert_sentiment'].dropna().mean(), 4) if df['bert_sentiment'].notna().any() else None,
    'bert_car_corr': round(df[['bert_sentiment','mcar_1d']].dropna().corr().iloc[0,1], 4) if df['bert_sentiment'].notna().sum() > 10 else None,
    'best_ml_r2': round(max(
        cross_val_score(RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=42),
                       X, y, cv=gkf, groups=groups, scoring='r2').mean(),
        0
    ), 4),
}
with open(f'{DATA_DIR}/analysis_v3.json', 'w') as f:
    json.dump(summary, f, indent=2)

p(f"\n✅ 완료!")
p(json.dumps(summary, indent=2, ensure_ascii=False))
