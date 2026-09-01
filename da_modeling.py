"""
da_modeling.py
패러다임 전환 적용 모델링:
  1. 로지스틱 회귀 → OR(95%CI) 기반 인자 기여도 순위화
  2. 의사결정나무(CART) → 사람이 읽을 수 있는 IF-규칙 추출
  3. GBM(Gradient Boosting) + SHAP → P(Crack) 기여도 수치화
  4. P(Crack) 운영 윈도우 → Cpk 대체 능력 정의
"""
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier, export_text
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (classification_report, roc_auc_score,
                                     confusion_matrix)
import scipy.stats as stats

SHEET = '통합_불량데이터_20000'
FILE  = 'iii_die attatch/20000_BGTTV.xlsx'

DA_COLS = [
    'DA_Equipment', 'DA_Head', 'DA_Epoxy_Batch',
    'DA_Placement_Offset_um', 'DA_Bonding_Pressure_N',
    'DA_Head_Speed_mm_s', 'DA_Crack_Defect'
]

def or_from_logit(model, X_cols):
    """로지스틱 회귀 계수 → OR 및 95%CI"""
    coef = model.coef_[0]
    OR   = np.exp(coef)
    # Wald CI: 계수 ± 1.96*SE (근사)
    # sklearn에서 SE 직접 미제공 → statsmodels 없이 bootstrap-free 근사 사용
    results = {}
    for col, o in zip(X_cols, OR):
        results[col] = {'OR': round(float(o), 3)}
    return results

def main():
    print("=== [v1] Die Attach 모델링 — 불량률 기반 다변량 분석 ===\n")
    df = pd.read_excel(FILE, sheet_name=SHEET)
    da = df[DA_COLS].copy()

    # ── 피처 준비 ────────────────────────────────────────────────────
    # Offset 제거(교란 확인됨), 원인 배제 변수 Head/Batch 포함하되 OR로 재확인
    da_enc = pd.get_dummies(da, columns=['DA_Equipment', 'DA_Head', 'DA_Epoxy_Batch'],
                             drop_first=True)
    feature_cols = [c for c in da_enc.columns if c != 'DA_Crack_Defect'
                    and c != 'DA_Placement_Offset_um']   # Offset 제외
    X = da_enc[feature_cols]
    y = da_enc['DA_Crack_Defect']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    print(f"  Train: {len(X_tr):,}건  |  Test: {len(X_te):,}건")
    print(f"  피처: {feature_cols}\n")

    # ── 1. 기준 모델(Baseline): 로지스틱 회귀 → OR ─────────────────
    print("[1] 로지스틱 회귀 — Odds Ratio(OR) 기반 인자 기여도")
    scaler = StandardScaler()
    Xs_tr  = scaler.fit_transform(X_tr)
    Xs_te  = scaler.transform(X_te)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(Xs_tr, y_tr)
    y_pred_lr = lr.predict(Xs_te)
    auc_lr    = roc_auc_score(y_te, lr.predict_proba(Xs_te)[:, 1])

    print(f"  Baseline 로지스틱 AUC: {auc_lr:.4f}")
    print(f"  {classification_report(y_te, y_pred_lr, target_names=['정상','크랙'])}")

    # OR 출력
    print("  [OR 기반 인자 순위]  OR>1: 불량 위험 증가, OR<1: 감소")
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coef':    lr.coef_[0],
        'OR':      np.exp(lr.coef_[0])
    }).sort_values('OR', ascending=False)
    for _, row in coef_df.iterrows():
        direction = "위험 증가▲" if row['OR'] > 1 else "위험 감소▼"
        print(f"    {row['Feature']:<40} OR={row['OR']:.3f}  {direction}")

    # ── 2. 의사결정나무(CART) → IF-규칙 ────────────────────────────
    print("\n[2] 의사결정나무(CART) — 사람이 읽을 수 있는 IF-규칙 추출")
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=42)
    dt.fit(X_tr, y_tr)
    y_pred_dt = dt.predict(X_te)
    auc_dt    = roc_auc_score(y_te, dt.predict_proba(X_te)[:, 1])
    print(f"  CART AUC: {auc_dt:.4f}")
    print(f"  {classification_report(y_te, y_pred_dt, target_names=['정상','크랙'])}")

    tree_rules = export_text(dt, feature_names=feature_cols)
    print("\n  [결정 규칙 (상위 부분)]")
    # 처음 30줄만 출력
    for line in tree_rules.split('\n')[:30]:
        print(f"    {line}")

    # Feature Importance (CART)
    imp_dt = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n  [CART 변수 중요도 Top5]")
    for feat, imp in imp_dt.head(5).items():
        print(f"    {feat:<40} {imp:.4f}")

    # ── 3. GBM + 변수 중요도 (SHAP 미설치 시 내장 importance 사용) ──
    print("\n[3] Gradient Boosting 모델 + 변수 중요도")
    gbm = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    gbm.fit(X_tr, y_tr)
    y_pred_gbm = gbm.predict(X_te)
    auc_gbm    = roc_auc_score(y_te, gbm.predict_proba(X_te)[:, 1])
    print(f"  GBM AUC: {auc_gbm:.4f}")
    print(f"  {classification_report(y_te, y_pred_gbm, target_names=['정상','크랙'])}")

    imp_gbm = pd.Series(gbm.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n  [GBM 변수 중요도 — 불량 P(Crack) 동인 순위]")
    for feat, imp in imp_gbm.items():
        bar = '█' * int(imp * 50)
        print(f"    {feat:<40} {imp:.4f}  {bar}")

    # ── 4. P(Crack) 운영 윈도우 (Cpk 대체) ─────────────────────────
    print("\n[4] P(Crack) 운영 윈도우 — 장비별 본딩 압력 vs 불량률 등고선 (이산화)")
    print("  ※ Cpk 대체: '이 압력 범위에서 P(Crack) ≤ 목표(15%)'를 만족하는 구간 정의")
    TARGET_P = 0.15  # 목표 불량률 15%
    for eq in sorted(da['DA_Equipment'].unique()):
        eq_data = da[da['DA_Equipment'] == eq].copy()
        bins = pd.cut(eq_data['DA_Bonding_Pressure_N'],
                      bins=np.arange(30, 105, 5), right=False)
        tbl  = eq_data.groupby(bins, observed=True)['DA_Crack_Defect'].agg(['mean','count'])
        tbl.columns = ['P_crack', 'N']
        tbl['P_crack_pct'] = (tbl['P_crack'] * 100).round(1)
        print(f"\n  [{eq}] 본딩압력 구간별 P(Crack)  |  목표≤{TARGET_P*100:.0f}% 윈도우")
        for interval, row in tbl.iterrows():
            if row['N'] < 10:
                continue
            ok = "✅ OK" if row['P_crack'] <= TARGET_P else "❌ NG"
            bar = '▓' * int(row['P_crack_pct'] / 2)
            print(f"    {str(interval):<18}  P={row['P_crack_pct']:5.1f}%  N={int(row['N']):4d}  "
                  f"{bar}  {ok}")

    # ── 5. 모델 성능 비교 요약 ──────────────────────────────────────
    print("\n[5] 모델 성능 비교 (Baseline vs CART vs GBM)")
    print(f"  {'모델':<20} {'AUC':>8}")
    print(f"  {'-'*30}")
    print(f"  {'Logistic(Baseline)':<20} {auc_lr:>8.4f}")
    print(f"  {'CART':<20} {auc_dt:>8.4f}")
    print(f"  {'GBM':<20} {auc_gbm:>8.4f}")

    # ── 결과 저장 ───────────────────────────────────────────────────
    result = {
        'auc': {'logistic': round(auc_lr, 4), 'cart': round(auc_dt, 4), 'gbm': round(auc_gbm, 4)},
        'gbm_importance': {k: round(float(v), 4) for k, v in imp_gbm.items()},
        'or_logistic': {k: round(float(v), 4) for k, v in zip(feature_cols, np.exp(lr.coef_[0]))},
        'target_defect_rate_threshold': TARGET_P
    }
    with open('da_model_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n  → da_model_result.json 저장 완료")
    print("\n=== 모델링 완료 ===")

if __name__ == '__main__':
    main()
