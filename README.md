# Die Attach Crack 불량 분석 (패러다임 전환)

본 저장소는 반도체 후공정(Packaging) **Die Attach** 단계에서 발생하는 크랙(Crack) 불량의 원인을 분석하고, 머신러닝 기반의 최적 공정 윈도우(Operating Window)를 제시하는 분석 결과를 담고 있습니다.

## 🚀 프로젝트 개요
공정 초기 셋업 데이터를 분석할 때 전통적인 양산 SPC 지표(Cpk, 단일 변수 T-test)를 적용할 경우, 혼합 모집단의 특성으로 인해 **교란 변수(Confounding Variable)** 를 진짜 원인으로 착각하는 치명적인 오류가 발생할 수 있습니다. 

본 분석에서는 데이터의 생애주기(초기 셋업 vs 양산)에 맞춰 다음과 같이 분석 패러다임을 전환하여 진짜 인과 인자를 찾아냈습니다.

1. **능력지수(Cpk) → 불량률(DPMO/Z-bench) 기반 지표**로 전환
2. **단변량 통계(p-value) → 다변량 효과 크기(Odds Ratio, Cohen's d)** 기반 평가
3. **규격 한계 → P(Crack) 불량 확률 모델링(Logistic Regression / GBM)** 으로 임계값 설정

## 📊 주요 분석 결과
* **장비 B(EQ_B)의 지배적 결함**: 다른 장비 대비 크랙 불량 위험도가 약 **3.69배(OR)** 높습니다.
* **교란 변수 통제**: `Placement Offset`은 EQ_B 장비의 대리지표(Proxy)였으며, 장비별 층화 분석을 통해 불량의 독립적 원인이 아님을 입증했습니다.
* **진짜 인과 후보 발굴**: 3개 장비 모두에서 일관되게 불량을 유발하는 **본딩 압력(Bonding Pressure)** 의 기여도를 입증했습니다.
* **운영 윈도우 설정**: 각 장비별로 P(Crack) <= 15%를 달성하기 위한 안전 본딩 압력 구간을 도출했습니다.

## 📁 주요 파일 구성
* `index.html`: 분석 결과와 **인터랙티브 JS 시뮬레이터**가 포함된 종합 리포트 대시보드
* `da_preprocessing_v2.py`: 장비별 층화를 통한 이상치 탐지 및 DPMO 산출 스크립트
* `da_eda_v2.py`: 효과 크기(Cohen's d, OR, ARD) 기반의 데이터 탐색 스크립트
* `da_modeling.py`: GBM, CART, Logistic Regression 예측 모델링 스크립트
* `da_report_generator.py`: Base64 내장 차트, MathJax, Mermaid가 적용된 고급 HTML 생성 로직

## 🛠 실행 방법
본 저장소의 `index.html` 파일을 브라우저로 열거나, GitHub Pages를 활성화하면 리포트와 시뮬레이터를 즉시 확인할 수 있습니다.

```bash
# 로컬에서 리포트 생성 스크립트 재실행 시
python -X utf8 da_report_generator.py
```