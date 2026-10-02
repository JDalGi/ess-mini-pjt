# ESS 배터리 수명 예측

초기 100사이클의 측정 신호로 배터리 총수명을 예측합니다. 배치별 EDA를 피처·모델 설계로 연결하고, 다른 배치에서의 예측 오차와 ESS 운영에 적용할 때의 한계를 평가합니다.

## 프로젝트 개요

- 데이터셋: MIT–Stanford Battery Dataset (Severson et al., Nature Energy 2019)
- 학습 데이터: Batch 1 (2017-05-12), 라벨 셀 46개
- 평가 데이터: Batch 2 (2018-02-20), 라벨 셀 39개
- EDA 대상: Batch 1·2·3. Batch 3 추가 모델 평가는 수행하지 않음
- 태스크: **Regression — Cycle Life 예측**
- 예측 시점: 100번째 사이클 종료 시점
- 지표: MAPE(%), 보조 지표 MAE·RMSE(사이클)
- 비교 목표: 과제에서 제시한 원논문 MAPE 9.1%

## 파일 구조

DAY 1·DAY 2 실행에 필요한 코드와 주요 생성 결과입니다. `results/`는 실행하면 생성됩니다.

```text
├── data/                             # 원본 MAT 데이터 보관
│   └── README.md                     # 데이터 파일명·배치 안내
├── notebooks/                        # 단계별 실행과 분석 해석
│   ├── 00_Scratch.ipynb              # Sample — 초기 데이터 탐색 예제
│   ├── 01_EDA.ipynb                  # DAY 1 배치별 EDA·모델 설계
│   ├── 02_feature_engineering.ipynb  # DAY 2 초기 피처 생성·품질 점검
│   └── 03_modeling.ipynb             # DAY 2 모델 선택·평가·오류 분석
├── src/                              # 노트북에서 사용하는 분석·학습 함수
│   ├── __init__.py                   # src 패키지 초기화
│   ├── preprocess.py                 # 원본 로딩·빈 기록 처리·전압축 정렬
│   ├── features.py                   # 초기 피처 추출·후보 묶음 정의
│   ├── eda.py                        # EDA 통계·그래프 생성
│   └── train.py                      # 그룹 분할·후보 탐색·모델 선택·평가
├── scripts/                          # 터미널 실행 도구
│   └── execute_notebooks.py          # 전체 또는 DAY별 노트북 순차 실행
├── results/                          # 실행으로 생성되는 분석·학습 결과
│   ├── eda/                          # DAY 1 통계·곡선·figures/
│   ├── auxiliary/                    # DAY 2 처리 가정·라벨 민감도 실험
│   ├── features.csv                  # 셀별 피처·라벨·점검 정보
│   ├── feature_manifest.csv          # 피처별 출처·관측 시점·설계 근거
│   ├── feature_quality.csv           # Batch 1 피처 결측·상수 여부
│   ├── batch1_split.csv              # CV·Hold-out 셀과 정책 그룹 배정
│   ├── cv_assignments.csv            # CV 폴드별 학습·검증 셀과 정책 그룹
│   ├── model_performance.csv         # MAPE·MAE·RMSE와 Gap 원값
│   ├── performance_report.csv        # 과제 지정 형식의 MAPE 성능표
│   ├── model_selection.csv           # 후보별 CV 점수·채택 여부·선택 근거
│   ├── selected_cv_folds.csv         # 선택 모델의 폴드별 MAPE·MAE·RMSE
│   ├── selected_model.csv            # 선택 피처·모델·타깃 변환·파라미터
│   ├── run_metadata.json             # 시드·표본 수·선택 기준·평가 한계
│   ├── predictions.csv               # Hold-out·Batch 2 셀별 예측
│   ├── error_analysis.csv            # Batch 2 셀별 오차·편향·피처 정보
│   ├── errors_by_life.csv            # 수명 구간별 오차와 평균 편향
│   ├── errors_by_feature_range.csv   # 선택 피처의 학습 범위 안·밖 오류
│   └── model.joblib                  # 최종 모델·입력 목록·예측 시점
├── pyproject.toml                    # Python 버전·의존성 정의
├── uv.lock                           # 의존성 버전 고정
└── README.md                         # 프로젝트 개요·실행 방법·결과 요약
```

`00_Scratch.ipynb`는 초기 탐색용 Sample이며 DAY 1·DAY 2 순차 실행 대상에서 제외됩니다. 최종 분석과 모델 평가 과정은 `01`~`03` 노트북에서 확인합니다.

결과 파일은 다음과 같이 연결해 확인할 수 있습니다.

| 확인할 내용        | 결과 파일                                                                                    | 설명                                                               |
| ------------------ | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 전략과 입력의 연결 | `feature_manifest.csv`, `feature_quality.csv`                                                | 피처의 원본 신호·관측 시점·설계 근거와 Batch 1 결측·상수 여부      |
| 데이터 분할        | `batch1_split.csv`, `cv_assignments.csv`                                                     | 셀별 CV·Hold-out 배정과 각 CV 폴드의 학습·검증 정책 그룹           |
| 모델 선택          | `model_selection.csv`, `selected_model.csv`, `selected_cv_folds.csv`                         | 후보 채택·탈락 근거, 최종 설정, 선택 모델의 폴드별 성능            |
| 실행 조건          | `run_metadata.json`                                                                          | 난수 시드, 표본 수, 폴드 수, 채택 기준, 기존 테스트 결과 확인 여부 |
| 예측과 오류 해석   | `predictions.csv`, `error_analysis.csv`, `errors_by_life.csv`, `errors_by_feature_range.csv` | 셀별 예측, 과대·과소 방향, 수명 구간과 입력 범위별 오류            |

## 환경 설정

검증 환경은 **Python 3.11.16, uv 0.12.19**입니다. uv가 설치된 환경에서 저장소를 내려받고 잠금 파일 기준으로 의존성을 설치합니다.

```bash
git clone https://github.com/JDalGi/ess-mini-pjt.git
cd ess-mini-pjt
uv sync --locked --python 3.11.16
```

원본 MAT 파일 세 개를 별도로 준비해 `data/`에 둡니다. 대용량 원본은 Git에 포함하지 않습니다. 별도 VarCharge 실험 파일은 사용하지 않습니다.

| 배치    | 필요한 파일                                            |
| ------- | ------------------------------------------------------ |
| Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` |
| Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` |
| Batch 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` |

### 실행 방법

프로젝트 루트에서 실행합니다. `uv run`이 프로젝트 `.venv`의 Python을 사용하므로 별도의 가상환경 활성화는 필요하지 않습니다. 실행 출력은 노트북에도 저장됩니다.

```bash
# 전체 실행: EDA → 피처 생성 → 모델링
uv run --locked python scripts/execute_notebooks.py

# DAY 1만 실행
uv run --locked python scripts/execute_notebooks.py --day 1

# DAY 2만 실행: 피처 생성 → 모델링
uv run --locked python scripts/execute_notebooks.py --day 2
```

DAY 2도 세 원본 파일을 읽으며 DAY 1을 먼저 실행할 필요는 없습니다. 노트북을 직접 실행할 경우 프로젝트의 `.venv/bin/python` 커널을 선택하고 `01_EDA → 02_feature_engineering → 03_modeling` 순서로 실행합니다. `03_modeling`은 `02_feature_engineering`이 생성한 `results/features.csv`를 읽습니다.

같은 코드·원본 데이터·환경에서 결과를 재현하려면 `uv.lock`을 유지하고 위의 `--locked` 명령으로 실행합니다. Hold-out 분할과 Random Forest·Gradient Boosting의 난수 시드는 **28**, GroupKFold는 셔플하지 않습니다. 최종 Ridge는 현재 설정에서 난수를 사용하는 학습 방식이 아닙니다. 재학습으로 선택 모델·분할·예측·성능의 재현을 확인했습니다.

## EDA

- **Cycle Life 분포:**
  - 중앙값은 Batch 1 858.5회, Batch 2 472회, Batch 3 1005.5회입니다. <500회 비율은 0.0%·71.8%·0.0%, >1000회 비율은 21.7%·7.7%·52.3%입니다.
  - **핵심 발견:** Batch 2에만 단수명 셀이 있어 배치 간 평가와 수명 구간별 오류 분석이 필요합니다.

- **열화 곡선 분석:**
  - 후기 감소가 중기보다 2배 넘게 빠른 셀은 배치별 84.8%·66.7%·77.3%입니다. Knee 후보 중앙값은 610·340·817.5회입니다.
  - **핵심 발견:** 열화는 후기에 가속되는 경우가 많으며, 전체 곡선에서 얻은 Knee는 초기 예측 입력에서 제외해야 합니다.

- **ΔQ(V) 곡선 분석:**
  - Q100(V)−Q10(V)를 계산했으며 상대적으로 수명이 짧은 집단에서 더 큰 음의 변화가 관찰됐습니다. 로그 분산–수명 Spearman 상관은 −0.871·−0.709·−0.797입니다. Batch 1·3은 <500회 셀이 없어 수명 사분위로 비교했습니다.
  - **핵심 발견:** 절대 초기 용량보다 전압별 변화 통계가 배치마다 일관된 수명 신호를 제공합니다.

- **충전 속도(C-rate)와 수명:**
  - Batch 1의 같은 8C→3.6C 정책에서도 전환 비율 15%·25%·35%에 따른 평균 수명은 1008.5·676.5·607.5회였습니다(각 2셀).
  - **핵심 발견:** 첫 C-rate만으로 수명을 설명하기 어려워 둘째 속도와 전환 조건을 함께 고려해야 합니다.

- **상관·다중공선성:**
  - 평균·최고온도 상관은 Batch 1·3에서 +0.951·+0.976이며, ΔQ 통계도 서로 중복됩니다.
  - **핵심 발견:** 대표 피처를 선택하고 규제 모델과 피처 묶음별 검증으로 중복을 줄여야 합니다. IR의 0값이 실제 측정값인지 미측정 표시인지, `newstructure`가 어떤 실험 조건의 차이를 나타내는지는 확인되지 않았습니다.

통계와 그래프는 `results/eda/`와 `results/eda/figures/`, 상세 해석은 `01_EDA.ipynb`에서 확인합니다.

## Modeling

### 피처 엔지니어링 전략

ΔQ 로그 분산을 출발점으로 다음 묶음을 비교합니다. 요약값은 2-100사이클, 용량 기울기는 10-100사이클의 Theil–Sen 추세, 실제 전류는 cycle 10의 시간 가중 통계입니다.

용량 급증값을 보존하면서 일부 관측값의 영향을 줄이기 위해 Theil–Sen 기울기를 사용하고, OLS와 보조 비교했습니다.

| 묶음     | 입력                                   | EDA 근거                              |
| -------- | -------------------------------------- | ------------------------------------- |
| A — 기본 | ΔQ 로그 분산                           | 세 배치에서 일관된 음의 수명 상관     |
| B — 용량 | A + 초기 QD 중앙값·강건 기울기         | 초기 수준과 변화 추세를 구분          |
| C — 온도 | A + 평균온도                           | 온도 변수 간 중복을 줄여 한 개씩 비교 |
| D — 정책 | A + 첫·둘째 C-rate·전환 비율·부가 표기 | 충전 단계 전체와 집단 차이를 고려     |
| E — 전류 | A + 실제 전류 평균·표준편차·p95        | 명목 정책 이외의 실측 전류 정보 검증  |

추가로 용량 변화·상대 변화·정책 결합항, 최고온도 교체, ΔQ 최솟값·평균의 대표 피처 교체를 비교합니다. IR·충전시간과 OLS 기울기, 높은 종료 용량 셀 제외는 보조 민감도 실험으로 확인합니다. `newstructure`는 Batch 1에서 모두 0이어서 표기 효과를 학습할 수 없습니다.

결측 대체·표준화는 각 학습 폴드의 Pipeline에서 적합합니다. 큰 양의 용량 급증은 보존하고 강건 추세를 사용합니다. 전체 기록 길이·종료 용량·Knee·후기 기울기는 입력에서 제외합니다.

### 모델 선택 및 근거

- **후보 모델:** 중앙값 기준선, 선형 회귀, Ridge·ElasticNet, 얕은 Random Forest·Gradient Boosting. 원래 수명과 로그 수명 타깃을 비교했습니다.
- **최종 모델:** ΔQ 최솟값 단일 입력의 **로그 수명 Ridge**(`alpha=0.1`). 로그 예측을 역변환해 원래 사이클 단위에서 평가합니다.
- **선택 이유:** 작은 학습 표본과 피처 중복을 고려해 단순한 입력·규제 모델을 비교했고, 채택 기준을 통과한 후보 중 CV MAPE 8.40%로 가장 낮았습니다. 평균 점수만 최저였던 용량 묶음 Gradient Boosting은 5폴드 중 3폴드만 개선돼 제외했습니다.

추가 피처는 같은 모델·타깃의 기준 묶음보다 **평균 CV MAPE가 1%p 이상 낮고 모든 폴드에서 개선될 때** 채택합니다. 이 기준은 DAY 1의 정성적 전략을 수치화한 운영 기준이며 통계적 유의성을 뜻하지 않습니다.

같은 수치 충전 정책의 셀은 같은 그룹에 묶습니다. Batch 1에서 CV용 36셀·Hold-out 10셀을 분리하고, 5폴드 그룹 CV로 모델을 선택합니다. Hold-out 평가 후 설정을 고정해 Batch 1 전체로 재학습합니다. Hold-out·Batch 2 점수는 재선택에 사용하지 않습니다.

## 성능 결과

| 구분                     | MAPE (%) | 비고                                     |
| ------------------------ | -------: | ---------------------------------------- |
| Train (Batch 1 CV)       |     8.40 | 정책 그룹 5폴드 평균; 모델 선택에도 사용 |
| Valid (Batch 1 Hold-out) |     6.52 | CV와 정책 그룹을 분리한 10셀             |
| Test (Batch 2)           |    26.09 | Batch 1 전체 학습 후 라벨 셀 39개 평가   |
| Gap (Train-Valid)        |    −1.88 | Valid−Train, %p                          |
| Gap (Valid-Test)         |   +19.57 | Test−Valid, %p; 배치 간 일반화 저하      |
| Gap (Target-Test)        |   +16.99 | Test−9.1, %p; 목표 미달                  |

Batch 2 MAE는 **132.26사이클**, RMSE는 **154.95사이클**입니다. 지정 형식의 표는 [results/performance_report.csv](results/performance_report.csv), 보조 지표 원값은 [results/model_performance.csv](results/model_performance.csv)에 저장합니다. 작은 Hold-out의 낮은 오차만으로 과적합이 없다고 판단할 수는 없습니다.

## 오류 분석

- **큰 오차 셀의 공통점:** Batch 2에서 상대 오차가 가장 큰 5셀은 모두 <500회 단수명이며 수명을 과대예측했습니다. 최대 오차 셀 `B2_018`은 실제 449회, 예측 779.95회로 APE 73.71%입니다. 단수명 28셀 전체의 MAPE는 28.99%, 평균 과대예측은 121.75회입니다.
- **원인 가설:** Batch 1 학습 수명 범위는 534~1227회로 단수명 구간이 없습니다. 배치별 초기 상태·측정 조건과 피처–수명 관계 차이도 오차에 기여했을 가능성이 있습니다. 범위 밖 여부만으로 원인을 확정하지는 않습니다.
- **개선 방향:** 실제 EOL 라벨·측정 조건을 확인하고 다양한 수명·운전 조건의 학습 셀을 확보합니다. 이후 새로운 독립 배치에서 검증합니다. 피처 범위·정책 집단별 오류와 처리 가정·라벨 민감도는 `03_modeling.ipynb`에 정리했습니다.

## ESS 도메인 해석

총수명 예측은 초기 셀 선별과 교체 계획의 참고 자료로 활용할 수 있습니다. 예측 총수명에서 100을 빼면 잔여 사이클 추정값을 얻지만 실제 교체 날짜를 바로 의미하지는 않습니다. 과대예측은 교체 지연으로 요구 용량·출력을 확보하지 못하게 할 수 있고, 과소예측은 조기 교체 비용을 늘릴 수 있습니다.

실제 BESS에 적용하려면 달력 열화·온도·SOC·운전 조건·셀 불균형과 팩 수준 차이를 반영한 현장 검증이 필요합니다. BMS 정보와 함께 판단하고 예측 불확실성과 과대예측 위험을 확인해야 합니다. 현재 모델은 자동 교체 결정이나 안전 이상·열폭주 예측을 지원할 수준으로 검증되지 않았습니다.

CV는 모델 선택에도 사용했고 Hold-out은 작은 단일 분할입니다. 세 배치 EDA와 기존 Batch 2 결과를 이미 확인한 수정 실험이며, 일부 라벨의 EOL 의미와 원논문 대비 평가 조건도 확인이 필요합니다.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. _Nature Energy_, 4, 383–391.

## 팀 구성

- 정문기 (울산캠퍼스 1반 U028): EDA, 피처 엔지니어링, 모델 개발, 성능 평가(Batch 2), 오류 분석 및 ESS 도메인 해석
