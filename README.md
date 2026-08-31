# ShootingEvaluation (사격술 평가 모델)
> **[논문] 딥러닝 탐지 및 회귀 기반 사격술 평가 모델 설계**

[![Demo](https://img.shields.io/badge/Demo-Web%20Page-blue?logo=googlechrome&logoColor=white)](http://seojunha.com/ai)
[![Thesis](https://img.shields.io/badge/Thesis-dCollection%20Ajou-8A2BE2?logo=read-the-docs&logoColor=white)](https://dcoll.ajou.ac.kr/dcollection/common/orgView/000000036251)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/Detection-Ultralytics%20YOLO-00FFFF?logo=yolo&logoColor=black)
![XGBoost](https://img.shields.io/badge/Regression-XGBoost-11B5E4?logo=xgboost&logoColor=white)

---

## 1. 연구 배경 및 필요성

군 사격술 훈련에서 실거리 사격(50m, 100m, 200m 표적 총 20발)에 앞서 실시하는 영점사격은 총기의 가늠자 조정과 사격 자세 교정을 위한 필수 과정입니다. 그러나 기존 훈련 방식은 다음과 같은 명확한 한계를 지닙니다.

* **주관적·비정량적 피드백**: 사로 통제관(교관)의 육안 관찰 및 구두 지도에 의존하여 통제관별 피드백 편차가 발생하고, 사수 입장에서 실거리 사격 성과를 직관적으로 예측하기 어렵습니다.
* **기존 선행 연구의 한계**: 레이저 빔 장착이나 가상현실(VR) 기반 모의 사격 훈련은 고가의 장비 도입 비용이 발생하며, 실탄 격발 시의 반동 제어나 총기 기능 고장 처치 경험을 제한하여 실제 전장과의 괴리가 발생합니다.

**ShootingEvaluation**은 기존 사격장 인프라와 실탄 사격 환경을 100% 그대로 유지하면서, **스캔된 영점사격 표적지(JPEG)로부터 합성곱 신경망(YOLO)을 통해 탄흔 좌표를 검출하고 8차원 기하학적 특성 벡터를 추출한 뒤 머신러닝 회귀(XGBoost) 모델을 통해 실거리 사격 명중수를 정량적으로 예측**하는 2-Stage 인공지능 평가 시스템입니다.

---

## 2. 데모 & 논문 링크

* **데모 웹페이지**: [바로 가기](http://seojunha.com/ai), [샘플 이미지](https://raw.githubusercontent.com/ICT-SJH/ShootingEvaluation/refs/heads/main/demo.jpg)
  > *실제 영점 표적지 이미지를 업로드하여 모델의 실거리 사격 예측 점수(0~20점)를 직접 테스트해볼 수 있습니다.*
* **학위 논문 원본 (아주대학교 중앙도서관 dCollection)**: [바로 가기](https://dcoll.ajou.ac.kr/dcollection/common/orgView/000000036251)
  > *서준하, 「딥러닝 탐지 및 회귀 기반 사격술 평가 모델 설계」, 아주대학교 정보통신대학원 석사학위논문, 2026.*

---

## 3. 제안 모델 프레임워크 및 파이프라인

본 시스템은 컴퓨터 비전 기반의 객체 탐지와 공간 통계 기반의 피처 엔지니어링, 기계학습 회귀 모델이 유기적으로 결합된 **2-Stage End-to-End 구조**로 동작합니다.

```text
[ 영점 표적지 이미지 (JPEG) ]
               │
               ▼
[ Stage 1: 탄흔 객체 탐지 (Ultralytics YOLO) ]
  - 미세 객체 탐지 특화 (Small-Target-Aware Label Assignment)
  - 탄흔 바운딩 박스 검출 및 중심 좌표 (x, y) 도출
               │
               ▼
[ Feature Engineering: 8차원 기하학적 특성 벡터화 ]
  ├── 1-2. mean_x, mean_y : 탄착군 중심점 좌푯값
  ├── 3-4. std_x, std_y   : 좌우 / 상하 분산도 (조준선 안정성)
  ├── 5.   cov_xy         : 탄착군 분포 기울기 방향성
  ├── 6.   extreme_spread : 탄착군 내 최대 확산 거리 (Max Distance)
  ├── 7.   r50            : 중심점 기준 탄흔 거리 중앙값 (탄착군 대표 반경)
  └── 8.   ellipticity    : 공분산 행렬 고윳값 비율 기반 타원율 (호흡/견착 상태)
               │
               ▼
[ Stage 2: 실거리 사격 점수 예측 (XGBoost Regressor) ]
  - 8D Feature Vector 기반 20발 만점 기준 명중수 연속값 추론
               │
               ▼
[ 정량적 예측 점수 (Number of Hits) 출력 및 시각 피드백 ]

```

---

## 4. 모델 평가 및 실험 결과

총 34장의 고해상도 영점 표적지(총 304개 탄흔 인스턴스)를 기반으로 Train(26장/232개), Val(4장/36개), Test(4장/36개) 데이터셋을 구축하여 하이퍼파라미터 튜닝 및 비교 실험을 수행했습니다.

### 1) Stage 1: 탄흔 객체 탐지 모델 비교 (Validation)

| 모델 아키텍처 | 주요 하이퍼파라미터 | Precision | Recall | F1-Score | mAP50 |
| --- | --- | --- | --- | --- | --- |
| **YOLO (채택)**<br> | **epoch 200, imgsz 640, batch 16, conf 0.36**<br> | **1.000**<br> | **0.944**<br> | **0.972**<br> | **0.972**<br> |
| Faster R-CNN | epoch 100, imgsz 640, batch 8, conf 0.07 | 0.931 | 0.944 | 0.938 | 0.931 |

> **Test Set 최종 검증 결과:** YOLO 채택 모델 기준 **Precision 0.972, Recall 0.972, F1-Score 0.972, mAP50 0.984** 달성
> 
> 

### 2) Stage 2: 실거리 사격 점수 예측 모델 비교 (Validation)

| 회귀 알고리즘 | 주요 하이퍼파라미터 | MAE (평균 절대 오차) | RMSE (평균 제곱근 오차) |
| --- | --- | --- | --- |
| **XGBoost (채택)**<br> | **max_depth 3, eta 0.3, num_round 100**<br> | **1.3779**<br> | **1.4200**<br> |
| Random Forest | max_depth 6, min_samples_split 8, n_est 100 | 1.6990 | 1.7376 |

---

## 5. 사로 통제관(전문가 집단) 대비 성능 검증

군 복무 중 사로 통제관 임무 수행 경험이 있는 간부 13명을 대상으로 동일한 Test 표적지 4장에 대해 육안 관찰 후 점수를 예측하도록 설문을 진행했습니다. 불성실 응답(평균 10점 미만 2건)을 이상치로 필터링한 후 본 제안 AI 모델과의 성능을 비교 검증했습니다.

| 평가 집단 | MAE (평균 절대 오차) | RMSE (평균 제곱근 오차) |
| --- | --- | --- |
| 사로 통제관 전문가 집단 (Raw, N=13) | 3.4615 | 4.0961 |
| 사로 통제관 전문가 집단 (이상치 제거, N=11) | 2.3182 | 2.5945 |
| **본 제안 AI 모델 (YOLO + XGBoost)**<br> | **1.0014**<br> | **1.2873**<br> |

```text
Target 1 (실제 19.00점) ── AI 예측: 16.79점 │ 전문가(정제): 16.55점 │ 전문가(Raw): 14.31점[cite: 10, 14]
Target 2 (실제 20.00점) ── AI 예측: 18.74점 │ 전문가(정제): 16.00점 │ 전문가(Raw): 13.77점[cite: 10, 14]
Target 3 (실제 13.00점) ── AI 예측: 12.61점 │ 전문가(정제): 15.09점 │ 전문가(Raw): 13.46점[cite: 10, 14]
Target 4 (실제 17.00점) ── AI 예측: 17.15점 │ 전문가(정제): 16.27점 │ 전문가(Raw): 14.54점[cite: 10, 14]

```

> **검증 결론:** 제안 AI 모델은 정제된 전문가 집단의 육안 예측 대비 **MAE 기준 오차를 약 56.8% 감소(1.0014)**시키며, 주관적 구두 피드백을 대체할 수 있는 정량적이고 일관된 데이터 기반 평가 모델의 유효성을 입증했습니다.
> 
> 

---

## 6. 기술 스택 (Tech Stack)

* **Languages & Core**: Python 3.12+, C/C++


* **Object Detection & Deep Learning**: Ultralytics YOLO, Faster R-CNN, PyTorch, Torchvision


* **Machine Learning & Math**: XGBoost, Random Forest, Scikit-Learn, NumPy, SciPy (Spatial Distance, Covariance Matrix)


* **Visualization & Analysis**: Matplotlib, Seaborn, Pandas


* **Environment**: Google Colab (Tesla T4 GPU), Linux
