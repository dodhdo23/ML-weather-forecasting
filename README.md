# 🌡️ 기온 편차 예측 AI 프로젝트

## 📌 Overview
이 프로젝트는 **다음날 평균 기온이 해당 날짜의 기후 평균보다 얼마나 높거나 낮은지를 예측**하는 것을 목표로 합니다.  
기상 관측소의 시계열 데이터를 기반으로 머신러닝 모델(XGBoost, LightGBM, Ridge Stacking 등)을 활용하여 기온 편차를 예측합니다.

---

## 📂 Dataset
- **Train 데이터**: `train_dataset.csv`  
  - 13,132행 × 342열 (target 포함)  
  - target = `다음날 평균 기온 – 해당 날짜의 기후 평균 기온(climatology_temp)`  
- **Test 데이터**: `test_dataset.csv`  
  - 3,004행 × 341열 (target 제외)  

**관측소**  
- Train: 동두천, 서울, 강화, 인천, 이천, 양평 (2019~2024년 데이터)  
- Test: 파주, 수원  

**주요 변수 예시**  
- `surface_temp_0 ~ 23` : 지면 온도  
- `dew_point_0 ~ 23` : 이슬점 온도  
- `humidity_0 ~ 23` : 습도  
- `wind_speed_0 ~ 23` : 풍속  
- `visibility_0 ~ 23` : 시정  
- `precipitation_0 ~ 23` : 강수량  
- `sea_level_pressure_0 ~ 23` : 해면 기압  
- `station_name`, `date`, `climatology_temp`, `id` 등  

---

## 🛠 Data Preprocessing
1. **결측치 처리**
   - `-9999` → NaN 변환  
   - `sunshine_duration`, `snow_depth`: 의미 있는 결측치는 0으로 치환  
   - 나머지 결측: **관측소별 그룹화 후 선형 보간 + 평균 대체**  

2. **Feature Engineering**
   - `date`에서 월/요일/계절 등 파생  
   - 24시간 변수를 요약(평균·최대·최소·표준편차)  
   - `station_name` 원-핫 인코딩  
   - 풍향(`wind_direction`) → **sin/cos 변환**  

3. **Feature Selection**
   - 다중공선성(VIF), 상관계수, RMSE 비교를 통해 불필요한 변수 제거  
   - 예: `humidity_mean`, `surface_temp_max`, `snow_depth`, `min_cloud_height`, `visibility`, `dew_point_min` 등 제거  

---

## 🤖 Modeling
- **Base Models**
  - XGBoostRegressor (`n_estimators=1900, max_depth=5, learning_rate=0.04`, etc.)  
  - LightGBM (`n_estimators=1300, max_depth=6, num_leaves=48`, etc.)  

- **Meta Model**
  - Ridge Regression (스태킹, L2 규제)  

- **검증 전략**
  - `GroupKFold` (관측소 단위 분할) → 정보 누수 방지  
  - `KNNImputer` 활용한 결측치 보간  

---

## 📊 Performance
- **Baseline (XGBoost + Simple Imputer)** : RMSE **1.63**  
- **1차 스태킹 (XGB + LGB + RF + Ridge)** : RMSE **1.54**  
- **GroupKFold 적용 후** : RMSE **1.57** (현실적 검증 반영)  
- **Optuna + BayesianSearchCV 튜닝 (XGB 최적화)** : RMSE **1.46**  
- **XGB + LGB Voting 앙상블** : RMSE **1.41**  
- **최종 Ridge Stacking (XGB + LGB + Ridge, PCA 포함)** : RMSE **1.385**  

📈 **Kaggle Public Score: 0.792**

---

## 📌 Conclusion
- 관측소별 특성을 반영한 전처리와 파생 피처 생성이 모델 성능에 크게 기여  
- XGBoost와 LightGBM의 상호보완적 관계를 활용한 Ridge Stacking으로 최종 성능 극대화  
- RMSE 1.385, Kaggle Score 0.792 달성  
