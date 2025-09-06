# =========================
# 1. 라이브러리 및 전처리 함수
# =========================
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# =========================
# 날짜 파생 피처 생성 (윤년 대응)
# =========================
def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    'date' 컬럼(형식: 'MM-DD') → month·day·weekday·is_weekend·season_* 추가
    윤년(2000)을 붙여 2월 29일도 파싱 가능
    """
    # ① 2000년(윤년) 붙여서 datetime 변환
    dt = pd.to_datetime('2000-' + df['date'], format='%Y-%m-%d')

    # ② 기본 파생
    df['month']      = dt.dt.month
    df['day']        = dt.dt.day
    df['weekday']    = dt.dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    # ③ 시즌 One-Hot
    season = (dt.dt.month % 12) // 3          # 0:봄 1:여름 2:가을 3:겨울
    df[['season_0','season_1','season_2','season_3']] = (
        pd.get_dummies(season)
          .reindex(columns=[0,1,2,3], fill_value=0)
          .values
    )
    return df
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def apply_pca(train_df, test_df, pca_cols, n_components=2):
    """
    다중공선성 높은 변수들을 PCA로 축소하여 새로운 성분을 생성합니다.
    생성된 성분은 'pca_1', 'pca_2', ... 형태로 train/test에 추가됩니다.

    Args:
        train_df (DataFrame): 훈련 데이터
        test_df (DataFrame): 테스트 데이터
        pca_cols (list of str): PCA에 사용할 컬럼 목록
        n_components (int): 주성분 개수 (기본값=2)

    Returns:
        train_df, test_df (DataFrame): PCA 성분이 추가된 결과 반환
    """
    # 1. 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df[pca_cols])
    X_test_scaled  = scaler.transform(test_df[pca_cols])

    # 2. PCA 적용
    pca = PCA(n_components=n_components, random_state=42)
    train_pca = pca.fit_transform(X_train_scaled)
    test_pca  = pca.transform(X_test_scaled)

    # 3. 컬럼명 지정
    pca_names = [f'pca_{i+1}' for i in range(n_components)]

    # 4. 결과 붙이기
    train_df[pca_names] = pd.DataFrame(train_pca, index=train_df.index)
    test_df[pca_names]  = pd.DataFrame(test_pca,  index=test_df.index)

    return train_df, test_df


def drop_redundant_features(df):
    """불필요하거나 성능 저해 피처 제거"""
    drop_cols = []

    # 1) surface_temp: 일교차(range)만 남기고 나머지 제거
    drop_cols += [
        "surface_temp_mean",
        "surface_temp_max",
        "surface_temp_min",
        "surface_temp_std"
    ]


    # 2) pressure: sea_level_pressure_mean만 남기고 모두 제거
    drop_cols += ["sea_level_pressure_max", "sea_level_pressure_min", "sea_level_pressure_std"]


    drop_cols += [f"local_pressure_{s}" for s in ["mean", "max", "min", "std"]]


    # 3) dew_point: min만 제거
    drop_cols += ["dew_point_min"]


    # 4) humidity: std만 남기고 mean/max/min 제거
    drop_cols += [
        "humidity_mean",
        "humidity_max",
        "humidity_min"
    ]



    # 5) snow_depth: 전체 요약치 제거
    drop_cols += [
        "snow_depth_mean",
        "snow_depth_max",
        "snow_depth_min",
        "snow_depth_std"
    ]


    # 6) min_cloud_height: 모든 시간대 컬럼 제거
    drop_cols += [f"min_cloud_height_{i}" for i in range(24)]

    # 7) visibility: 전체 요약치 제거
    drop_cols += [
        "visibility_mean",
        "visibility_max",
        "visibility_min",
        "visibility_std"
    ]


    # 8) sunshine_duration: std 제거 (mean/max/min만 유지)
    drop_cols += ["sunshine_duration_std"]


    # 9) vapor_pressure: 원본 24시간 컬럼 모두 제거
    drop_cols += [f"vapor_pressure_{i}" for i in range(24)]

    # 10) cloud_cover: std 제거 (mean/max/min만 유지)
    drop_cols += ["cloud_cover_std"]


    # 실제로 존재하는 컬럼만 선택해 제거
    existing = [c for c in drop_cols if c in df.columns]
    df.drop(columns=existing, inplace=True)
    return df


# =========================
# 2-1. 결측치 처리 및 보간 (train/test 공통)
# =========================


# 전처리 적용
train_df = preprocess(train)
test_df = preprocess(test)

pca_cols = ['dew_point_mean', 'humidity_mean', 'surface_temp_mean', 'vapor_pressure_mean']
train_df, test_df = apply_pca(train_df, test_df, pca_cols=pca_cols, n_components=2)


# 피처 제거
train_df = drop_redundant_features(train_df)
test_df = drop_redundant_features(test_df)

# ⭐️ 날짜 파생 피처 추가
train_df = add_date_features(train_df)
test_df  = add_date_features(test_df)

# date 문자열은 이제 불필요
train_df.drop(columns=['date'], inplace=True)
test_df.drop(columns=['date'],  inplace=True)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
train_ohe = ohe.fit_transform(train[['station_name']])
test_ohe = ohe.transform(test[['station_name']])
ohe_cols = ohe.get_feature_names_out(['station_name'])

# 인코딩 결과를 train_df와 test_df에 붙이기
train_df[ohe_cols] = pd.DataFrame(train_ohe, index=train_df.index)
test_df[ohe_cols] = pd.DataFrame(test_ohe, index=test_df.index)

# 문자열 컬럼 제거
train_df.drop(columns=['station_name'], inplace=True)
test_df.drop(columns=['station_name'], inplace=True)
