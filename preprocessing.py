# -9999는 기기 오류 → NaN으로 처리
import numpy as np
for df in (train, test):
    df.replace(-9999, np.nan, inplace=True)

#데이터 특성 : 일조량은 밤이면 관측값이 없지만 실제론 ‘0 시간’이 의미.
night_hours = [0,1,2,3,4,5,22,23]
for h in night_hours:
    col = f'sunshine_duration_{h}'
    for df in (train, test):
        if col in df.columns:
            df[col] = df[col].fillna(0)

#특성 : 서울·경기권 관측소는 눈이 오지 않은 날이 다수 → 결측 = 눈 0 cm 로 해석
snow_cols = [c for c in train.columns if 'snow_depth_' in c]
for df in (train, test):
    df[snow_cols] = df[snow_cols].fillna(0)

# 4-4) 그룹별 보간 (station 기준)
for df in (train, test):
    df.sort_values(['station','date'], inplace=True)
    numeric_cols = df.select_dtypes('number').columns

    # (1) station 그룹별로 short-gap 선형 보간
    df[numeric_cols] = (
        df.groupby('station')[numeric_cols]
          .transform(lambda grp: grp.interpolate(limit=3))
    )
    # (2) 선형 보간 후에도 남은 NaN을 station별 평균으로 채움
    df[numeric_cols] = (
        df.groupby('station')[numeric_cols]
          .transform(lambda grp: grp.fillna(grp.mean()))
    )

for df in (train, test):
    df['month'] = df['date'].str.split('-').str[0].astype(int)
#특성 : 기온 편차는 계절성이 강함. “일교차 요약” 전에 월만 추출해도 효과.

print(train.columns[:20])          # 앞 20개 컬럼명만 미리 보기
print('station_name' in train.columns)
