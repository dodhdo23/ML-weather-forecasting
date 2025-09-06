from sklearn.impute import KNNImputer
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# =========================
# 1. 학습·평가용 데이터 준비
# =========================
target = "target"
drop_cols = ["id", "station", target]

X      = train_df.drop(columns=drop_cols)
y      = train_df[target]
X_test = test_df.drop(columns=["id", "station"])
groups = train_df["station"]
# 결측 보간: KNNImputer
imputer = KNNImputer(n_neighbors=3, weights="distance")
X      = pd.DataFrame(imputer.fit_transform(X),      columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# =========================
# 2. 1단계(base) 모델 정의
# =========================
models = {
    "xgb": xgb.XGBRegressor(
        n_estimators=1900,
        learning_rate=0.04,
        max_depth=5,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.6,
        reg_lambda=4.4,
        tree_method="hist",
        early_stopping_rounds=50,          # 그대로 사용 가능
        random_state=42,
        n_jobs=-1
    ),

    "lgb": lgb.LGBMRegressor(
        n_estimators=1300,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=48,                     # 2^6 ~=64
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        reg_alpha=0.3,
        random_state=42
    )
}

# =========================
# 3. KFold 학습
# =========================
n_splits = 3
kf = GroupKFold(n_splits=n_splits)


oof_preds  = {name: np.zeros(len(X))      for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y, groups=groups)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    for name, model in models.items():
        print(f"{name}: fold {fold} ⮕ {type(model)}")

        if name == "lgb":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                eval_metric='rmse',
                verbose=100
            )
        else:  # XGB
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=100
            )

        oof_preds[name][val_idx] += model.predict(X_val)
        test_preds[name]         += model.predict(X_test) / n_splits

# =========================
# 4. 2단계(meta) 모델 – Ridge
# =========================
oof_stack  = np.column_stack([oof_preds[name]  for name in models])
test_stack = np.column_stack([test_preds[name] for name in models])

meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_stack, y)
final_preds = meta_model.predict(test_stack)

print("✅ Stacking RMSE:",
      np.sqrt(mean_squared_error(y, meta_model.predict(oof_stack))))

submission = pd.DataFrame({
    "id": test_df["id"].values,        # test셋 고유 ID
    "target": final_preds              # Ridge 메타모델 최종 예측 결과
})

# 저장
save_path = "/content/drive/MyDrive/stacking_submission.csv"
submission.to_csv(save_path, index=False)

print("✅ 제출 파일이 저장되었습니다 →", save_path)
print(submission.head())