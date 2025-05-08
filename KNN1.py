# import pandas as pd
# from sklearn.impute import KNNImputer
#
# # 1. 데이터 로딩
# df = pd.read_csv("motion_missing.csv")
#
# # 2. KNN Imputer
# knn_imputer = KNNImputer(n_neighbors=5)
# df_knn_imputed = knn_imputer.fit_transform(df)
# df_knn_imputed = pd.DataFrame(df_knn_imputed, columns=df.columns)
# df_knn_imputed.to_csv("motion_imputed_knn.csv", index=False)
#
# print("KNN Imputation 완료 -> motion_imputed_knn.csv 저장됨")


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

df_original = pd.read_csv("motion_original.csv")
df_imputed = pd.read_csv("motion_imputed_knn.csv")
df_missing = pd.read_csv("motion_missing.csv")

#결측이 있던 위치찾기
mask = df_missing.isnull()

# 3. 오차 비교: 결측 위치만 추출하여 RMSE 계산
original_vals = df_original[mask]
imputed_vals = df_imputed[mask]

# 모든 값을 1D로 펼쳐서 비교
rmse = np.sqrt(mean_squared_error(original_vals.stack(), imputed_vals.stack()))
print("RMSE (결측 위치만):", rmse)

print(f"RMSE (결측 위치 기준): {rmse:.4f}")