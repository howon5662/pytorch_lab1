# import pandas as pd
# from sklearn.experimental import enable_iterative_imputer  # MICE 사용을 위한 import
# from sklearn.impute import IterativeImputer
#
# # 데이터 로딩
# df = pd.read_csv("motion_missing.csv")
#
# # MICE 모델 정의
# imputer = IterativeImputer(random_state=0)
#
# # 결측치 채우기
# df_imputed = imputer.fit_transform(df)
#
# # DataFrame으로 변환
# df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
#
# # 결과 저장
# df_imputed.to_csv("motion_imputed_mice.csv", index=False)


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

df_original = pd.read_csv("motion_original.csv")
df_imputed = pd.read_csv("motion_imputed_mice.csv")
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
