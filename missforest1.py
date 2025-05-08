import sys
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base']=sklearn.neighbors._base

import pandas as pd
from missingpy import MissForest

df_missing = pd.read_csv("motion_missing.csv")

#MissForest모델 초기화
imputer = MissForest()

#imputation 실행 -> 결과를 numpy배열로 전환
df_imputed = imputer.fit_transform(df_missing)

#numpy결과를 dataframe으로 다시 변환(df_missing의 열 이름 유지)
df_imputed = pd.DataFrame(df_imputed, columns=df_missing.columns)

#파일 생성
df_imputed.to_csv("motion__imputed.csv",index=False)
