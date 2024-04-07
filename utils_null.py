import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings 
warnings.filterwarnings('ignore')

# 결측치 검증 및 대체 : 전제조건 연속형 데이터 프레임
class NullCheckSubstitution:
    def __init__(self, dataframe):
        self.nul_cols = dataframe.columns[dataframe.isnull().sum() != 0] # 결측치가 있는 컬럼
        self.df = dataframe
        
    def null_check(self):
        print('본 결측치 체크는 연속형 변수에 대하여 진행합니다.')
        nul_cols = self.df.columns[self.df.isnull().sum() != 0] # 결측치가 있는 컬럼
        nonul_cols = self.df.columns[self.df.isnull().sum() == 0] # 결측치가 없는 컬럼
        if len(nul_cols) == 0 :
            print('결측치 데이터 컬럼 없음!')
            result = self.df
        else:
            #method = int(input('method(1.mean, 2.median, 3.KNN) : '))
            method = 3
            result = self.null_substitution(method, nul_cols, nonul_cols)
        return result

    def null_check_again(self, df):
        nul_cols = df.columns[df.isnull().sum() != 0] # 결측치가 있는 컬럼
        print(f'결측치 대체 후 검정 결측치 컬럼갯수 : {len(nul_cols)} ')
        if len(nul_cols) == 0:
            flag_null = 0
        else:
            flag_null = 1
        return flag_null
    
    def null_substitution_spt_mean(self):
        df_median = self.df
        nul_cols = df_median.columns[df_median.isnull().sum() != 0] # 결측치가 있는 컬럼
        for col in nul_cols:
            nul_index = df_median[col][df_median[col].isnull()].index
            df_median.loc[nul_index, col] = df_median[col].mean()
            result = df_median.copy()
        return result

    def null_substitution_spt_median(self):
        df_median = self.df
        nul_cols = df_median.columns[df_median.isnull().sum() != 0] # 결측치가 있는 컬럼
        for col in nul_cols:
            nul_index = df_median[col][df_median[col].isnull()].index
            df_median.loc[nul_index, col] = df_median[col].median()
            result = df_median.copy()

        return result

    def null_substitution_spt_knn(self):
        df_knn = self.df
        from sklearn.impute import KNNImputer
        KNN_data = df_knn[self.nul_cols]
        # - 모델링
        imputer = KNNImputer()
        df_filled = imputer.fit_transform(KNN_data)
        df_filled = pd.DataFrame(df_filled, columns=KNN_data.columns)
        df_knn[KNN_data.columns] = df_filled
        result = df_knn.copy()

        return result
    
    def null_substitution(self, method, nul_cols, nonul_cols):
        print(f'결측치 데이터 컬럼 : {len(nul_cols)}개, {list(nul_cols)} ')
        print(f'{self.df.isnull().sum()}')
        
        flag = 1
        while flag:
            if method == 1:
                print('평균값 대체')
                result = self.null_substitution_spt_mean()
                flag = self.null_check_again(result)
                
            elif method == 2:
                print('중앙값 대체')
                result = self.null_substitution_spt_median()
                flag = self.null_check_again(result)
                
            elif method == 3:
                print('KNN 알고리즘 대체') 
                result = self.null_substitution_spt_knn()
                flag = self.null_check_again(result)
            else:
                print('대체 방법을 다시 선택해 주세요')
                result = pd.DataFrame()
                flag = 0
        return result
'''(사용법)
numerics = Numeric_cols(df)
df_numeric = numerics.extract_numeric_cols()

missingvalue = NullCheckSubstitution(df_numeric)
df_checkNull = missingvalue.null_check()
'''
