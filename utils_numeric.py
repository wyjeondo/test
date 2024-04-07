import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings 
warnings.filterwarnings('ignore')

# 연속형(수치형)과 범주형 컬럼 구분 및 인코딩 작업
class NumericColsWEncoding:
    def __init__(self, df):
        self.df = df
        
    def num_columns(self): # 연속형 / 범주형 데이터 컬럼 리스트
        all_columns = self.df.columns.tolist()
        num_columns = self.df._get_numeric_data().columns.tolist()
        cat_columns = list(set(all_columns) - set(num_columns))
        return num_columns, cat_columns

    def extract_numeric_cols(self): # 연속형 데이터 추출
        num_cols = self.num_columns()[0]
        df_numeric = self.df[num_cols]
        return df_numeric
    
    def extract_category_cols(self): # 범주형 데이터 추출
        cat_cols = self.num_columns()[1]
        df_category = self.df[cat_cols]
        return df_category

    def label_encoding(self, data, col): # 범주형 변수 인코딩
        le = LabelEncoder()
        le = le.fit(data[col])
        col_new = col+'_encd'
        data[col_new] = le.transform(data[col])
        return data[col_new]

    def onehot_encoding(self, data):
        col_name = data.columns
        data = pd.get_dummies(data=data, columns=col_name)
        data = data.fillna(0)
        return data

    def classified_input_data(self, input_data, type_):
        if type_ == 1:
            encoded_data = self.onehot_encoding(input_data)
        elif type_ == 2:
            encoded_data = pd.DataFrame()
            cols_list = input_data.columns
            for col in cols_list:
                result = self.label_encoding(input_data, col)
                encoded_data = pd.concat([encoded_data, result], axis=1)
        else:
            print('Wrong input variables.. input_data or type_')
            encoded_data = pd.DataFrame()
        return encoded_data
        
    def make_encoded_dataframe(self, input_data):
        flag = 1
        encoded_df = pd.DataFrame()
        while flag:
            #type_ = int(input('Select type of encoding : 1. onehot 2. label ...'))
            type_ = 2
            if (type_ == 1) | (type_==2):
                encoded_df = self.classified_input_data(input_data, type_)
                flag = 0
            else:
                print('pleas select type of encoding ...')
        return encoded_df
'''
# 수치형 컬럼 추출
test_class = NumericColsWEncoding(dataframe)
test = test_class.extract_numeric_cols()

# 인코딩
test_class = NumericColsWEncoding(dataframe)
test = test_class.extract_category_cols()
df = dataframe[test.columns]
test_df = test_class.make_encoded_dataframe(df)
'''