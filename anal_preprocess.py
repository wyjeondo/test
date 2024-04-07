import pandas as pd
import numpy as np
import utils_numeric
import utils_null 
import utils_outliner
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)

class DataProcess:
    def __init__(self, dataframe_input, dataframe_ori):
        self.origin = dataframe_ori
        self.data = dataframe_input

    def shapiro(self, col):
        if len(col) == 0:
            print('종속변수 없음')
        else:
            print(stats.shapiro(self.origin[col]))

    def extract_numeric_categoy_cols(self):
        input_data = self.data
        ncwe = utils_numeric.NumericColsWEncoding(input_data)
        self.numeric_df = ncwe.extract_numeric_cols()
        self.catgory_df = ncwe.extract_category_cols()
    
    def nemeric_check_cols(self):
        # 1차결측치 검증
        ncs = utils_null.NullCheckSubstitution(self.numeric_df)
        numeric_df_check = ncs.null_check()
        # 이상치 검증
        osf = utils_outliner.OutlinerSeekFind(numeric_df_check)
        temp_numeric_df_null = osf.make_subs_data()
        length_outliner = len(temp_numeric_df_null)
        if length_outliner != 0 :
            print('이상치를 결측치로 대체한 후 KNN 방식으로 대체 ... 2차결측치 검증 진행')
            ncs = utils_null.NullCheckSubstitution(temp_numeric_df_null)
            self.numeric_df_completed = ncs.null_check()
        else: 
            print('이상치 없음')
            self.numeric_df_completed = numeric_df_check  
        result = self.numeric_df_completed
        return result

    def numeric_correlation_cols(self):
        input_data = self.numeric_df
        df_cor = numeric_df_completed.corr(method='pearson')
        sns.heatmap(df_cor,
                    xticklabels = df_cor.columns,
                    yticklabels = df_cor.columns,
                    cmap = 'RdBu_r',
                    linewidth = 3)

    def category_preprocess(self):
        self.cat_df = self.catgory_df.copy()
        ncwe_cat = utils_numeric.NumericColsWEncoding(self.cat_df)
        self.trf_cat_df = ncwe_cat.make_encoded_dataframe(self.cat_df)
        self.null_cols = self.cat_df.columns[self.cat_df.isnull().sum() != 0]
        self.null_cols_encd = [col+'_encd' for col in self.null_cols]

    def dict_zip_data(self, dict_df, col1, col2):
        dict_data = dict_df[[col1,col2]]
        dict_data = dict_data.drop_duplicates()
        dict_col1 = dict_data[col1].values
        dict_col2 = dict_data[col2].values
        dict_zip_data = dict(zip(dict_col1, dict_col2))
        return dict_zip_data      

    def make_couple_df(self, data, col1, col2):
        coup_df = data[[col1, col2]]
        coup_df = coup_df.fillna('null_check')
        coup_idx = coup_df[coup_df[col1]=='null_check'].index
        coup_df.loc[coup_idx, col2] = None
        return coup_df       

    def category_tranformed_numeric(self):
        null_cols = self.null_cols
        null_cols_encd = self.null_cols_encd
        length = len(null_cols)
        ls_df = []
        ls_zip = []
        null_catg_df = pd.DataFrame()
        for i in range(length):
            col1 = null_cols[i]
            col2 = null_cols_encd[i]
            part_df = self.make_couple_df(self.cat_df, col1, col2)
            null_catg_df[col2] = part_df[col2].values
            ls_df.append(part_df)
            part_df = part_df.drop_duplicates()
            zip_df = self.dict_zip_data(part_df, col2, col1)
            ls_zip.append(zip_df)

        # 1차결측치 검증
        ncs_cat = utils_null.NullCheckSubstitution(null_catg_df)
        null_cat_df_check = ncs_cat.null_check()
        null_cat_df_check = null_cat_df_check.round(0)
        # knn 대체 기준 값으로 기존 'null_check' 대응하는 기존 값을 삭제하고 기준 값으로 대체한다.
        for col in null_cols_encd:
            self.cat_df[col] = null_cat_df_check[col].values
            self.trf_cat_df[col] = null_cat_df_check[col].values
        # 매칭을 통해서 범주형을 채운다.
        length = len(null_cols_encd)
        for i in range(length):
            self.cat_df[null_cols[i]] = self.cat_df[null_cols_encd[i]].map(ls_zip[i])
        catgory_df_completed = self.trf_cat_df.round(0)
        result = catgory_df_completed
        return result, self.cat_df
'''(사용법)
dp = DataProcess(input_data)
dp.shapiro()
dp.extract_numeric_categoy_cols()
numeric_data = dp.nemeric_check_cols()
dp.numeric_correlation_cols
dp.category_preprocess()
category_data, ctg_data = dp.category_tranformed_numeric()
'''