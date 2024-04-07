import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings 
warnings.filterwarnings('ignore')

# 이상치 데이터 확인 및 대체
class OutlinerSeekFind:
    def __init__(self, df):
        self.df_outliner = df

    def make_melt_data(self):
        melt_data = pd.melt(self.df_outliner, var_name='col', value_name='value')
        self.show_chart(melt_data)
        return melt_data
    
    def show_chart(self, melt_data):
        plt.figure(figsize=(15,7))
        sns.boxplot(x='col', y='value', data=melt_data)
        num = len(melt_data['col'].unique())
        plt.xticks(range(num), melt_data['col'].unique())
        plt.show()

    def outliners_iqr_index(self):
        outliner_df = self.df_outliner
        cols_ls = outliner_df.columns
        print(cols_ls)
        ls_outliner = []
        ls_cols = []
        ls_upper = []
        for col in cols_ls:
            quartile_1, quartile_3 = np.percentile(outliner_df[col], [25,75])
            iqr = quartile_3 - quartile_1
            lower_whis = quartile_1 - (iqr*1.5)
            upper_whis = quartile_3 + (iqr*1.5)
            length = len(list(outliner_df[(outliner_df[col] < lower_whis)|(outliner_df[col] > upper_whis)].index))
            if length != 0:
                ls_outliner.append(list(outliner_df[(outliner_df[col] < lower_whis)|(outliner_df[col] > upper_whis)].index))
                ls_cols.append(col)
                ls_upper.append(upper_whis)
            else:
                pass
        return ls_outliner, ls_cols, ls_upper

    def make_subs_data(self):
        ls_outliner, ls_cols, ls_upper = self.outliners_iqr_index()
        length = len(ls_upper)
        if length != 0:
            subs_data = pd.DataFrame()
            for i in range(length):
                outliner_indx = ls_outliner[i]
                col_name = ls_cols[i]
                upper = ls_upper[i]
                self.df_outliner[col_name] = self.subsititute_outliner_value(col_name, upper)
            result = self.df_outliner
        else:
            result = pd.DataFrame()
        return result
            
    def subsititute_outliner_value(self, target_col_name, target_value):
        target_data = self.df_outliner.copy()
        #method = int(input('기준을 설정해주세요: 1.초과, 2.이상, 3.이하, 4.미만, 5. 동일'))
        method = 1
        if method == 1:
            outliner_index = target_data[target_data[target_col_name]>target_value].index
        elif method == 2:
            outliner_index = target_data[target_data[target_col_name]>=target_value].index
        elif method == 3:
            outliner_index = target_data[target_data[target_col_name]<=target_value].index
        elif method == 4:
            outliner_index = target_data[target_data[target_col_name]<target_value].index
        elif method == 5:
            outliner_index = target_data[target_data[target_col_name]==target_value].index
        #--
        #target_data.loc[outliner_index, target_col_name] = target_data[target_col_name].median()
        #--
        target_data.loc[outliner_index, target_col_name] = None
        return target_data[target_col_name]
