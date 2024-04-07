import pandas as pd
import anal_preprocess as ap

class GetNumericCategoryAllData:
    def __init__(self, input_data, df):
        self.input_data = input_data
        self.df = df
    def main(self, col_out):
        input_data = self.input_data
        df = self.df
        dp = ap.DataProcess(input_data, df)
        dp.shapiro(col_out)
        dp.extract_numeric_categoy_cols()
        numeric_data = dp.nemeric_check_cols()
        dp.numeric_correlation_cols
        dp.category_preprocess()
        category_data, ctg_data = dp.category_tranformed_numeric()
        if len(category_data) == 0:
            print('범주형 데이터 없음')
            catg_data = category_data
        else:
            catg = ctg_data.columns
            ori_catg = input_data.columns
            intersection_catg = catg.intersection(ori_catg)
            catg_data = ctg_data[intersection_catg]
        return numeric_data, catg_data, ctg_data