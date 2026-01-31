import pandas as Q11pd

Q11df = Q11pd.read_excel("问题一新.xlsx")

Q11df['检测孕周'] = Q11df['检测孕周'].replace('16W\+1', '16.1428571428571', regex=True).astype(float)

Q11df = Q11df[Q11df['GC含量'] >= 0.4]

def remove_outliers(Q11df, Q11col_name):
    Q11Q1 = Q11df[Q11col_name].quantile(0.25)
    Q11Q3 = Q11df[Q11col_name].quantile(0.75)
    Q11IQR = Q11Q3 - Q11Q1
    Q11lower_bound = Q11Q1 - 1.5 * Q11IQR
    Q11upper_bound = Q11Q3 + 1.5 * Q11IQR
    return Q11df[(Q11df[Q11col_name] >= Q11lower_bound) & (Q11df[Q11col_name] <= Q11upper_bound)]

Q11df = remove_outliers(Q11df, 'Y染色体浓度')
Q11df = remove_outliers(Q11df, '孕妇BMI')
Q11df = remove_outliers(Q11df, '检测孕周')

Q11output_path = '处理后数据.xlsx'
Q11df.to_excel(Q11output_path, index=False)
