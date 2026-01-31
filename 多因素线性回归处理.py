import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

q1q1q1q1file_path = r"D:\HuaweiMoveData\Users\holala\Desktop\处理后数据.xlsx"
q1q1q1q1df = pd.read_excel(q1q1q1q1file_path)

q1q1q1q1df = q1q1q1q1df.replace([np.inf, -np.inf], np.nan)

q1q1q1q1df_clean = q1q1q1q1df.dropna()

q1q1q1q1target_column = 'Y染色体浓度'
q1q1q1q1y = q1q1q1q1df_clean[q1q1q1q1target_column]

q1q1q1q1X_columns = ['年龄', '检测孕周', '怀孕次数', '生产次数', 'BMI', '标记后的IVF妊娠']

q1q1q1q1X = q1q1q1q1df_clean[q1q1q1q1X_columns]

q1q1q1q1X_clean = q1q1q1q1X.copy()
q1q1q1q1y_clean = q1q1q1q1y.copy()

q1q1q1q1mask = ~(q1q1q1q1X_clean.isnull().any(axis=1) | q1q1q1q1y_clean.isnull() |
         np.isinf(q1q1q1q1X_clean).any(axis=1) | np.isinf(q1q1q1q1y_clean))

q1q1q1q1X_final = q1q1q1q1X_clean[q1q1q1q1mask]
q1q1q1q1y_final = q1q1q1q1y_clean[q1q1q1q1mask]

q1q1q1q1X_with_const = sm.add_constant(q1q1q1q1X_final)
q1q1q1q1model = sm.OLS(q1q1q1q1y_final, q1q1q1q1X_with_const).fit()

q1q1q1q1coefficients = q1q1q1q1model.params
q1q1q1q1p_values = q1q1q1q1model.pvalues

q1q1q1q1results_df = pd.DataFrame({
    '变量': ['截距'] + q1q1q1q1X_columns,
    '系数': q1q1q1q1coefficients.values,
    '标准误': q1q1q1q1model.bse.values,
    't值': q1q1q1q1model.tvalues.values,
    'p值': q1q1q1q1p_values.values,
    '显著性': ['***' if q1q1q1q1p < 0.001 else '**' if q1q1q1q1p < 0.01 else '*' if q1q1q1q1p < 0.05 else '' for q1q1q1q1p in q1q1q1q1p_values.values]
})

print(f"详细回归结果:")
print(q1q1q1q1results_df.to_string(index=False, float_format='%.6f'))
