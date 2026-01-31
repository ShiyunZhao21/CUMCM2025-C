import pandas as Q13pd
import seaborn as Q13sns
import matplotlib.pyplot as Q13plt

Q13df = Q13pd.read_excel("处理后数据.xlsx")

Q13df = Q13df[['检测孕周', '年龄', '身高', '体重', '孕妇BMI', 'Y染色体浓度']]
Q13df = Q13df.rename(columns={
    '检测孕周': 'Gestation_Weeks',
    '年龄': 'Age',
    '身高': 'Height',
    '体重': 'Weight',
    '孕妇BMI': 'BMI',
    'Y染色体浓度': 'Y_Chromosome'})

Q13corr = Q13df.corr(method='spearman')

Q13plt.figure(figsize=(10, 8))
Q13sns.set(font_scale=1.2)
Q13heatmap = Q13sns.heatmap(
    Q13corr,
    vmin=-1,
    vmax=1,
    annot=True,
    fmt=".3f",
    cmap='coolwarm',
    linewidths=0.5,
    annot_kws={"size": 12}
)

Q13plt.title('Spearman Correlation Heatmap', fontsize=16)
Q13heatmap.set_xticklabels(Q13heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
Q13heatmap.set_yticklabels(Q13heatmap.get_yticklabels(), rotation=0)

Q13plt.tight_layout()
Q13plt.show()
