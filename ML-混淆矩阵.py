import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

Q41_file_path = '女胎处理后的数据.xlsx'
Q41_data = pd.read_excel(Q41_file_path, sheet_name='Sheet1')

plt.rcParams['font.sans-serif'] = ['SimHei']

Q41_features = [
    '孕妇BMI',
    '原始读段数',
    '在参考基因组上比对的比例',
    '重复读段的比例',
    '唯一比对的读段数',
    '13号染色体的Z值',
    'X染色体的Z值',
    '13号染色体的GC含量',
    'GC含量',
    '被过滤掉读段数的比例',
    '18号染色体的GC含量',
    '18号染色体的Z值',
    '21号染色体的GC含量',
    '21号染色体的Z值'
]

Q41_target = '染色体的非整倍体'

Q41_X = Q41_data[Q41_features]
Q41_y = Q41_data[Q41_target]

Q41_X = Q41_X.fillna(Q41_X.median())
Q41_y = Q41_y.fillna(0)

Q41_scaler = StandardScaler()
Q41_X_scaled = Q41_scaler.fit_transform(Q41_X)

Q41_smote = SMOTE(random_state=42)
Q41_X_res, Q41_y_res = Q41_smote.fit_resample(Q41_X_scaled, Q41_y)

Q41_class_weights = compute_class_weight('balanced', classes=np.unique(Q41_y_res), y=Q41_y_res)
Q41_class_weight_dict = {0: Q41_class_weights[0], 1: Q41_class_weights[1]}

Q41_X_train, Q41_X_test, Q41_y_train, Q41_y_test = train_test_split(
    Q41_X_res, Q41_y_res, test_size=0.2, random_state=42
)

Q41_cost_sensitive_model = LogisticRegression(
    class_weight=Q41_class_weight_dict,
    penalty='l1',
    solver='liblinear',
    max_iter=1000
)

Q41_cost_sensitive_model.fit(Q41_X_train, Q41_y_train)

Q41_y_probs = Q41_cost_sensitive_model.predict_proba(Q41_X_test)[:, 1]
Q41_THRESHOLD = 0.35
Q41_y_pred = (Q41_y_probs >= Q41_THRESHOLD).astype(int)

print("模型评估报告(阈值=0.35):")
print(classification_report(Q41_y_test, Q41_y_pred))
print("\n混淆矩阵(阈值=0.35):")
Q41_cm = confusion_matrix(Q41_y_test, Q41_y_pred)
print(Q41_cm)

plt.figure(figsize=(8, 6))
sns.heatmap(Q41_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['无病', '有病'],
            yticklabels=['无病', '有病'])
plt.title('混淆矩阵 (阈值=0.35)')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

joblib.dump(Q41_cost_sensitive_model, 'cost_sensitive_model.pkl')
joblib.dump(Q41_scaler, 'scaler.pkl')
print("模型和标准化器已保存为 cost_sensitive_model.pkl 和 scaler.pkl")

Q41_sample = Q41_X.iloc[[0]]
Q41_sample_scaled = Q41_scaler.transform(Q41_sample)
Q41_sample_prob = Q41_cost_sensitive_model.predict_proba(Q41_sample_scaled)[0][1]
Q41_prediction = 1 if Q41_sample_prob >= Q41_THRESHOLD else 0
print(f"\n样本预测结果(阈值=0.35): {'有病' if Q41_prediction == 1 else '无病'}")
print(f"预测概率: {Q41_sample_prob:.4f}")

print("\n模型系数:")
Q41_coefficients = pd.DataFrame({
    '特征': Q41_features,
    '系数': Q41_cost_sensitive_model.coef_[0]
})
print(Q41_coefficients.sort_values(by='系数', ascending=False))
