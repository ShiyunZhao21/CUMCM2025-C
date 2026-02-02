from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


Q42_file_path = '女胎处理后的数据.xlsx'
Q42_data = pd.read_excel(Q42_file_path, sheet_name='Sheet1')

Q42_features = [
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

Q42_target = '染色体的非整倍体'

Q42_X = Q42_data[Q42_features]
Q42_y = Q42_data[Q42_target]

Q42_X = Q42_X.fillna(Q42_X.median())
Q42_y = Q42_y.fillna(0)

Q42_scaler = StandardScaler()
Q42_X_scaled = Q42_scaler.fit_transform(Q42_X)

Q42_smote = SMOTE(random_state=42)
Q42_X_res, Q42_y_res = Q42_smote.fit_resample(Q42_X_scaled, Q42_y)

Q42_class_weights = compute_class_weight('balanced', classes=np.unique(Q42_y_res), y=Q42_y_res)
Q42_class_weight_dict = {0: Q42_class_weights[0], 1: Q42_class_weights[1]}

Q42_X_train, Q42_X_test, Q42_y_train, Q42_y_test = train_test_split(
    Q42_X_res, Q42_y_res, test_size=0.2, random_state=42
)

Q42_cost_sensitive_model = LogisticRegression(
    class_weight=Q42_class_weight_dict,
    penalty='l1',
    solver='liblinear',
    max_iter=1000
)

Q42_cost_sensitive_model.fit(Q42_X_train, Q42_y_train)

Q42_y_probs = Q42_cost_sensitive_model.predict_proba(Q42_X_test)[:, 1]
Q42_THRESHOLD = 0.35
Q42_y_pred = (Q42_y_probs >= Q42_THRESHOLD).astype(int)

print("模型评估报告(阈值=0.35):")
print(classification_report(Q42_y_test, Q42_y_pred))
print("\n混淆矩阵(阈值=0.35):")
print(confusion_matrix(Q42_y_test, Q42_y_pred))

joblib.dump(Q42_cost_sensitive_model, 'cost_sensitive_model.pkl')
joblib.dump(Q42_scaler, 'scaler.pkl')
print("模型和标准化器已保存为 cost_sensitive_model.pkl 和 scaler.pkl")

Q42_sample = Q42_X.iloc[[0]]
Q42_sample_scaled = Q42_scaler.transform(Q42_sample)
Q42_sample_prob = Q42_cost_sensitive_model.predict_proba(Q42_sample_scaled)[0][1]
Q42_prediction = 1 if Q42_sample_prob >= Q42_THRESHOLD else 0
print(f"\n样本预测结果(阈值=0.35): {'有病' if Q42_prediction == 1 else '无病'}")
print(f"预测概率: {Q42_sample_prob:.4f}")

print("\n模型系数:")
Q42_coefficients = pd.DataFrame({
    '特征': Q42_features,
    '系数': Q42_cost_sensitive_model.coef_[0]
})
print(Q42_coefficients.sort_values(by='系数', ascending=False))
