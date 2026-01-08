import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 目标变量分布图 (Target Distribution)
# 什么时候用：训练前必看。看标签是否正态分布，是否有极端离群点。
# 如果呈现长尾分布，通常需要对 y 做 np.log1p 转换。
# ==========================================
plt.figure(figsize=(10, 5))
sns.histplot(train['exam_score'], kde=True, color='blue')
plt.title('Target Distribution: Exam Score')
plt.xlabel('Score')
plt.show()

# ==========================================
# 2. 相关性热力图 (Correlation Heatmap)
# 什么时候用：特征工程后。看新特征与标签的相关性，以及特征间是否多重共线性。
# 相关性绝对值 > 0.8 的两个特征，通常可以删掉其中一个。
# ==========================================
plt.figure(figsize=(12, 10))
corr = train_processed.corr()
mask = np.triu(np.ones_like(corr, dtype=bool)) # 只显示下三角，避免重复
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# ==========================================
# 3. 箱线图 (Feature vs Target Boxplot)
# 什么时候用：分析类别特征（如性别、父母学历）对分数的影响。
# 如果箱子的中位数（线）明显错位，说明这个特征非常有区分度。
# ==========================================
plt.figure(figsize=(12, 6))
sns.boxplot(x='parental_education', y='exam_score', data=train)
plt.xticks(rotation=45)
plt.title('Parental Education vs Exam Score')
plt.show()

# ==========================================
# 4. 特征重要性排序 (Feature Importance)
# 什么时候用：模型跑完后。决定下一轮特征工程删减哪些“垃圾特征”。
# ==========================================
import pandas as pd
# 假设 model 是你训练好的 xgb 或 lgb
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
feat_imp.head(15).plot(kind='barh', figsize=(10, 8), color='teal')
plt.title('Top 15 Feature Importance')
plt.gca().invert_yaxis() # 让最重要的排在最上面
plt.show()