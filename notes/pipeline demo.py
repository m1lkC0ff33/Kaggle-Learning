from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

# 1. 定义哪些列需要什么处理
numeric_features = ['study_hours', 'attendance', 'efficiency_index']
categorical_features = ['gender', 'parental_education']

# 2. 为数值列创建子流水线 (填补缺失值 -> 标准化)
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 3. 为类别列创建子流水线 (填补缺失值 -> One-Hot编码)
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 4. 合并所有预处理逻辑 (ColumnTransformer)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('cat', cat_transformer, categorical_features)
    ]
)

# 5. 组合成终极 Pipeline (预处理 -> 模型)
# 这样做的好处：当你调用 pipeline.predict(test_raw) 时，
# 它会自动对测试集做一模一样的转换，绝不会出现列不对应的错误。
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=1000, learning_rate=0.01))
])

# 6. 训练与使用
# 此时传进去的是原始 DataFrame，不是处理后的
full_pipeline.fit(train_raw, y_train) 
preds = full_pipeline.predict(test_raw)

# 7. 查看 Pipeline 的中间步骤 (比如看特征重要性)
# 因为有 OneHot，特征名字会变，查看起来稍显复杂：
ohe_features = full_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
all_feat_names = numeric_features + list(ohe_features)