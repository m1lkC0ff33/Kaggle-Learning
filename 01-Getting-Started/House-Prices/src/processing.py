import pandas as pd
import numpy as np

def preprocess_data(train_df, test_df):
    # 合并数据
    all_data = pd.concat((train_df.drop(['SalePrice'], axis=1), test_df)).reset_index(drop=True)
    
    # 1. 类别型填充 "None"
    cols_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    for col in cols_none:
        all_data[col] = all_data[col].fillna('None')

    # 2. 数值型填充 0 (包含你报错的 MasVnrArea 和 GarageYrBlt)
    cols_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                 'MasVnrArea']
    for col in cols_zero:
        all_data[col] = all_data[col].fillna(0)

    # 3. LotFrontage 智能填充 + 兜底
    # 先按邻里中位数填
    all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    # 如果还有漏网的（某邻里全是NaN），用全局中位数填
    all_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data["LotFrontage"].median())

    # 4. 权重映射
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    for col in qual_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].map(quality_map)

    # 5. 特征构造
    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    # 6. 最后的终极兜底：对所有数值列进行中位数填充（防止测试集有诡异缺失）
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].median())

    # 7. 独热编码
    all_data = pd.get_dummies(all_data)
    
    # 拆分
    n_train = len(train_df)
    train_final = all_data[:n_train].copy()
    test_final = all_data[n_train:].copy()
    train_final['SalePrice'] = np.log1p(train_df['SalePrice'])
    
    return train_final, test_final