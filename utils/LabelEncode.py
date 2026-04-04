import pandas as pd
from sklearn.preprocessing import LabelEncoder

def auto_label_encode(train_df, test_df, target_col=None, exclude_cols=None):
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    encoders = {}
    
    exclude_list = (exclude_cols if exclude_cols else []) + ([target_col] if target_col else [])
    cat_cols = [col for col in train_copy.select_dtypes(include=['object']).columns if col not in exclude_list]
    
    for col in cat_cols:
        le = LabelEncoder()
        full_series = pd.concat([train_copy[col], test_copy[col]], axis=0).astype(str).fillna('NaN')
        le.fit(full_series)
        
        train_copy[col] = le.transform(train_copy[col].astype(str).fillna('NaN'))
        test_copy[col] = le.transform(test_copy[col].astype(str).fillna('NaN'))
        
        encoders[col] = le
        
    print(f"✅ Categorical columns encoded: {cat_cols}")
    return train_copy, test_copy, encoders