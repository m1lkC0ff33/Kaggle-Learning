def check_data_quality(df):
    summary = pd.DataFrame({
        'Type': df.dtypes,
        'Missing': df.isnull().sum(),
        'Missing%': (df.isnull().sum() / len(df)) * 100,
        'Unique': df.nunique()
    })
    print(f"Dataset Shape: {df.shape}")
    print(f"Duplicates: {df.duplicated().sum()}")
    return summary