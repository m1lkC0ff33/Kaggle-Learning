def run_kfold_regression(X, y, X_test, model_obj, n_splits=5):
    """
    通用的 K-Fold 训练框架
    model_obj: 未 fit 的模型实例，如 XGBRegressor(**params)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 克隆原始模型防止参数残留
        from sklearn.base import clone
        model = clone(model_obj)
        
        # 这里的 fit 参数可以根据不同模型做适配
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=100, verbose=False)
        
        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / n_splits
        
        rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
        print(f"Fold {fold+1} RMSE: {rmse:.4f}")
        
    print(f"Overall OOF RMSE: {np.sqrt(mean_squared_error(y, oof_preds)):.4f}")
    return oof_preds, test_preds