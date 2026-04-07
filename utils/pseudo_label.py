def get_pseudo_labeled_data(X_train, y_train, X_test, model, threshold):
    """
    输入: 原始训练集、模型（已fit）、测试集、置信度阈值
    输出: 扩充后的 X_combined, y_combined
    """
    probs = model.predict_proba(X_test)
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)

    mask = max_probs >= threshold
    X_pseudo = X_test[mask].copy()
    y_pseudo = preds[mask]
    
    print(f">>> 阈值 {threshold}: 捕获到 {len(X_pseudo)} 条伪标签数据 (占测试集 {len(X_pseudo)/len(X_test):.1%})")

    X_combined = pd.concat([X_train, X_pseudo], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, pd.Series(y_pseudo)], axis=0).reset_index(drop=True)
    
    return X_combined, y_combined