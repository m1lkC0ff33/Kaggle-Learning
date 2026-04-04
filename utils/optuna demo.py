import optuna

def objective(trial):
    # 1. 定义搜索范围
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
    }
    
    # 2. 运行 K-Fold (直接复用你之前的逻辑)
    # 这里返回 5-Fold 的平均验证集 RMSE
    score = run_kfold_with_params(X, y, param) 
    return score

# 3. 开启“全自动”寻找最优解
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) # 跑 50 次实验

print(f"🏆 最优参数: {study.best_params}")