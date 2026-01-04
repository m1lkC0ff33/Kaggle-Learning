import matplotlib.pyplot as plt
import seaborn as sns

def plot_xgb_learning_curve(model):
    """
    专门为 XGBoost 绘制学习曲线的函数
    """
    # 1. 提取结果
    results = model.evals_result()
    
    # 2. 这里的 key 取决于你在 fit 时 eval_set 的顺序
    # 假设我们习惯：第一个是 Train (validation_0), 第二个是 Val (validation_1)
    train_rmse = results['validation_0']['rmse']
    val_rmse = results['validation_1']['rmse']
    
    epochs = len(train_rmse)
    x_axis = range(0, epochs)
    
    # 3. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_rmse, label='Train (Learning)')
    plt.plot(x_axis, val_rmse, label='Validation (Simulated Exam)')
    
    # 标注最佳迭代点
    if hasattr(model, 'best_iteration'):
        plt.axvline(x=model.best_iteration, color='red', linestyle='--', 
                    label=f'Best Iteration: {model.best_iteration}')
        plt.scatter(model.best_iteration, model.best_score, color='red')
    
    plt.title('XGBoost Learning Curve: Training vs Validation')
    plt.xlabel('Number of Trees (Epochs)')
    plt.ylabel('RMSE (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_feature_importance(model, top_n=20):
    """
    看看哪些特征对模型贡献最大
    """
    import pandas as pd
    # 获取特征重要性数据
    importance = model.feature_importances_
    features = model.get_booster().feature_names
    
    df_importance = pd.DataFrame({'feature': features, 'importance': importance})
    df_importance = df_importance.sort_values(by='importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=df_importance, palette='viridis')
    plt.title(f'Top {top_n} Important Features')
    plt.show()