import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, features, top_n=15):
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n))
    plt.title(f'Top {top_n} Features')
    plt.show()