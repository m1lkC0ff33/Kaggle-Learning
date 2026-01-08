import time

def save_submission(preds, test_df, id_col='id', target_col='exam_score', prefix='sub'):
    """自动带时间戳保存提交文件"""
    timestamp = time.strftime("%m%d_%H%M")
    filename = f"../submissions/{prefix}_{timestamp}.csv"
    
    submission = pd.DataFrame({
        id_col: test_df[id_col],
        target_col: preds
    })
    
    # 确保目录存在
    os.makedirs('../submissions', exist_ok=True)
    submission.to_csv(filename, index=False)
    print(f"Submission saved to: {filename}")