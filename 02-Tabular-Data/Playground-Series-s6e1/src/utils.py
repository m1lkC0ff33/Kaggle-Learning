import pandas as pd
import os
from datetime import datetime

def save_submission(test_preds, test_raw, folder='../submissions'):
    """
    工程化提交函数：
    1. 自动提取原始 ID
    2. 确保输出格式符合 sample_submission
    3. 自动生成带时间戳的文件名防止覆盖
    """
    # 1. 提取 ID 列 (Kaggle 要求提交的文件通常包含 ID 和 Target)
    if 'id' not in test_raw.columns:
        raise ValueError("原始测试集中缺少 'id' 列，无法生成提交文件。")
    
    # 2. 构建 DataFrame
    submission = pd.DataFrame({
        'id': test_raw['id'],
        'exam_score': test_preds
    })
    
    # 3. 确保目录存在
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 4. 生成文件名：包含时间戳，方便追溯不同的实验版本
    timestamp = datetime.now().strftime("%m%d_%H%M")
    file_name = f"sub_{timestamp}.csv"
    file_path = os.path.join(folder, file_name)
    
    # 5. 保存
    submission.to_csv(file_path, index=False)
    print(f"🚀 提交文件已生成: {file_path}")
    return file_path