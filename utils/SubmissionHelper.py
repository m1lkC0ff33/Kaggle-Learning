import pandas as pd
import time
import os

def save_submission(preds, test_df, id_col, target_col, save_dir='../submissions', prefix='sub'):
    timestamp = time.strftime("%m%d_%H%M")
    filename = os.path.join(save_dir, f"{prefix}_{timestamp}.csv")
    
    os.makedirs(save_dir, exist_ok=True)
    
    submission = pd.DataFrame({
        id_col: test_df[id_col],
        target_col: preds
    })
    
    submission.to_csv(filename, index=False)
    
    if os.path.exists(filename):
        print("-" * 30)
        print(f"✅ [SUCCESS] Submission file generated!")
        print(f"📍 Location: {os.path.abspath(filename)}")
        print(f"📊 Shape: {submission.shape}")
        print("-" * 30)
    else:
        print(f"❌ [ERROR] Failed to save file at {filename}")
        
    return filename