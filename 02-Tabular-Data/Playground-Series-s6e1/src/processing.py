import pandas as pd
import numpy as np

def process_data(df, is_train=True):
    curr_df = df.copy()

    # Cut down id and course, useless features
    drop_cols = ['id', 'course']
    curr_df.drop(columns=[c for c in drop_cols if c in curr_df.columns], inplace=True)

    facility_map = {'low': 1, 'medium': 2, 'high': 3}
    curr_df['facility_rating_num'] = curr_df['facility_rating'].map(facility_map)

    diff_map = {'easy': 0, 'moderate': 1, 'hard': 2}
    curr_df['exam_difficulty_enc'] = curr_df['exam_difficulty'].map(diff_map)

    quality_map = {'poor': 1, 'average': 2, 'good': 3}
    curr_df['sleep_quality_val'] = curr_df['sleep_quality'].map(quality_map)

    # Feature Engineering
    curr_df['sleep_index'] = curr_df['sleep_hours'] * curr_df['sleep_quality_val']
    curr_df['efficiency_index'] = (curr_df['class_attendance'] / 100) * curr_df['study_hours']

    # One-Hot Encoding
    ohe_cols = ['gender', 'internet_access', 'study_method']
    curr_df = pd.get_dummies(curr_df, columns=[c for c in ohe_cols if c in curr_df.columns], 
                            drop_first=True, prefix_sep='_')
    bool_cols = curr_df.select_dtypes(include=['bool']).columns
    curr_df[bool_cols] = curr_df[bool_cols].astype(int)

    curr_df.columns = [col.replace(' ', '_').replace('-', '_') for col in curr_df.columns]

    final_drop = ['facility_rating', 'exam_difficulty', 'sleep_quality']
    curr_df.drop(columns=[c for c in final_drop if c in curr_df.columns], inplace=True)
    
    return curr_df