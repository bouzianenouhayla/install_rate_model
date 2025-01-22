import joblib
from utils.utils_preprocess import *
import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import log_loss, roc_auc_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, train_test_split
from transformation_pipeline import *
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import argparse
from imblearn.over_sampling import SMOTE


# Main function to run the training
def main(args):
    
    
    X, y = load_and_process_data(args.test_file)

    #clean column names
    X.columns = [clean_column_name(col) for col in X.columns]
    
    #change data types
    with open('data/column_types.json', 'r') as json_file:
        column_types = json.load(json_file)
    
    change_data_types(X, column_types)

    #drop some columns

    columns_to_drop = ['bid_id', 'user_id', "purchase_since_install", 'install_date', 'session_start_date', 'previous_session_start_date','bid_timestamp']
    X_transfrom = X.drop(columns = columns_to_drop)
   
    
    loaded_pipeline = TransformationPipeline.load_pipeline("models/fitted_pipeline_test.pkl", "metadata/pipeline_metadata_test.json")
    test_data = loaded_pipeline.transform(X_transfrom)

    ## some additional features 

    X['hour_of_day'] = X['bid_timestamp'].dt.hour
    X['day_of_week'] = X['bid_timestamp'].dt.dayofweek
    X['bid_since_install'] = (X['bid_timestamp'] - X['install_date']).dt.days
    X['play_time_ratio'] = X['play_time'] / X['total_time'].replace(0, np.nan)
    X['game_count_time_interaction'] = X['game_count'] * X['total_time']

    X_test = pd.merge(test_data,X[["bid_timestamp", 'hour_of_day','day_of_week','bid_since_install','play_time_ratio' ,'game_count_time_interaction']], left_index=True, right_index=True).set_index("bid_timestamp")


    # load the  XGBoost model
    model = joblib.load('models/xgboost_model.pkl')
    
    # Make the predictions
    test_probs = model.predict_proba(X_test)[:, 1]

    #calibration
    iso_regressor = joblib.load('models/iso_regressor.pkl')
    

    calibrated_test_probs = iso_regressor.transform(test_probs)

    roc_auc = roc_auc_score(y, calibrated_test_probs)

    print(f'ROC AUC: {roc_auc:.4f}')
    precision, recall, thresholds = precision_recall_curve(y, calibrated_test_probs)

    thresholds_1_percent = np.arange(0, 1.01, 0.01)  # Thresholds from 0 to 1 with a step of 0.01
    precision_at_thresholds = []
    recall_at_thresholds = []
    
    # For each threshold, find the corresponding precision and recall
    for t in thresholds_1_percent:
        # Get the index of the closest threshold in the original precision-recall curve
        idx = np.searchsorted(thresholds, t)
        precision_at_thresholds.append(precision[idx])
        recall_at_thresholds.append(recall[idx])

    
    # Create a DataFrame to display precision and recall at each threshold
    precision_recall_df = pd.DataFrame({
        'Threshold': thresholds_1_percent,
        'Precision': precision_at_thresholds,
        'Recall': recall_at_thresholds
    })
    precision_recall_df.to_csv("results.csv")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the model from a parquet file.')
    parser.add_argument('test_file', type=str, help='Path to the testing parquet file')

    args = parser.parse_args()

    main(args)