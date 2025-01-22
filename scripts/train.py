import joblib
from utils_preprocess import *
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
    # Load the training data
    
    X, y = load_and_process_data(args.train_file)

    #clean column names
    X.columns = [clean_column_name(col) for col in X.columns]

    #change data types
    with open('data/column_types.json', 'r') as json_file:
        column_types = json.load(json_file)
    
    change_data_types(X, column_types)

    #drop some columns

    columns_to_drop = ['bid_id', 'user_id', "purchase_since_install", 'install_date', 'session_start_date', 'previous_session_start_date','bid_timestamp']
    X_transfrom = X.drop(columns = columns_to_drop)
    # Fit the pipeline for data transformation

    with open('transformations_metadata.json', 'r') as json_file:
        transformations_metadata = json.load(json_file)

    
    pipeline = TransformationPipeline()


    # Set transformations metadata before adding steps
    pipeline.transformations_metadata = transformations_metadata
    
    function_map = {
   
    "expand_all_column": ExpandDictStep(),
    "expand_top_apps": JSONToFeaturesStep(top_n=50),
    "aggergate_impressions": AggregateImpressionsStep(),
    "encoder": CategoricalEncodingStep()
   
    }
    for group_name, group_info in transformations_metadata.items():
        for transformation_name in group_info["transformations"]:
            transformation = function_map[transformation_name]
            if hasattr(transformation, "fit_transform"): 
                pipeline.add_step(
                    f"{transformation_name}",
                    transformation,
                    group_info["columns"]
                )
            else:  # Handles custom function-based transformations
                for column in group_info["columns"]:
                    pipeline.add_step(
                        f"{transformation_name}",
                        FunctionStep(transformation),
                        [column]
                    )
    train_data = pipeline.fit_transform(X_transfrom)
   
    # Save the pipeline and metadata
    pipeline.save_pipeline("models/fitted_pipeline.pkl", "metadata/pipeline_metadata.json")
    

    
    ## some additional features 

    X['hour_of_day'] = X['bid_timestamp'].dt.hour
    X['day_of_week'] = X['bid_timestamp'].dt.dayofweek
    X['bid_since_install'] = (X['bid_timestamp'] - X['install_date']).dt.days
    X['play_time_ratio'] = X['play_time'] / X['total_time'].replace(0, np.nan)
    X['game_count_time_interaction'] = X['game_count'] * X['total_time']

    X_train = pd.merge(train_data,X[["bid_timestamp", 'hour_of_day','day_of_week','bid_since_install','play_time_ratio' ,'game_count_time_interaction']], left_index=True, right_index=True).set_index("bid_timestamp")



    ## Oversampling of the data
    # Initialize SMOTE
    smote = SMOTE(sampling_strategy=0.15, random_state=42)
    
    # Apply SMOTE to the training data
    X_train_res, y_train_res = smote.fit_resample(X_train.fillna(0), y)

    # Combine the resampled data back into a DataFrame
    train_data_resampled = pd.concat([X_train_res, y_train_res], axis=1)

    # Sort the resampled data by the 'bid_timestamp' column
    train_data_resampled = train_data_resampled.sort_index()

    X_train_resampled = train_data_resampled.drop(columns=['install_label'])
    y_train_resampled = train_data_resampled['install_label']

    # Define the XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.01,
        max_depth=6,
        objective='binary:logistic',  # For binary classification
        use_label_encoder=False,     # Avoid label encoding warning
        random_state=42,
        subsample = 0.8,
        colsample_bytree = 0.8,
        #scale_pos_weight=10,
        alpha=0.1,  # L1 regularization (Lasso)
        lambda_=1,  # L2 regularization (Ridge)
    )
    
    # Train the model
    model.fit(
        X_train_resampled,
        y_train_resampled,  
        verbose=True
    )

    

    # Save the trained model using joblib
    joblib.dump(model, 'models/xgboost_model.pkl')
    train_probs = model.predict_proba(X_train_resampled)[:, 1]
    #calibration
    iso_regressor = IsotonicRegression(out_of_bounds='clip')  
    iso_regressor.fit(train_probs, y_train_resampled)


    # Save the trained isotonic regressor
    joblib.dump(iso_regressor, 'models/iso_regressor.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model from a parquet file.')
    parser.add_argument('train_file', type=str, help='Path to the training parquet file')

    args = parser.parse_args()

    main(args)