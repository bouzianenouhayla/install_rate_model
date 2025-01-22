import pandas as pd
from utils.utils_preprocess import *
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
import pandas as pd
import numpy as np




class CategoricalEncodingStep:
    def __init__(self,col_name= True, high_cardinality_threshold=100, n_features=50):
      
        self.high_cardinality_threshold = high_cardinality_threshold
        self.n_features = n_features
        self.col_name = True
 
    
    def fit(self, data):
        cardinality = data.nunique()
        if cardinality <= self.high_cardinality_threshold:
            # Use OneHotEncoder
            encoder = OneHotEncoder(handle_unknown='ignore')
            encoder.fit(np.array([data.tolist()]).T)
        else:
            # Use FeatureHasher
            encoder = FeatureHasher(n_features=self.n_features, input_type='string')
            
        
        self.encoder = encoder
    
    def transform(self, data, col_name):
       
        transformed_data = data.copy()
        
        if isinstance(self.encoder, OneHotEncoder):
            encoded = self.encoder.transform(np.array([data.tolist()]).T)
            encoded_df = pd.DataFrame(
                encoded.toarray(),
                columns=self.encoder.get_feature_names_out([col_name]),
                index=data.index
            )
            # Replace column with its encoded features
            transformed_data = encoded_df
        elif isinstance(self.encoder, FeatureHasher):
            # Convert column values to strings for hashing
            #print(np.array([data.tolist()]).T)
            hashed = self.encoder.transform(np.array([data.tolist()]).T)
            # Replace column with hashed features
            hashed_df = pd.DataFrame(
                    hashed.toarray(),
                    columns=[f"{col_name}_hash_{i}" for i in range(hashed.shape[1])],
                    index=data.index
            )
                
            transformed_data = hashed_df
        return transformed_data
    
    def fit_transform(self, data, columns):
       
        self.fit(data, columns)
        return self.transform(data)



class AggregateImpressionsStep:
    def fit(self, data):
        pass

    def transform(self, data):
        # Extract aggregate features from the array of dictionaries in each row
        def extract_features(row):
            if isinstance(row,(list, np.ndarray) ):
                df = pd.DataFrame(row.tolist())
                return [
                    len(df),  # Total impressions
                    df["cpm"].mean() if "cpm" in df and not df["cpm"].isna().all() else 0,  # Average CPM
                    df["duration"].mean() if "duration" in df and not df["duration"].isna().all() else 0,  # Average Duration
                    df["is_clicked"].sum() if "is_clicked" in df else 0,  # Total Clicked
                    df["is_skipped"].sum() if "is_skipped" in df else 0,  # Total Skipped
                    (df["placement"] == "fs").sum() if "placement" in df else 0,  # Count of Fullscreen Ads
                    (df["placement"] == "rv").sum() if "placement" in df else 0,  # Count of Rewarded Ads
                ]
            else:
                # Return default values if row is not a list
                return [0, 0, 0, 0, 0, 0, 0]

        # Apply feature extraction
        transformed = data.apply(extract_features)

        # Convert to DataFrame
        transformed_df = pd.DataFrame(
            transformed.tolist(),
            #columns=transformed.columns,
            index=data.index
        )

        return transformed_df

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def get_feature_names_out(self,data, columns):
        feature_names = []
        for column in columns:
            feature_names.extend([
                f"{column}.total_count",
                f"{column}.cpm_mean",
                f"{column}.duration_mean",
                f"{column}.clicked_count",
                f"{column}.skipped_count",
                f"{column}.placement_fs_count",
                f"{column}.placement_rv_count",
            ])
        return feature_names

class JSONToFeaturesStep:
    def __init__(self, top_n=None):
        """
        Parameters:
        - top_n: Number of most frequent keys to retain. If None, keep all keys.
        """
        self.top_n = top_n
        self.top_keys = None

    def fit(self, data):
        """Identify the top-n keys if top_n is set."""
        all_keys = {}
        for value in data:
            try:
                parsed = json.loads(value.replace("'", '"'))
                for key in parsed.keys():
                    all_keys[key] = all_keys.get(key, 0) + 1
            except (json.JSONDecodeError, AttributeError):
                continue
        if self.top_n:
            self.top_keys = sorted(all_keys, key=all_keys.get, reverse=True)[:self.top_n]
        else:
            self.top_keys = list(all_keys.keys())

    def transform(self, data, column_name="source_app"):
        """Expand the JSON strings into separate columns."""
        expanded_data = []
        for value in data:
            try:
                parsed = json.loads(value.replace("'", '"'))
                row = {f"{column_name}.{key}": parsed.get(key, 0) for key in self.top_keys}
            except (json.JSONDecodeError, AttributeError):
                row = {f"{column_name}.{key}": 0 for key in self.top_keys}
            expanded_data.append(row)
        return pd.DataFrame(expanded_data)

    def fit_transform(self, data, column_name="source_app"):
        self.fit(data)
        return self.transform(data, column_name=column_name)

    def get_feature_names_out(self, data, columns):
        """Generate feature names using the column name and keys."""
        feature_names = []
        for column in columns:
            feature_names.extend([f"{column}.{key}" for key in self.top_keys])
        return feature_names


class ExpandDictStep:
    
        
    def fit(self, data):
        pass

    def transform(self, data):
        return expand_dict_column(data)

    def fit_transform(self, data):
        return self.transform(data)

    def __call__(self, data):
        # This makes the object callable like a function
        return self.transform(data)

    def get_feature_names_out(self,data, columns):

        feature_names = []
        for column in columns:
            # Extract the dictionaries in the column
            dict_column = data[column].dropna()  # Ignore NaN values for key extraction
            
            # Get unique keys from all dictionaries in the column
            all_keys = set()
            for value in dict_column:
                if isinstance(value, dict):  # Ensure the value is a dictionary
                    all_keys.update(value.keys())  # Add the keys of the dictionary to the set
            
            # Generate feature names based on the keys
            feature_names.extend([f"{column}.{key}" for key in all_keys])
        
        return feature_names



class TransformationPipeline:
    def __init__(self):
        self.steps = []
        self.transformations_metadata = None  # Store metadata here

    def add_step(self, name, step, columns):
        """Add a step to the pipeline."""
        self.steps.append({"name": name, "step": step, "columns": columns})

               

    def fit(self, data):
        for step_info in self.steps:
            step = step_info["step"]
            for column in step_info["columns"]:
                if hasattr(step, "fit"):
                    step.fit(data[column])

    def transform(self, data):

        for step_info in self.steps:
            step = step_info["step"]
            print(step)
            for column in step_info["columns"]:
                if hasattr(step, "transform"):
                   
                    # Handle expanded outputs
                    if hasattr(step, "get_feature_names_out"):
                        
                        transformed = step.transform(data[column])
                        # Use get_feature_names_out for feature name generation
                        transformed.columns = step.get_feature_names_out(data,[column])
                        data = data.drop(columns=column).join(transformed)
                    elif hasattr(step, "categories_"):
                        # Handle OneHotEncoder-like transformations
                        transformed = step.transform(data[column])
                        transformed_df = pd.DataFrame(
                            
                            transformed,
                            columns=[
                                f"{column}_{category}"
                                for category in step.categories_[0]
                            ],
                            index=data.index,
                        )
                        data = data.drop(columns=column).join(transformed_df)

                    elif hasattr(step, "col_name"):
                         transformed = step.transform(data[column], column) 
                         #transformed.columns = step.get_feature_names_out(data,[column])
                         data = data.drop(columns=column).join(transformed)
                    
                    else:
                        # Direct replacement or multi-column outputs
                        if transformed.shape[1] == 1:
                            data[column] = transformed.ravel()
                        else:
                            transformed.columns = [f"{column}_feature_{i}" for i in range(transformed.shape[1])]
                            
                            data = data.drop(columns=column).join(transformed)
                            
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)





    def save_pipeline(self, pipeline_filepath, metadata_filepath):
        """
        Save the fitted pipeline and metadata.
        """
        if self.transformations_metadata is None:
            raise ValueError("Transformations metadata must be set before saving the pipeline.")
        
        # Save the metadata as JSON
        with open(metadata_filepath, "w") as f:
            json.dump(self.transformations_metadata, f, indent=4)
    
        # Save the entire pipeline using pickle
        with open(pipeline_filepath, "wb") as f:
            pickle.dump(self, f)

   


    @staticmethod
    def load_pipeline(pipeline_filepath, metadata_filepath):
        """
        Load the fitted pipeline and metadata.
        """
        # Load the metadata
        with open(metadata_filepath, "r") as f:
            transformations_metadata = json.load(f)
        
        # Load the entire pipeline from pickle
        with open(pipeline_filepath, "rb") as f:
            pipeline = pickle.load(f)
    
        # Update the metadata in the loaded pipeline
        pipeline.transformations_metadata = transformations_metadata
    
        return pipeline













    
