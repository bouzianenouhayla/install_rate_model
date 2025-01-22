import pandas as pd
import json
import re


def flatten_json(nested_json, parent_key='', sep='.'):
    """Flatten a nested JSON object into a dictionary with dot-separated keys."""
    items = {}
    for k, v in nested_json.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_json(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items



def change_data_types(df, column_types):
    """Preprocess the data based on predefined column types."""
    
    timestamp_columns = column_types.get('timestamp_features', [])
    categorical_columns = column_types.get('cat_features', [])
    numerical_columns = column_types.get('numerical_features', [])
    
    
    
    for col in timestamp_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    for col in categorical_columns:
        df[col] = df[col].astype('category')

    for col in numerical_columns:
        df[col] = df[col].astype('float')

    return df


# Function to load and process the dataset
def load_and_process_data(csv_path):
    # Load the dataset
    df = pd.read_parquet(csv_path)
    
    # Separate features and target
    X = df.drop('install_label', axis=1) 
    y = df['install_label']  

    return X, y
# for col in json_columns:
#         if col in df.columns:
#             # Ensure the column contains valid JSON
#             try:
#                 # Apply JSON transformation if the column is string type
#                 df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else {} if x is None else x)

#                 # Check if all values in the column are dictionaries
#                 if df[col].apply(lambda x: isinstance(x, dict)).any():
#                     # Flatten the JSON data into separate columns
#                     flattened_data = df[col].apply(lambda x: flatten_json(x))
#                     flattened_df = pd.json_normalize(flattened_data)
                    

#                     # Drop the original JSON column and add the new flattened columns
#                     df = pd.concat([df.drop(columns=[col]), flattened_df], axis=1)

#             except (ValueError, TypeError):
#                 pass  # Ignore if it's not a valid JSON-like column







def clean_column_name(col_name):
    """Clean column names by replacing spaces, removing special characters, and making them readable."""
    # Replace spaces with underscores
    col_name = col_name.replace(" ", "_")
    
    # Shorten time periods (if applicable)
    col_name = col_name.replace("last 1 days", "1d").replace("last 7 days", "7d").replace("last 28 days", "28d")
    
    # Replace commas with underscores
    col_name = col_name.replace(",", "_")
    # Replace commas with underscores
    col_name = col_name.replace(".", "_")
    
    # Replace 'by' with an underscore and more concise names
    col_name = col_name.replace("by source_app", "app")
    
    # Remove any other special characters
    col_name = re.sub(r'[^\w\s]', '', col_name)
    
    # Convert to lowercase for consistency
    return col_name.lower()





def flatten_dict_column(df, column):
    """
    Flatten a column that contains a dictionary (or JSON-like structure) into separate columns.
    Each key in the dictionary becomes a separate feature, and the value becomes the feature value.
    """
    # First, make sure we don't have any NaN values in the column that would break the conversion.
    df[column] = df[column].fillna('{}')  # Fill NaN with empty dict string for processing
    
    # Convert the dictionary column to a dict format (if it's a JSON string)
    df[column] = df[column].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    # Expand the dictionary into multiple columns
    dict_expanded = df[column].apply(pd.Series)
    
    # Rename the columns so that they reflect the original column name (to avoid collision)
    dict_expanded.columns = [f"{column}_{col}" for col in dict_expanded.columns]
    
    # Concatenate the expanded columns back into the original dataframe
    df = pd.concat([df, dict_expanded], axis=1)
    
    # Optionally, drop the original dictionary column
    df = df.drop(columns=[column])
    
    return df




def compute_sum_and_avg(df, col_name):
    sum_column_name = f"{col_name}_sum"
    avg_column_name = f"{col_name}_avg"
    
    # Parse the JSON strings and calculate sums and averages
    df[sum_column_name] = df[col_name].apply(
        lambda x: sum(json.loads(x).values()) if pd.notna(x) else 0
    )
    df[avg_column_name] = df[col_name].apply(
        lambda x: sum(json.loads(x).values()) / len(json.loads(x)) if pd.notna(x) else 0
    )
    
    # Drop the original column
    df = df.drop(columns=[col_name])
    return df

def aggregate_by_type(df, dict_features):
    for col in dict_features:
         df = compute_sum_and_avg(df, col)

    return df



def expand_dict_column(data):
    # Function to parse a JSON string into a dictionary
    
    #dicts = data.apply(parse_json_string) 
    # Convert the dictionaries into separate columns
    expanded_df = data.apply(pd.Series).astype(float)  # Expanding dict into separate columns
    
    # Return the expanded DataFrame with separate columns
    return expanded_df



