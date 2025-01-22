import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#from wordcloud import WordCloud
import json
import re

# Function to identify categorical columns
def get_categorical_columns(df):
    """Identify all categorical columns in the DataFrame."""
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def plot_categorical_distribution(df, column, top_n=10):
    """Plot the distribution of a categorical column."""
    top_classes = df[column].value_counts().head(top_n)
    return top_classes  # Return the counts for plotting

def analyze_categorical_columns(df, top_n=10, rare_threshold=5):
    """Perform EDA on all categorical columns in the DataFrame and plot in a grid."""
    categorical_columns = get_categorical_columns(df)
    num_columns = len(categorical_columns)
    if num_columns == 0:
        print("No categorical columns found.")
        return
    
    # Determine grid size
    cols = 3  # Number of columns in the grid
    rows = (num_columns + cols - 1) // cols  # Calculate rows required
    
    # Set up the figure for subplots
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    axes = axes.flatten()  # Flatten axes for easy iteration
    
    for i, col in enumerate(categorical_columns):
        ax = axes[i]
        # Get top categories and their counts
        top_classes = plot_categorical_distribution(df, col, top_n)
        
        # Plot bar chart
        sns.barplot(x=top_classes.values, y=top_classes.index, ax=ax, palette="viridis")
        ax.set_title(f"Top {top_n} Classes for '{col}'")
        ax.set_xlabel("Count")
        ax.set_ylabel("Class")
    
    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Remove empty subplots
    
    plt.tight_layout()
    plt.show()
    

def remove_big_outliers(df, columns, threshold_factor=5):
    """Remove very big outliers based on a threshold factor of IQR for specified columns."""
    filtered_df = df.copy()
    
    for col in columns:
        # Calculate the IQR
        Q1 = filtered_df[col].quantile(0.25)
        Q3 = filtered_df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds with a larger multiplier for extreme values
        lower_bound = Q1 - threshold_factor * IQR
        upper_bound = Q3 + threshold_factor * IQR
        
        # Remove only very big outliers (values beyond 5 or 10 times the IQR)
        filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]
    
    return filtered_df

def visualize_numerical_columns(df, numerical_columns=None, bins=20, plots_per_page=12, remove_big_outliers_flag=True):
    """
    Visualize numerical columns using histograms and boxplots in a grid, with independent scales for each.
    
    Parameters:
    - df: DataFrame containing the data.
    - numerical_columns: List of columns to visualize (optional). If None, it will use all numerical columns.
    - bins: Number of bins to use in histograms (default 20).
    - plots_per_page: Number of plots per page/grid (default 12).
    - remove_big_outliers_flag: Whether to remove very big outliers before plotting (default True).
    """
    if numerical_columns is None:
        # Select numerical columns if not provided
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    num_columns = len(numerical_columns)
    if num_columns == 0:
        print("No numerical columns found.")
        return
    
    # Remove very big outliers if requested
    if remove_big_outliers_flag:
        df = remove_big_outliers(df, numerical_columns)
    
    # Split into smaller groups for large number of columns
    num_pages = (num_columns + plots_per_page - 1) // plots_per_page  # Calculate number of pages
    print(f"Visualizing {num_columns} numerical columns in {num_pages} pages.")
    
    # Loop through each page of plots
    for page in range(num_pages):
        start_idx = page * plots_per_page
        end_idx = min((page + 1) * plots_per_page, num_columns)
        columns_to_plot = numerical_columns[start_idx:end_idx]
        
        # Set up the figure for subplots for this page
        cols = 3  # Number of columns in the grid
        rows = (len(columns_to_plot) + cols - 1) // cols  # Calculate rows required
        
        # Set up the figure with adjusted size for smaller grids
        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 6))
        axes = axes.flatten()  # Flatten axes for easy iteration
        
        for i, col in enumerate(columns_to_plot):
            ax = axes[i]
            
            # Plot Histogram
            sns.histplot(df[col], bins=bins, kde=True, ax=ax, color='skyblue', stat='density', linewidth=0)
            ax.set_title(f'Distribution of {col}', fontsize=10)
            ax.set_xlabel(f'{col}', fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Plot Boxplot in the same subplot but with independent scale
            ax_box = ax.twinx()  # Create a secondary y-axis for the boxplot
            sns.boxplot(x=df[col], ax=ax_box, color='orange', width=0.2, fliersize=3)
            ax_box.set_ylabel('Boxplot', fontsize=8)
            ax_box.tick_params(axis='both', which='major', labelsize=8)

        # Turn off unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
    
    

   

    
def separate_columns_by_type(df):
    # Lists to hold different categories of columns
    timestamp_columns = []
    json_columns = []
    categorical_columns = []
    numerical_columns = []
    id_columns = []
    
    for col in df.columns:
        # Step 1: Skip numerical columns from datetime conversion
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_columns.append(col)
            continue  # Skip this column from datetime checks
        
        # Step 2: Try to convert the column to datetime
        try:
            converted_column = pd.to_datetime(df[col], errors='raise', format=None)
            timestamp_columns.append(col)
            continue  # Skip further checks for datetime columns
        except (ValueError, TypeError):
            # If it cannot be converted, move to next step
            pass
        
        # Step 3: Check if the column contains JSON-like strings (assuming as str/dict)
        try:
            if (isinstance(df[col][0], np.ndarray)):
                json_columns.append(col)            
            else:
                is_json = df[col].dropna().astype(str).apply(
                lambda x: (isinstance(json.loads(x.replace("'", '"')),  (dict))) if x else False).all()
                if is_json:
                    json_columns.append(col)
                else:
                    categorical_columns.append(col)
        except (ValueError, TypeError):
            # ValueError occurs if `json.loads` fails (i.e., not JSON-like), treat as categorical
            categorical_columns.append(col)
    
        
        
    
    return categorical_columns, json_columns, numerical_columns, timestamp_columns