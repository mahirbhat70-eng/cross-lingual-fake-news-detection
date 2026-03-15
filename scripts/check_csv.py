import pandas as pd
import os

# The name of the column that contains the labels
LABEL_COLUMN_NAME = 'Category' 

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'data', 'dataset', 'hindi_news_dataset.csv')

try:
    df = pd.read_csv(csv_path)
    if LABEL_COLUMN_NAME in df.columns:
        print(f"Unique values found in the '{LABEL_COLUMN_NAME}' column:")
        print(df[LABEL_COLUMN_NAME].unique())
    else:
        print(f"Warning: Could not find the column '{LABEL_COLUMN_NAME}'.")
except Exception as e:
    print(f"An error occurred: {e}")