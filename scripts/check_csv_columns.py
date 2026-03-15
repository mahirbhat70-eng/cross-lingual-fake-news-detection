# scripts/check_csv_columns.py
import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'data', 'dataset', 'hindi_news_dataset.csv')

try:
    df = pd.read_csv(csv_path)
    print("Columns found in your CSV file:")
    print(list(df.columns))
except FileNotFoundError:
    print(f"Error: The file was not found at {csv_path}")
except Exception as e:
    print(f"An error occurred: {e}")