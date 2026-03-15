# scripts/check_csv_labels.py
import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'data', 'dataset', 'hindi_news_dataset.csv')

try:
    df = pd.read_csv(csv_path)
    print("Columns found:", list(df.columns))

    if 'Category' in df.columns:
        print("\nUnique values found in the 'Category' column:")
        print(df['Category'].unique())
    else:
        print("\nCould not find the 'Category' column.")

except Exception as e:
    print(f"An error occurred: {e}")