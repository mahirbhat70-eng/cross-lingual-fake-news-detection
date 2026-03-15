import pandas as pd
import json
import os

def convert_csv_to_json(csv_path, json_path):
    """
    Reads the Indian news CSV, converts text labels ('REAL'/'FAKE') to
    binary (0/1), formats it, and saves it as a JSON file.
    """
    print(f"Reading data from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: The file {csv_path} was not found.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return

    required_columns = ['text', 'label']
    if not all(col in df.columns for col in required_columns):
        print(f"❌ Error: CSV must contain 'text' and 'label' columns. Found: {list(df.columns)}")
        return

    # --- NEW: Map text labels to integers ---
    print("Found text labels. Converting 'REAL' to 0 and 'FAKE' to 1...")
    label_map = {'REAL': 0, 'FAKE': 1}
    df['label'] = df['label'].map(label_map)

    # Keep only the necessary columns and drop any rows with missing values
    df = df[required_columns].dropna()

    # Ensure labels are integers
    df['label'] = df['label'].astype(int)

    print(f"Writing {len(df)} formatted articles to {json_path}...")
    df.to_json(json_path, orient='records', indent=4)

    print(f"✅ Successfully converted {os.path.basename(csv_path)}!")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'data', 'dataset')

    input_csv_file = os.path.join(dataset_dir, 'news_dataset.csv')
    output_json_file = os.path.join(dataset_dir, 'indian_news.json')

    convert_csv_to_json(input_csv_file, output_json_file)