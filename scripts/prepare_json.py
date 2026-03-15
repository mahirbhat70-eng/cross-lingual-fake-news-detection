import pandas as pd
import json
import os

def convert_csv_to_json(csv_path, json_path, label, language_code):
    """
    Reads a CSV file, formats its content into a specific JSON structure,
    and saves it as a JSON file.
    """
    print(f"Reading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # Ensure the text column matches your CSV files ('title' or 'text')
    text_column = 'text' 
    if text_column not in df.columns:
        # Fallback for older CSV formats
        if 'title' in df.columns:
            text_column = 'title'
        else:
            print(f"Error: Neither 'text' nor 'title' column found. Available columns: {list(df.columns)}")
            return

    formatted_data = []
    for index, row in df.iterrows():
        # Ensure text is not empty or just whitespace
        if isinstance(row[text_column], str) and row[text_column].strip():
            news_item = {
                "text": str(row[text_column]).strip(),
                "label": label,
                "language": language_code
            }
            formatted_data.append(news_item)

    print(f"Writing {len(formatted_data)} formatted articles to {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=4)

    print(f"Successfully converted!")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, 'data', 'dataset')

    # --- Process the FAKE news CSV ---
    fake_csv_file = os.path.join(dataset_dir, 'politifact_fake.csv')
    fake_json_file = os.path.join(dataset_dir, 'politifact_fake.json')
    convert_csv_to_json(
        csv_path=fake_csv_file,
        json_path=fake_json_file,
        label=1,  # 1 for Fake
        language_code='en'
    )

    # --- Process the REAL news CSV ---
    real_csv_file = os.path.join(dataset_dir, 'politifact_real.csv')
    real_json_file = os.path.join(dataset_dir, 'politifact_real.json')
    convert_csv_to_json(
        csv_path=real_csv_file,
        json_path=real_json_file,
        label=0,  # 0 for Real
        language_code='en'
    )