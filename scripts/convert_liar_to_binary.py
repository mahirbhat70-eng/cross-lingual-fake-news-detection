import os
import json
import pandas as pd

def convert_liar_multiclass_to_binary(input_path, output_path):
    """
    Converts a multiclass liar dataset (text labels) into binary labels.

    Rules:
        'half-true', 'mostly-true', 'true' → 0  (True side)
        'pants-on-fire', 'false', 'barely-true' → 1  (False side)
    """
    print(f"Loading data from {input_path}...")

    # Check if the file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # Load the JSON data using pandas
    try:
        df = pd.read_json(input_path)
    except ValueError as e:
        print(f"Error reading JSON file: {e}")
        return

    print(f"Found {len(df)} total records to process.")
    print("Applying binary label conversion...")

    # Define label mapping function
    def map_label(label):
        if label in ['half-true', 'mostly-true', 'true']:
            return 0  # true side
        elif label in ['pants-on-fire', 'false', 'barely-true']:
            return 1  # false side
        else:
            return label  # unexpected value — keep as is for debugging

    # Apply label conversion
    df['label'] = df['label'].apply(map_label)

    # Save the new binary JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient='records', indent=4, force_ascii=False)

    print(f"Saving binary data to {output_path}...")
    print(f"Process complete. Created liar_binary.json with {len(df)} records.")


if __name__ == "__main__":
    # Construct paths using os for robustness
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "dataset", "liar_multiclass.json")
    output_file = os.path.join(base_dir, "data", "dataset", "liar_binary.json")

    convert_liar_multiclass_to_binary(input_file, output_file)
