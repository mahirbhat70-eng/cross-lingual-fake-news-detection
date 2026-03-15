import os
import pandas as pd
import json

def merge_specific_files(dataset_directory, file_list, output_filename):
    """
    Loads a specific list of JSON files, merges them, removes duplicates,
    shuffles the result, and saves it to a new JSON file.
    """
    all_records = []
    
    print("🚀 Starting the merge process...")
    
    # Loop through the specific list of files to merge
    for filename in file_list:
        filepath = os.path.join(dataset_directory, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: File not found, skipping: {filename}")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_records.extend(data)
                print(f"✅ Loaded {len(data)} records from {filename}")
        except Exception as e:
            print(f"❌ Error reading {filename}: {e}")

    if not all_records:
        print("No data was loaded. Aborting merge.")
        return

    # Use pandas to easily handle deduplication and shuffling
    df = pd.DataFrame(all_records)
    
    # Remove duplicates based on the 'text' column, keeping the first instance
    df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    
    # Shuffle the dataset for randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the final, merged dataframe
    output_path = os.path.join(dataset_directory, output_filename)
    print(f"💾 Merged {len(df)} unique records. Saving to {output_filename}...")
    df.to_json(output_path, orient='records', indent=4)
    
    print("🎉 Merging complete!")


if __name__ == "__main__":
    # Define the location of your datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "data", "dataset")

    # --- This list explicitly defines which files to merge ---
    files_to_merge = [
        "liar_binary.json",
        "politifact_real.json",
        "politifact_fake.json",
        "indian_news.json"
    ]

    # Define the name of the final output file
    final_output_file = "combined_dataset.json"

    merge_specific_files(dataset_dir, files_to_merge, final_output_file)