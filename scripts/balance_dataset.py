import pandas as pd
from sklearn.utils import resample
import os

def balance_dataset(input_path, output_path):
    print(f"⚖️ Loading dataset from {input_path} to balance...")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: The file {input_path} was not found.")
        return

    df = pd.read_json(input_path)
    
    # --- NEW: Filter to ensure only 0 and 1 labels are present ---
    df = df[df['label'].isin([0, 1])]
    df['label'] = df['label'].astype(int)

    print(f"Original distribution: \n{df.label.value_counts()}")

    # Separate classes
    df_class_0 = df[df.label == 0]
    df_class_1 = df[df.label == 1]
    
    # --- NEW: Dynamically identify majority and minority classes ---
    if len(df_class_0) > len(df_class_1):
        majority_df = df_class_0
        minority_df = df_class_1
    else:
        majority_df = df_class_1
        minority_df = df_class_0
        
    print(f"Downsampling the majority class (label: {majority_df['label'].iloc[0]}) to match the minority class (size: {len(minority_df)})...")

    # Undersample majority class
    df_majority_downsampled = resample(majority_df, 
                                     replace=False,
                                     n_samples=len(minority_df), # to match minority class size
                                     random_state=42)

    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, minority_df])
    
    # Shuffle the final balanced dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nNew balanced distribution: \n{df_balanced.label.value_counts()}")

    print(f"💾 Saving balanced dataset to {output_path}...")
    df_balanced.to_json(output_path, orient='records', indent=4)
    print("🎉 Balancing complete!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'data', 'dataset')
    
    input_file = os.path.join(dataset_dir, 'combined_dataset.json')
    output_file = os.path.join(dataset_dir, 'combined_balanced.json')

    balance_dataset(input_file, output_file)