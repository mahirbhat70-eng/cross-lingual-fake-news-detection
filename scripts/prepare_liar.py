import os
import pandas as pd
from datasets import load_dataset, concatenate_datasets

def standardize_liar_dataset_from_local_files(output_dir):
    print("🚀 Loading local LIAR dataset from TSV files...")
    base_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'liar_dataset')
    
    data_files = {
        "train": os.path.join(base_data_dir, 'train.tsv'),
        "test": os.path.join(base_data_dir, 'test.tsv'),
        "validation": os.path.join(base_data_dir, 'valid.tsv')
    }

    for split, path in data_files.items():
        if not os.path.exists(path):
            print(f"❌ Error: LIAR dataset file not found at {path}.")
            return

    column_names = ['id', 'label', 'statement', 'subject', 'speaker', 'job_title', 'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context']
    dataset = load_dataset('csv', data_files=data_files, sep='\t', names=column_names)
    combined_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    print(f"✅ Combined all LIAR splits with {len(combined_dataset)} entries.")

    def map_labels_and_rename(batch):
        # This line implements your rule: labels 3, 4, 5 become 0 (Real)
        # and all others (0, 1, 2) become 1 (Fake).
        new_labels = [0 if label in [3, 4, 5] else 1 for label in batch['label']]
        
        return {
            "text": batch['statement'],
            "label": new_labels,
            "language": ["en"] * len(batch['statement'])
        }

    print("⚙️ Standardizing LIAR dataset to binary labels...")
    processed_dataset = combined_dataset.map(map_labels_and_rename, batched=True, num_proc=4, remove_columns=combined_dataset.column_names)

    output_path = os.path.join(output_dir, 'liar.json')
    print(f"💾 Saving standardized data to {output_path}...")
    df = processed_dataset.to_pandas()
    df.to_json(output_path, orient="records", indent=4)
    print(f"🎉 Successfully created {output_path}!")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_dir = os.path.join(base_dir, 'data', 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    standardize_liar_dataset_from_local_files(dataset_dir)