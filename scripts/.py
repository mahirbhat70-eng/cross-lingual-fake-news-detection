import os
from datasets import load_dataset, concatenate_datasets

def standardize_liar_dataset_optimized(output_dir):
    """
    Downloads the LIAR dataset, standardizes it using optimized batch processing,
    and saves it as a JSON file.
    """
    print("Downloading LIAR dataset from Hugging Face Hub...")
    dataset = load_dataset("liar")
    
    # Combine the train, validation, and test splits into one large dataset
    combined_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    print(f"Combined all splits into a single dataset with {len(combined_dataset)} entries.")

    def map_labels_and_rename(batch):
        """
        This function processes a batch of examples at once.
        """
        # The labels from the dataset are integers from 0 to 5
        # We map labels 3, 4, 5 (half-true, mostly-true, true) to 0 (Real)
        # and labels 0, 1, 2 (pants-fire, false, barely-true) to 1 (Fake)
        new_labels = [0 if label in [3, 4, 5] else 1 for label in batch['label']]
        
        # Create the new batch with the correct structure
        return {
            "text": batch['statement'], # Rename 'statement' column to 'text'
            "label": new_labels,
            "language": ["en"] * len(batch['statement']) # Add language for each entry
        }

    print("Standardizing the dataset using optimized batch processing...")
    # Use .map() for high-speed, parallel processing
    # `batched=True` sends data to the function in chunks for speed
    # `num_proc=4` uses 4 CPU cores to do the work in parallel
    processed_dataset = combined_dataset.map(
        map_labels_and_rename, 
        batched=True, 
        num_proc=4, 
        remove_columns=combined_dataset.column_names # Remove old, unneeded columns
    )

    output_path = os.path.join(output_dir, 'liar.json')
    print(f"Saving standardized data to {output_path}...")
    
    # Use the library's built-in JSON exporter for efficiency
    processed_dataset.to_json(output_path, orient="records", lines=False, indent=4)
        
    print(f"Successfully created {output_path}.")

if __name__ == '__main__':
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'dataset')
    standardize_liar_dataset_optimized(dataset_dir)