from datasets import load_dataset, DatasetDict 
from tqdm import tqdm # Import tqdm

df = load_dataset("Jiiwonn/rocov2-questions-radiology")

# List of splits to process
split_list = ["train", "validation", "test"]
new_splits = {}
current_global_id = 0 # Initialize global ID counter

# Process each split
for split in split_list:
    print(f"Processing {split} split and adding unique IDs...")
    split_dataset = df[split]
    
    # --- Code to add unique IDs ---
    ids_for_this_split = []
    for _ in tqdm(range(len(split_dataset)), desc=f"Adding IDs to {split}"):
        ids_for_this_split.append(current_global_id)
        current_global_id += 1
    
    # Add the 'id' column to the dataset
    split_dataset = split_dataset.add_column("id", ids_for_this_split)
    # --- End of ID adding code ---
    
    # Store the modified dataset split
    new_splits[split] = split_dataset

df_with_questions = DatasetDict(new_splits)
df_with_questions.push_to_hub("Jiiwonn/roco2-question-id-dataset")