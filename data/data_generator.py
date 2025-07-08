# make qrel.txt(gt)

import os
import json 
from datasets import load_dataset 
from tqdm import tqdm 


def get_mbeir_task_id(query_modality, pos_cand_modality):
    return 0

df = load_dataset("Jiiwonn/roco2-question-id-dataset")

base_output_dir = "/workspace/MultiMR/embedding"
qrels_dir = os.path.join(base_output_dir, "qrels")
os.makedirs(qrels_dir, exist_ok=True)


split_list = ["train", "validation", "test"]

for split_name in split_list:

    current_split_data = df[split_name]


    dataset_name_for_qrels = "rocov2-questions-radiology"
    qrels_file_name = f"{dataset_name_for_qrels}_{split_name}_qrels.txt"
    qrels_file_path = os.path.join(qrels_dir, qrels_file_name)

    print(f"\nGenerating qrels file {qrels_file_path} for split: {split_name}...")

    with open(qrels_file_path, "w", encoding="utf-8") as outfile:
        for entry in tqdm(current_split_data, desc=f"Processing {split_name} entries"):
            qid = entry["id"]
            cand_id = qid
            task_id = 0 

            outfile.write(f"{qid} 0 {cand_id} 1 {task_id}\n")

    print(f"Generated qrels file: {qrels_file_path}")
