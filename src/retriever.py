import os
import argparse
from omegaconf import OmegaConf
from collections import defaultdict
from datetime import datetime
import json

import numpy as np
import csv
import gc

import faiss
import pickle
import torch # torch.cuda.empty_cache()를 위해 추가

def get_mbeir_task_name(task_id):
    task_names = {
        0: "Image-to-Text Retrieval",
        # 필요하다면 다른 task_id에 대한 이름을 추가하세요.
    }
    return task_names.get(task_id, f"Unknown Task {task_id}")

# QREL 파일 형식: query_id Q0 doc_id relevance_score task_id
def load_qrel(qrel_path):
    qrel = defaultdict(set)
    qid_to_taskid = {}
    print(f"Loading QREL from: {qrel_path}")
    if not os.path.exists(qrel_path):
        raise FileNotFoundError(f"QREL file not found at: {qrel_path}")
    with open(qrel_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                doc_id = parts[2]
                relevance_score = int(parts[3])
                task_id = parts[4] if len(parts) > 4 else "0"

                if relevance_score > 0:
                    qrel[qid].add(doc_id)
                qid_to_taskid[qid] = task_id
            else:
                print(f"Skipping malformed QREL line: {line.strip()}")
    print(f"Loaded {len(qrel)} queries from QREL.")
    return qrel, qid_to_taskid

def create_index(config):
    # 각 모달리티 조합에 대해 인덱스를 생성하고 저장합니다.
    modality_suffixes = {"i": "img_only", "t": "txt_only", "it": "img_txt"}
    
    # 생성된 인덱스들의 경로를 저장할 딕셔너리
    indexed_paths = {}

    for suffix_key, modality_type in modality_suffixes.items():
        print(f"\n--- Creating FAISS index for candidate modality: {modality_type} ---")
        embed_data_file = f"cand_{config.split}_{suffix_key}_embed.npy"
        embed_data_path = os.path.join(config.mulimr_dir, config.embed_dir_name, config.split, config.model_name, embed_data_file)
        
        embed_data_id_file = f"cand_{config.split}_ids.npy"
        embed_data_id_path = os.path.join(config.mulimr_dir, config.embed_dir_name, config.split, config.model_name, embed_data_id_file)
        
        if not os.path.exists(embed_data_path) or not os.path.exists(embed_data_id_path):
            print(f"Warning: Missing embedding files for {modality_type}. Skipping index creation.")
            print(f"Expected: {embed_data_path} and {embed_data_id_path}")
            continue

        embedding_list = np.load(embed_data_path).astype("float32")
        hashed_id_list = np.load(embed_data_id_path)
        print(f"Embedder Log: Load the cand embed npy file: {embed_data_path}")
        print(f"Embedder Log: Load the cand embed idxnpy file: {embed_data_id_path}")

        # Check unique ids
        assert len(hashed_id_list) == len(set(hashed_id_list)), f"IDs should be unique for {modality_type}"
        
        faiss.normalize_L2(embedding_list)
        d = embedding_list.shape[1]
        
        index_config = config.index_config
        faiss_config = index_config.faiss_config
        
        if faiss_config.dim != d:
             print(f"Warning: Index dimension mismatch for {modality_type}. Config: {faiss_config.dim}, Data: {d}. Adjusting config.")
             faiss_config.dim = d # config의 dim을 실제 임베딩 차원으로 업데이트
        
        metric = getattr(faiss, faiss_config.metric)
        cpu_index = faiss.index_factory(
            faiss_config.dim,
            f"IDMap,{faiss_config.idx_type}",
            metric,
        )
        print("Creating FAISS index with the following parameters:")
        print(f"Index type: {faiss_config.idx_type}")
        print(f"Metric: {faiss_config.metric}")
        print(f"Dimension: {faiss_config.dim}")
        
        # Distribute the index across multiple GPUs
        ngpus = faiss.get_num_gpus()
        if ngpus > 0:
            print(f"Number of GPUs used for indexing: {ngpus}")
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            index_gpu = faiss.index_cpu_to_all_gpus(cpu_index, co=co, ngpu=ngpus)
            print("Add data to the GPU index")
            index_gpu.add_with_ids(embedding_list, hashed_id_list)
            index_cpu_final = faiss.index_gpu_to_cpu(index_gpu)
            del index_gpu 
            
        else: # Moved 'else' block to handle CPU-only case
            print("No GPUs detected or configured. Using CPU for indexing.")
            print("Add data to the CPU index")
            cpu_index.add_with_ids(embedding_list, hashed_id_list)
            index_cpu_final = cpu_index

        index_path = os.path.join(
            config.mulimr_dir,
            config.embed_dir_name,
            config.split,
            config.model_name,
            f"cand_{suffix_key}_index.faiss",
        )
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index_cpu_final, index_path)
        
        print(f"Successfully indexed {index_cpu_final.ntotal} documents for {modality_type}")
        print(f"Index saved to: {index_path}")
        indexed_paths[suffix_key] = index_path 

        del embedding_list
        del hashed_id_list 
        del cpu_index
        del index_cpu_final
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache emptied after index creation.")
    
    return indexed_paths 

def compute_mrr(relevant_docs, retrieved_indices):
    """
    Mean Reciprocal Rank (MRR) for a single query.
    Args:
        relevant_docs (set): Set of relevant document IDs for the current query.
        retrieved_indices (list): Ordered list of retrieved document IDs.
    Returns:
        float: Reciprocal Rank for the query (1/rank) or 0.0 if no relevant doc found.
    """
    if not relevant_docs: # 관련 문서가 없으면 계산할 수 없음
        return 0.0

    for rank, doc_id in enumerate(retrieved_indices):
        if doc_id in relevant_docs:
            return 1.0 / (rank + 1) # rank는 0부터 시작하므로 +1
    return 0.0 # 관련 문서를 찾지 못한 경우 (fallback)

def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    """
    Computes binary Recall@k for a single query.
    Args:
        relevant_docs (set): Set of relevant document IDs for the current query.
        retrieved_indices (list): Ordered list of retrieved document IDs.
        k (int): The 'k' for Recall@k.
    Returns:
        float: 1.0 if any relevant doc is found in top-k, else 0.0.
    """
    if not relevant_docs:
        return 0.0

    # retrieved_indices는 이미 k개로 제한되어 있으므로, 단순히 처음부터 k개를 사용
    top_k_retrieved_indices_set = set(retrieved_indices[:k]) 
    relevant_docs_set = set(relevant_docs)

    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0
    else:
        return 0.0

# 3. FAISS 인덱스 검색 함수
def search_index(query_embed_path, cand_index_path, batch_size, num_cand_to_retrieve):
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache emptied before loading query embeddings and index.")
    
    print(f"Loading query embeddings from: {query_embed_path}")
    query_embeddings = np.load(query_embed_path).astype('float32')
    faiss.normalize_L2(query_embeddings) 

    print(f"Loading FAISS index from: {cand_index_path}")
    if not os.path.exists(cand_index_path):
        raise FileNotFoundError(f"FAISS index file not found at: {cand_index_path}")
    index_cpu = faiss.read_index(cand_index_path)
    print(f"Faiss: Number of documents in the index: {index_cpu.ntotal}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache emptied after loading index_cpu.")
        
    
    ngpus = faiss.get_num_gpus()
    index_to_use = None 
    if ngpus > 0:
        print(f"Faiss: Number of GPUs used for searching: {ngpus}")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        index_gpu = faiss.index_cpu_to_all_gpus(index_cpu, co=co, ngpu=ngpus)
        index_to_use = index_gpu
    else:
        print("Faiss: No GPUs detected or configured. Using CPU for search.")
        index_to_use = index_cpu 

    all_distances = []
    all_indices = []

    for i in range(0, len(query_embeddings), batch_size):
        batch = query_embeddings[i : i + batch_size]
        distances, indices = index_to_use.search(batch, num_cand_to_retrieve)
        all_distances.append(distances)
        all_indices.append(indices)
    
    # --- GPU 및 CPU 인덱스 메모리 해제 로직 시작 ---
    if ngpus > 0 and 'index_gpu' in locals(): # 'index_gpu'가 정의되었을 때만 del 수행
        del index_gpu
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache emptied after GPU search.")

    del index_cpu # CPU 인덱스도 사용 후 메모리 해제
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache emptied after CPU index clean-up.")
    # --- GPU 및 CPU 인덱스 메모리 해제 로직 끝 ---

    final_distances = np.vstack(all_distances)
    final_indices = np.vstack(all_indices)

    return final_distances, final_indices

def unhash_qid(hashed_id):
    return str(hashed_id)

def unhash_did(hashed_id):
    return str(hashed_id)

def run_retrieval(config, indexed_paths):
    mulimr_dir = config.mulimr_dir
    mulimr_data_dir = config.mulimr_data_dir 
    retrieval_config = config.retrieval_config
    qrel_dir_name = retrieval_config.qrel_dir_name
    embed_dir_name = config.embed_dir_name
    model_name = config.model_name
    
    results_dir_name = retrieval_config.results_dir_name
    exp_results_dir = os.path.join(mulimr_dir, results_dir_name, model_name)
    os.makedirs(exp_results_dir, exist_ok=True)
    exp_tsv_results_dir = os.path.join(exp_results_dir, "final_tsv")
    os.makedirs(exp_tsv_results_dir, exist_ok=True)
    
    splits = []
    dataset_types = ["test"] 
    for split_name in dataset_types:
        retrieval_dataset_config = getattr(retrieval_config, f"{split_name}_datasets_config", None)
        if retrieval_dataset_config and retrieval_dataset_config.enable_retrieve:
            dataset_name_list = getattr(retrieval_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(retrieval_dataset_config, "correspond_cand_pools_name", None)
            qrel_name_list = getattr(retrieval_dataset_config, "correspond_qrels_name", None)
            metric_names_list = getattr(retrieval_dataset_config, "correspond_metrics_name", None)
                    
            splits.append(
                (
                    split_name,
                    dataset_name_list,
                    cand_pool_name_list,
                    qrel_name_list,
                    metric_names_list,
                )
            )
            assert (
                len(dataset_name_list) == len(cand_pool_name_list) == len(qrel_name_list) == len(metric_names_list)
            ), "Mismatch between datasets and candidate pools and qrels."

    print("-" * 30)
    for (
        split_name,
        dataset_name_list,
        cand_pool_name_list,
        qrel_name_list,
        metric_names_list,
    ) in splits:
        
        print(
            f"Split: {split_name}, Retrieval Datasets: {dataset_name_list}, Candidate Pools: {cand_pool_name_list}, Metric: {metric_names_list})"
        )
        print("-" * 30)
        
    eval_results = []
    qrel_dir = os.path.join(mulimr_data_dir, qrel_dir_name) 
    
    query_modalities = {"i": "img_only", "t": "txt_only", "it": "img_txt"}
    candidate_modalities = {"i": "img_only", "t": "txt_only", "it": "img_txt"}

    for (
        split,
        dataset_name_list,
        cand_pool_name_list,
        qrel_name_list,
        metric_names_list,
    ) in splits:
        for dataset_name_cfg, cand_pool_name_cfg, qrel_name_cfg, metric_names in zip(
            dataset_name_list, cand_pool_name_list, qrel_name_list, metric_names_list
        ):
            dataset_name_cfg = dataset_name_cfg.lower()
            cand_pool_name_cfg = cand_pool_name_cfg.lower()
            qrel_name_cfg = qrel_name_cfg.lower()

            qrel_path = os.path.join(qrel_dir, f"rocov2-questions-radiology_{split}_qrels.txt")
            qrel, qid_to_taskid = load_qrel(qrel_path)

            for q_suffix, q_mode_str in query_modalities.items():
                for c_suffix, c_mode_str in candidate_modalities.items():
                    print("\n" + "=" * 50)
                    print(f"Retrieving for Query Type: {q_mode_str} (Q_{q_suffix}) | Candidate Type: {c_mode_str} (C_{c_suffix})")
                    print(f"Dataset: {dataset_name_cfg} | Split: {split} | CandPool: {cand_pool_name_cfg}")
                    print("=" * 50)

                    embed_query_id_path = os.path.join(mulimr_dir, embed_dir_name, split, model_name, f"query_{split}_ids.npy")
                    if not os.path.exists(embed_query_id_path):
                        print(f"Skipping: Query ID file not found at {embed_query_id_path}")
                        continue
                    hashed_query_ids = np.load(embed_query_id_path)
                    print(f"Loaded query IDs from: {embed_query_id_path}")

                    embed_query_path = os.path.join(mulimr_dir, embed_dir_name, split, model_name, f"query_{split}_{q_suffix}_embed.npy")
                    if not os.path.exists(embed_query_path):
                        print(f"Skipping: Query embedding file not found at {embed_query_path}")
                        continue
                    print(f"Loaded query embeddings from: {embed_query_path}")

                    cand_index_path = indexed_paths.get(c_suffix) 
                    if not cand_index_path or not os.path.exists(cand_index_path):
                        print(f"Skipping: Candidate FAISS index not found for {c_mode_str} at expected path {cand_index_path}")
                        continue
                    print(f"Loading candidate FAISS index from: {cand_index_path}")

                    metric_list = [metric.strip() for metric in metric_names.split(",")]
                    
                    metric_recall_list = [metric for metric in metric_list if "recall" in metric.lower()]
                    max_k_for_recall = 0
                    if metric_recall_list:
                        max_k_for_recall = max([int(metric.split("@")[1]) for metric in metric_recall_list])
                    

                    k = max_k_for_recall 
                    k = max(k, 1000) 
                    if k == 0:
                        k = 100

                        
                    print(f"Searching with k={k} (max candidates to retrieve)")
                    retrieved_cand_dist, retrieved_indices = search_index(
                        embed_query_path,
                        cand_index_path,
                        batch_size=16,
                        num_cand_to_retrieve=k, 
                    )

                    recall_values_by_task = defaultdict(lambda: defaultdict(list))
                    mrr_values_by_task = defaultdict(list) 

                    for i, retrieved_indices_for_qid in enumerate(retrieved_indices):
                        retrieved_indices_for_qid_unhashed = [unhash_did(idx) for idx in retrieved_indices_for_qid]
                        qid = unhash_qid(hashed_query_ids[i])
                        relevant_docs = qrel.get(qid, set()) 
                        task_id = qid_to_taskid.get(qid, "0") 

                        for metric in metric_recall_list:
                            k_val = int(metric.split("@")[1])
                            recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid_unhashed, k_val)
                            recall_values_by_task[task_id][metric].append(recall_at_k)
                        
                        mrr_score = compute_mrr(relevant_docs, retrieved_indices_for_qid_unhashed)
                        mrr_values_by_task[task_id].append(mrr_score)

                    for task_id in sorted(recall_values_by_task.keys(), key=int):
                        task_name = get_mbeir_task_name(int(task_id))
                        result = {
                            "TaskID": int(task_id),
                            "Task": task_name,
                            "Dataset": dataset_name_cfg,
                            "Split": split,
                            "CandPool": cand_pool_name_cfg,
                            "QueryModality": q_mode_str,
                            "CandidateModality": c_mode_str,
                        }
                        
                        for metric in metric_recall_list:
                            if recall_values_by_task[task_id][metric]:
                                mean_recall_at_k = round(sum(recall_values_by_task[task_id][metric]) / len(recall_values_by_task[task_id][metric]), 4)
                                result[metric] = mean_recall_at_k
                                print(f"Retriever: Mean {metric} for Q_{q_suffix}-C_{c_suffix} (Task {task_id}): {mean_recall_at_k}")
                            else:
                                result[metric] = 0.0 
                        
                        mean_mrr = round(sum(mrr_values_by_task[task_id]) / len(mrr_values_by_task[task_id]), 4)
                        result["MRR"] = mean_mrr
                        print(f"Retriever: Mean MRR for Q_{q_suffix}-C_{c_suffix} (Task {task_id}): {mean_mrr}")
                        eval_results.append(result)
            
    print("\n" + "-" * 30)
    print("Saving overall evaluation results to TSV file...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_tsv_path = os.path.join(exp_tsv_results_dir, f"evaluation_results_{timestamp}.tsv") 
    if eval_results:

        all_fieldnames = set()
        for res in eval_results:
            all_fieldnames.update(res.keys())
        
        fieldnames = ["TaskID", "Task", "Dataset", "Split", "CandPool", "QueryModality", "CandidateModality"]
        
        def sort_recall_metrics(metric_name):
            if metric_name.startswith("Recall@"):
                try:
                    return int(metric_name.split("@")[1]) 
                except ValueError:
                    return float('inf') 
            return float('inf') 

        recall_metrics_to_add = sorted(
            [f for f in all_fieldnames if f.startswith("Recall@")],
            key=sort_recall_metrics
        )
        fieldnames.extend(recall_metrics_to_add) 

        fieldnames.append("MRR")
        remaining_fields = sorted(list(all_fieldnames - set(fieldnames)))
        fieldnames.extend(remaining_fields)

        with open(output_tsv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')

            writer.writeheader()
            writer.writerows(eval_results)

        print(f"Evaluation results successfully saved to: {output_tsv_path}")
    else:
        print("No evaluation results to save.")
    print("-" * 30)
        
def parse_arguments():
    parser = argparse.ArgumentParser(description="retrieval")
    parser.add_argument("--config_path", default="./config/retrieval.yaml",help="Path to the config file.")
    return parser.parse_args()

def main(config):
    indexed_paths = create_index(config)
    run_retrieval(config, indexed_paths)
    
if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)
    main(config)