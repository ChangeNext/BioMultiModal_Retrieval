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
import torch 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

def get_mbeir_task_name(task_id):
    task_names = {
        0: "Image-to-Text Retrieval",

    }
    return task_names.get(task_id, f"Unknown Task {task_id}")

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
    modality_suffixes = {"i": "img_only", "t": "txt_only", "it": "img_txt"}
    
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

        assert len(hashed_id_list) == len(set(hashed_id_list)), f"IDs should be unique for {modality_type}"
        
        faiss.normalize_L2(embedding_list)
        d = embedding_list.shape[1]
        
        index_config = config.index_config
        faiss_config = index_config.faiss_config
        
        if faiss_config.dim != d:
            print(f"Warning: Index dimension mismatch for {modality_type}. Config: {faiss_config.dim}, Data: {d}. Adjusting config.")
            faiss_config.dim = d 
        metric = getattr(faiss, faiss_config.metric)
        
        ngpus = faiss.get_num_gpus()
        
        if ngpus > 0:
            print(f"Number of GPUs used for indexing: {ngpus}")
            cpu_index = faiss.index_factory(
                faiss_config.dim,
                f"IDMap,{faiss_config.idx_type}",
                metric,
            )
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            index_gpu = faiss.index_cpu_to_all_gpus(cpu_index, co=co, ngpu=ngpus)
            print("Add data to the GPU index")
            index_gpu.add_with_ids(embedding_list, hashed_id_list)
            index_cpu_final = faiss.index_gpu_to_cpu(index_gpu)
            del index_gpu
        else:
            print("No GPUs detected or configured. Using CPU for indexing.")
            cpu_index = faiss.index_factory(
                faiss_config.dim,
                f"IDMap,{faiss_config.idx_type}",
                metric,
            )
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

def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    if not relevant_docs:
        return 0.0

    top_k_retrieved_indices_set = set(retrieved_indices[:k])
    relevant_docs_set = set(relevant_docs)

    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0
    else:
        return 0.0

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
    
    if ngpus > 0 and 'index_gpu' in locals():
        del index_gpu
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache emptied after GPU search.")


    final_distances = np.vstack(all_distances)
    final_indices = np.vstack(all_indices)

    return final_distances, final_indices

def unhash_qid(hashed_id):
    return str(hashed_id)

def unhash_did(hashed_id):
    return str(hashed_id)

def load_original_embeddings_with_ids(embed_path, id_path):
    """
    원본 임베딩 파일과 ID 파일을 로드하여 ID에 따라 정렬된 임베딩과 ID 리스트를 반환합니다.
    시각화에 사용됩니다.
    """
    if not os.path.exists(embed_path) or not os.path.exists(id_path):
        print(f"Error: Original embedding files not found at {embed_path} or {id_path}")
        return None, None

    embeddings = np.load(embed_path).astype('float32')
    ids = np.load(id_path)

    id_to_embedding = {str(id_val): emb for id_val, emb in zip(ids, embeddings)}
    print(f"Loaded {len(id_to_embedding)} original embeddings with IDs from {embed_path} and {id_path}")
    return id_to_embedding

def visualize_retrieval(query_embedding, retrieved_embeddings, retrieved_ids, retrieved_ranks, k_val, plot_title, output_dir, file_prefix="retrieval_visualize"):
    """
    주어진 쿼리 임베딩과 검색된 상위 K개의 임베딩을 2D로 차원 축소하여 시각화하고 저장합니다.
    """

    all_embeddings = np.vstack([query_embedding, retrieved_embeddings])\
    if all_embeddings.shape[0] >= 2: 
        if all_embeddings.shape[0] > 50: 
            print(f"Performing t-SNE on {all_embeddings.shape[0]} embeddings...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(all_embeddings.shape[0] - 1, 30), n_iter=1000)
        else:
            print(f"Performing PCA on {all_embeddings.shape[0]} embeddings...")
            reducer = PCA(n_components=2)
        
        reduced_embeddings = reducer.fit_transform(all_embeddings)
    else:
        print("Not enough data points for PCA/t-SNE. Skipping dimension reduction.")
        reduced_embeddings = np.zeros((all_embeddings.shape[0], 2))
        if all_embeddings.shape[1] <= 2:
            reduced_embeddings[:, :all_embeddings.shape[1]] = all_embeddings
        else:
            pass 

    df = pd.DataFrame(reduced_embeddings, columns=['Component 1', 'Component 2'])

    df['Type'] = ['Query'] + ['Retrieved'] * len(retrieved_embeddings)
    
    query_label = "Query"
    try:
        q_id_part = file_prefix.split('_')[0][1:]
        query_label = f"Q-{q_id_part}"
    except IndexError:
        pass

    display_ids = [query_label] + [f"ID-{_id} (Rank:{rank})" for _id, rank in zip(retrieved_ids, retrieved_ranks)]
    df['ID'] = display_ids


    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df,
        x='Component 1',
        y='Component 2',
        hue='Type',
        style='Type',
        s=100, 
        alpha=0.8,
        palette={'Query': 'red', 'Retrieved': 'blue'}
    )

    for i, row in df.iterrows():
        color = 'red' if row['Type'] == 'Query' else 'black'
        plt.text(row['Component 1'] + 0.01, row['Component 2'] + 0.01, row['ID'], 
                 fontsize=8, ha='left', va='bottom', color=color)

    plt.title(plot_title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)

    file_name = f"{file_prefix}.png"
    save_path = os.path.join(output_dir, file_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Visualization saved to: {save_path}")

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

    viz_output_dir = os.path.join(exp_results_dir, "visualization")
    os.makedirs(viz_output_dir, exist_ok=True)
    
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
                    query_embeddings_all = np.load(embed_query_path).astype('float32')
                    print(f"Loaded query embeddings from: {embed_query_path}")

                    cand_index_path = indexed_paths.get(c_suffix) 
                    if not cand_index_path or not os.path.exists(cand_index_path):
                        print(f"Skipping: Candidate FAISS index not found for {c_mode_str} at expected path {cand_index_path}")
                        continue
                    print(f"Loading candidate FAISS index from: {cand_index_path}")

                    metric_list = [metric.strip() for metric in metric_names.split(",")]
                    metric_recall_list = [metric for metric in metric_list if "recall" in metric.lower()]
                    
                    if not metric_recall_list:
                        print("No recall metrics defined for this configuration. Skipping evaluation.")
                        continue
                        
                    k = max([int(metric.split("@")[1]) for metric in metric_recall_list])
                    
                    print(f"Searching with k={k}")
                    retrieved_cand_dist, retrieved_indices = search_index(
                        embed_query_path,
                        cand_index_path,
                        batch_size=retrieval_config.batch_size,
                        num_cand_to_retrieve=k,
                    )

                    viz_query_idx = 0 
                    if len(hashed_query_ids) > viz_query_idx:
                        selected_query_id_hashed = hashed_query_ids[viz_query_idx]
                        selected_query_embedding = query_embeddings_all[viz_query_idx] 
                        selected_retrieved_indices_hashed = retrieved_indices[viz_query_idx] 
                        
                        viz_k = retrieval_config.viz_k if hasattr(retrieval_config, 'viz_k') else 100
                        if k < viz_k:
                            print(f"Warning: viz_k ({viz_k}) is greater than search k ({k}). Visualizing up to search k.")
                            viz_k = k
                        
                        cand_embed_data_file = f"cand_{split}_{c_suffix}_embed.npy"
                        cand_embed_data_path = os.path.join(mulimr_dir, embed_dir_name, split, model_name, cand_embed_data_file)
                        cand_embed_data_id_file = f"cand_{split}_ids.npy"
                        cand_embed_data_id_path = os.path.join(mulimr_dir, embed_dir_name, split, model_name, cand_embed_data_id_file)
                        
                        cand_id_to_embedding = load_original_embeddings_with_ids(cand_embed_data_path, cand_embed_data_id_path)
                        
                        retrieved_embeddings_for_viz = []
                        retrieved_ids_for_viz = []
                        retrieved_ranks_for_viz = [] 
                        
                        if cand_id_to_embedding is not None:
                            for rank, idx_hashed in enumerate(selected_retrieved_indices_hashed[:5]):
                                unhashed_id = unhash_did(idx_hashed) 
                                if unhashed_id in cand_id_to_embedding:
                                    retrieved_embeddings_for_viz.append(cand_id_to_embedding[unhashed_id])
                                    retrieved_ids_for_viz.append(unhashed_id)
                                    retrieved_ranks_for_viz.append(rank + 1) 
                                else:
                                    print(f"Warning: Original embedding not found for retrieved ID {unhashed_id}. Skipping.")
                            
                            if retrieved_embeddings_for_viz:
                                retrieved_embeddings_for_viz = np.array(retrieved_embeddings_for_viz).astype('float32')
                                
                                plot_title = (
                                    f"Retrieval Viz: Q_{q_suffix}-C_{c_suffix} "
                                    f"(Top 5 for Query ID: {unhash_qid(selected_query_id_hashed)})"
                                )
                                
                                visualize_retrieval(
                                    query_embedding=selected_query_embedding,
                                    retrieved_embeddings=retrieved_embeddings_for_viz,
                                    retrieved_ids=retrieved_ids_for_viz,
                                    retrieved_ranks=retrieved_ranks_for_viz, 
                                    k_val=5,
                                    plot_title=plot_title,
                                    output_dir=viz_output_dir,
                                    file_prefix=f"Q{unhash_qid(selected_query_id_hashed)}_C{c_suffix}_top{viz_k}_{datetime.now().strftime('%H%M%S')}"
                                )
                            else:
                                print("No retrieved embeddings to visualize after filtering.")
                        else:
                            print("Original candidate embeddings not available. Skipping visualization.")
                    else:
                        print(f"Skipping visualization: Query index {viz_query_idx} out of bounds for current query set.")
                    
                    recall_values_by_task = defaultdict(lambda: defaultdict(list))
                    for i, retrieved_indices_for_qid in enumerate(retrieved_indices):
                        retrieved_indices_for_qid_unhashed = [unhash_did(idx) for idx in retrieved_indices_for_qid]
                        qid = unhash_qid(hashed_query_ids[i])
                        relevant_docs = qrel.get(qid, set()) 
                        task_id = qid_to_taskid.get(qid, "0") 

                        for metric in metric_recall_list:
                            k_val = int(metric.split("@")[1])
                            recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid_unhashed, k_val)
                            recall_values_by_task[task_id][metric].append(recall_at_k)

                    for task_id, recalls in recall_values_by_task.items():
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
                            if recalls[metric]: 
                                mean_recall_at_k = round(sum(recalls[metric]) / len(recalls[metric]), 4)
                            else:
                                mean_recall_at_k = 0.0 
                            result[metric] = mean_recall_at_k
                            print(f"Retriever: Mean {metric} for Q_{q_suffix}-C_{c_suffix}: {mean_recall_at_k}")
                        eval_results.append(result)
            
    print("\n" + "-" * 30)
    print("Saving overall evaluation results to TSV file...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_tsv_path = os.path.join(exp_tsv_results_dir, f"evaluation_results_{timestamp}.tsv") 
    if eval_results:
        fieldnames = list(eval_results[0].keys())

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

    if not hasattr(config.retrieval_config, 'viz_k'):
        config.retrieval_config.viz_k = 100 
    if not hasattr(config.retrieval_config, 'batch_size'):
        config.retrieval_config.batch_size = 8 
    main(config)