import sys
import os
current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from data.dataset import Candidates_Dataset, CandidatePoolCollator, Query_Dataset, QueryCollator
from datasets import load_dataset
import transformers
import numpy as np 

def build_model_from_config(config):
    model_config = config.model
    model=None
    if model_config.merge== "score":
        from src.models.model import CLIPScoreFusion
        # from src.models.biomodel import CLIPBIOcoreFusion
        model = CLIPScoreFusion(
            model_name=model_config.pretrained_clip_model_dir,
            device = config.device,
            )
    elif model_config.merge == "weight":
        from src.model import CLIPWeightFusion
        model = CLIPWeightFusion(
            model_name=model_config.pretrained_clip_model_dir,
            device = config.device,
        )
    elif model_config.merge == "mlp":
        from src.model import CLIPMLPFusion
        model = CLIPMLPFusion(
            model_name=model_config.pretrained_clip_model_dir,
            device = config.device,
        ) 

    model.float()
    
    ckpt_config = model_config.ckpt_config
    checkpoint_path = os.path.join(config.mulimr_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
    assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
    print(f"loading CLIPScoreFusion checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False)["model"])

    return model

@torch.no_grad()
def generate_embeds_dataset(model, data_loader, device, use_fp16=True):
    embeddings_tensor = []
    id_list = []
    for batch in tqdm(data_loader, desc="Generating Embeddings"):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=True)
            elif isinstance(value, transformers.tokenization_utils_base.BatchEncoding):
                for k, v in value.items():
                    batch[key][k] = v.to(device)

        with autocast(enabled=use_fp16):
            embeddings_batched, ids_list_batched  = model(batch, encode_batch=True)
        embeddings_tensor.append(embeddings_batched.half())
        id_list.extend(ids_list_batched)
        
    embedding_tensor = torch.cat(embeddings_tensor, dim=0)
    embedding_list = embedding_tensor.half().cpu().numpy()
    
    return embedding_list, id_list
    
from omegaconf import OmegaConf
import argparse

def main(config):
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_config = config.model.ckpt_config
    model = build_model_from_config(config)
    model.eval()
    
    image_size = (model.clip_model.visual.input_resolution, model.clip_model.visual.input_resolution)
        
    # Ensure the model has an 'encode' method before generating embeddings
    if not callable(getattr(model, "get_img_preprocess_fn")):
        raise AttributeError("The provided model does not have an 'img_preprocess_fn' attribute.")
    if not callable(getattr(model, "get_tokenizer")):
        raise AttributeError("The provided model does not have a 'tokenizer' attribute.")
    
    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()
    model.to(config.device)
    
    dataset = load_dataset(config.dataset.name)
    
    # ---
    ## Data Loaders Setup
    # ---
    
    query_modes = {
        "img_only": "i",
        "txt_only": "t",
        "img_txt": "it"
    }

    query_datasets = {}
    candidate_datasets = {}
    
    for mode, suffix in query_modes.items():
        query_datasets[suffix] = Query_Dataset(
            dataset=dataset,
            dataset_dict_split=config.split,
            img_preprocess_fn=img_preprocess_fn,
            query_modes=mode
        )
        candidate_datasets[suffix] = Candidates_Dataset(
            dataset=dataset,
            dataset_dict_split=config.split,
            img_preprocess_fn=img_preprocess_fn,
            query_modes=mode
        )
            
    q_collator = QueryCollator(
        tokenizer = tokenizer,
        image_size = image_size[0]
    )
    
    c_collator = CandidatePoolCollator(
        tokenizer = tokenizer,
        image_size = image_size[0]
    )
    
    query_data_loaders = {suffix: DataLoader(ds, batch_size=16, shuffle=False, collate_fn=q_collator, num_workers=0) 
                          for suffix, ds in query_datasets.items()}
    candidate_data_loaders = {suffix: DataLoader(ds, batch_size=16, shuffle=False, collate_fn=c_collator, num_workers=0) 
                              for suffix, ds in candidate_datasets.items()}

    embeddings_output_dir = os.path.join(config.mulimr_dir, config.embed_dir_name, config.split, ckpt_config.ckpt_name)
    os.makedirs(embeddings_output_dir, exist_ok=True)
    
    q_id_list, c_id_list = [], []
    for suffix, q_data_loader in query_data_loaders.items():
        print(f"Generating query embeddings for {suffix} mode...")
        q_embedding_list, q_id_list_ = generate_embeds_dataset(model, q_data_loader, config.device, use_fp16=True)
        q_id_list = q_id_list_
        q_embed_data_name = f"query_{config.split}_{suffix}_embed.npy"
        q_embed_path = os.path.join(embeddings_output_dir, q_embed_data_name)
        np.save(q_embed_path, q_embedding_list)
        print(f"Query Embedder Log: Saved embeddings to {q_embed_path}.")
    
    for suffix, c_data_loader in candidate_data_loaders.items():
        print(f"Generating candidate embeddings for {suffix} mode...")
        c_embedding_list, c_id_list_ = generate_embeds_dataset(model, c_data_loader, config.device, use_fp16=True)
        c_id_list = c_id_list_
        c_embed_data_name = f"cand_{config.split}_{suffix}_embed.npy"
        c_embed_path = os.path.join(embeddings_output_dir, c_embed_data_name)
        np.save(c_embed_path, c_embedding_list)
        print(f"Cand Embedder Log: Saved embeddings to {c_embed_path}.")
    
    
    q_id_data_name = f"query_{config.split}_ids.npy"
    c_id_data_name = f"cand_{config.split}_ids.npy"
    q_id_path = os.path.join(embeddings_output_dir, q_id_data_name)
    c_id_path = os.path.join(embeddings_output_dir, c_id_data_name)
    np.save(q_id_path, q_id_list)
    np.save(c_id_path, c_id_list)
    
    print(f"Query Embedder Log: Saved IDs to {q_id_path}.")        
    print(f"Cand Embedder Log: Saved IDs to {c_id_path}.")
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Embeddings")
    parser.add_argument("--config_path", default="./config/embedding_config.yaml",help="Path to the config file.")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)
    main(config)