import gradio as gr
import faiss
import numpy as np
import os
import torch
from omegaconf import OmegaConf
from PIL import Image
import transformers
from datasets import load_dataset
import torch
from src.models.model import CLIPScoreFusion
from data.dataset import QueryCollator
from src.embedder import build_model_from_config
import argparse

## text masking
def _get_padded_text_with_mask(txt):
        return (txt, 1) if txt not in [None, ""] else ("", 0)

## image masking    
def _get_padded_image_with_mask(img):
    return (img, 1) if img is not None else (padded_image, 0)

## Search funtion
def perform_search(query_embedding, top_k=10):
    if FAISS_INDEX_READY is None:
        return [], [], []

    if torch.cuda.is_available():
        torch.cuda.empty_cache() 

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)


    distances, indices = FAISS_INDEX_READY.search(query_embedding.astype('float32'), top_k)
    
    retrieved_contents = []
    retrieved_ids = []
    retrieved_scores = []

    for i, doc_id_raw in enumerate(indices[0]): 
        doc_id = doc_id_raw 
        score = distances[0][i]
        
        content_info = ID_TO_CONTENT.get(doc_id, None)
        
        if content_info:
            retrieved_contents.append(content_info)
            retrieved_ids.append(doc_id)
            retrieved_scores.append(score)
        else:
            print(f"Warning: ID {doc_id} not found in ID_TO_CONTENT mapping.")
            retrieved_contents.append({"type": "missing", "image_content": None, "text_content": f"Content not found for ID: {doc_id}"})
            retrieved_ids.append(doc_id)
            retrieved_scores.append(score)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return retrieved_contents, retrieved_ids, retrieved_scores

## interface funtion
def query_interface(text_query, image_query_pil, top_k=10):

    if FAISS_INDEX_READY is None:
        return "Error: FAISS index not loaded.", *([None]*20)
    if not ID_TO_CONTENT:
        return "Error: Document content mapping not loaded.", *([None]*20)

    query_embedding = None
    status_message = ""

    if not text_query and image_query_pil is None:
        return "Please provide at least a text query or an image query.", *([None]*20)

    try:
        query_txt_str, query_txt_mask_val = _get_padded_text_with_mask(text_query)
        text_inputs = tokenizer([query_txt_str]) 
        query_txt_mask = torch.tensor([query_txt_mask_val], dtype=torch.long)
        
        if image_query_pil is None:
            image_query_pil == None
        else:
            image_query_pil = img_preprocess_fn(image_query_pil.convert("RGB"))
        query_img_tensor, query_img_mask_val = _get_padded_image_with_mask(image_query_pil)
        
        print("text_mask: ",query_txt_mask_val)
        print("image_mask: ", query_img_mask_val)
        
        query_img_tensor = torch.tensor(query_img_tensor).unsqueeze(0)
        query_img_mask = torch.tensor([query_img_mask_val], dtype=torch.long)
        

        batch_for_model = {
            "txt_batched": text_inputs.to(config.device),
            "image_batched": query_img_tensor.to(config.device),
            "txt_mask_batched": query_txt_mask.to(config.device),
            "image_mask_batched": query_img_mask.to(config.device),
            "id_list": [1]
        }
        status_message = f"Searching for query."
        
        with torch.no_grad():
            query_embedding, _ = model(batch_for_model, encode_batch=True)
            query_embedding = query_embedding.cpu().numpy()
    except Exception as e:
        return f"Error embedding query: {e}", *([None]*20)
    
    
    retrieved_contents_info, retrieved_ids, retrieved_scores = perform_search(query_embedding, top_k=top_k)

    output_values_flat_list = [] 

    for i, content_info in enumerate(retrieved_contents_info):
        if i >= TOP_K_DEMO:
            break

        rank = i + 1
        doc_id = retrieved_ids[i]
        score = retrieved_scores[i]
        
        label_text = f"ID: {doc_id}, Score: {score:.4f}"

        image_value = None
        text_value = ""

        image_value = content_info["image_content"]
        text_value = f"{label_text}\n{content_info['text_content']}"
        
        output_values_flat_list.append(image_value)
        output_values_flat_list.append(text_value)

    while len(output_values_flat_list) < TOP_K_DEMO * 2:
        output_values_flat_list.append(None) 
        output_values_flat_list.append("")   

    return status_message, *output_values_flat_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Retriever Demo")
    parser.add_argument("--config_path", default="./config/demo.yaml", help="Path to the config file.")
    parser.add_argument("--merge", default="weight", help="Merge Type")
    parser.add_argument("--ckpt_name", default="vit-l-14/weight/_epoch_10.pth", help="Eval Model PATH")
    parser.add_argument("--pretrained_clip_model_dir", default="ViT-L/14 ", help="Enable Gradio sharing link") # Gradio 공유 링크 활성화를 위한 인자 추가
    
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)

    config.model.merge = args.merge
    config.model.pretrained_clip_model_dir = args.pretrained_clip_model_dir
    config.model.ckpt_config.ckpt_name = args.ckpt_name

    mulimr_dir = config.mulimr_dir
    embed_dir_name = config.retrieval_config.embed_dir_name
    index_dir_name = config.retrieval_config.index_dir_name

    FAISS_INDEX_PATH = os.path.join(
        config.mulimr_dir, config.mulimr_data_dir, index_dir_name, config.model.ckpt_config.ckpt_name, f"cand_it_index.faiss"
    )
    CAND_IDS_PATH = os.path.join(
        config.mulimr_dir, config.mulimr_data_dir, index_dir_name, config.model.ckpt_config.ckpt_name, f"cand_test_it_embed.npy"
    )

    print(f"Attempting to load FAISS index from: {FAISS_INDEX_PATH}")
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at: {FAISS_INDEX_PATH}\nPlease check the ckpt_name argument and path.")

    print("Loading Hugging Face dataset...")
    hf_dataset = load_dataset("Jiiwonn/roco2-question-id-dataset")
    print("Hugging Face dataset loaded successfully.")
    ID_TO_CONTENT = {}
    for entry in hf_dataset['test']:
        ID_TO_CONTENT[entry['id']] = {
            "image_content": entry.get('image', None),
            "text_content": entry.get('caption', None)
        }
    print("ID to content mapping created.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.device = device
    print(f"Using device: {device}")

    print("Building model from config...")
    model = build_model_from_config(config)
    model.to(config.device)
    model.eval()
    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer = model.get_tokenizer()
    print("Model loaded successfully.")

    image_size = (model.clip_model.visual.input_resolution, model.clip_model.visual.input_resolution)
    H, W = image_size
    padded_image = torch.zeros((3, H, W), device=device)

    print("Loading FAISS index...")
    index_cpu = faiss.read_index(FAISS_INDEX_PATH)
    ngpus = faiss.get_num_gpus()
    FAISS_INDEX_READY = None
    if ngpus > 0 and torch.cuda.is_available():
        print(f"Moving FAISS index to {ngpus} GPU(s).")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        index_gpu = faiss.index_cpu_to_all_gpus(index_cpu, co=co, ngpu=ngpus)
        if hasattr(index_gpu, 'nprobe'):
            index_gpu.nprobe = 10
        FAISS_INDEX_READY = index_gpu
        print("FAISS index ready on GPU.")
    else:
        FAISS_INDEX_READY = index_cpu
        if hasattr(FAISS_INDEX_READY, 'nprobe'):
            FAISS_INDEX_READY.nprobe = 10
        print("FAISS index ready on CPU.")

    ## Gradio 인터페이스 구성 및 실행
    TOP_K_DEMO = 10
    status_textbox = gr.Textbox(label="Status", lines=1, interactive=False)
    
    # 인터페이스 함수 생성    
    output_image_components_list = []
    output_text_components_list = [] 
    final_outputs_for_click = [status_textbox] 
    for i in range(TOP_K_DEMO):
        img_comp = gr.Image(label=f"Result {i+1} Image", interactive=False, height=200, scale=1)
        txt_comp = gr.Textbox(label=f"Result {i+1} Text", interactive=False, lines=5, scale=2)

        output_image_components_list.append(img_comp)
        output_text_components_list.append(txt_comp)

        final_outputs_for_click.append(img_comp)
        final_outputs_for_click.append(txt_comp)
    
    with gr.Blocks(title="Multimoda Retriever") as demo:
        gr.Markdown("의료 멀티모달 검색 Demo")
        gr.Markdown("Enter a text query or upload a radiology image to find relevant documents (images and texts).")

        with gr.Row(): 
            text_input = gr.Textbox(label="Enter Text Query", scale=1)
            image_input = gr.Image(type="pil", label="Upload Image Query", scale=1)
            
        with gr.Row():
            search_button = gr.Button("Search", variant="primary", scale=1)
            top_k_slider = gr.Slider(minimum=1, maximum=TOP_K_DEMO, value=TOP_K_DEMO, step=1, label="Top K Results", scale=2)

        status_textbox.render() 

        for i in range(TOP_K_DEMO):
            with gr.Row():
                output_image_components_list[i].render() 
                output_text_components_list[i].render() 

        search_button.click(
            fn=query_interface,
            inputs=[text_input, image_input, top_k_slider],
            outputs=final_outputs_for_click 
        )

    print("\nLaunching Gradio Demo...")
    demo.launch(share=True)