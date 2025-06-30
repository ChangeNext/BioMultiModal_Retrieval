import torch
from torch import nn
import torch.nn.functional as F
import clip
import torch.distributed.nn
from accelerate import Accelerator
from open_clip import create_model_from_pretrained, get_tokenizer

accelerator = Accelerator()
class CLIPBIOcoreFusion(nn.Module):
    def __init__(
        self, 
        model_name: str = "ViT-B/32",
        device="cpu", 
        jit=False, 
        download_root=None, 
        config=None,
        gather_embeddings=False
    ):
        super().__init__()
        self.model_name = model_name

        self.clip_model, self.img_preprocess_fn = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)
        self.clip_model = self.clip_model.to(device)
        self.gather_embeddings = gather_embeddings
        self.loss_function = nn.CrossEntropyLoss()
        
    def get_img_preprocess_fn(self):
        return self.img_preprocess_fn
    
    def get_tokenizer(self):
        def tokenizer_wrapper(texts: list[str], context_length=256):
                return self.tokenizer(texts, context_length=context_length)
        return tokenizer_wrapper
    
    def encode_text(self, text_tensor):
        return self.clip_model.encode_text(text_tensor)
    
    def encode_image(self, image_tensor):
        return self.clip_model.encode_image(image_tensor)

    def fuse_embeddings(self, img_emb, txt_emb):
        fused_emb = img_emb + txt_emb
        return fused_emb
    
    def encode_multimodal_input(self, txt_tensor, img_tensor, txt_mask, img_mask):
        txt_emb = self.encode_text(txt_tensor) * txt_mask.unsqueeze(-1)
        img_emb = self.encode_image(img_tensor) * img_mask.unsqueeze(-1)
        return self.fuse_embeddings(txt_emb, img_emb)
    
    def get_logit_scale(self):
        return self.clip_model.logit_scale.exp().float()
    
    def compute_inbatch_contrastive_loss(self, batch):
        txt_batched = batch["txt_batched"]
        image_batched = batch["image_batched"]
        txt_mask_batched = batch["txt_mask_batched"]
        image_mask_batched = batch["image_mask_batched"]
        index_mapping = batch["index_mapping"]
        
        embeddings = self.encode_multimodal_input(txt_batched, image_batched, txt_mask_batched, image_mask_batched)
        q_embeds = embeddings[torch.tensor(index_mapping["query"]).flatten()]  # shape: [bs, embed_dim]
        p_embeds = embeddings[torch.tensor(index_mapping["pos_cand"]).flatten()]  # shape: [bs, embed_dim]
        n_embeds = None
        
        bs = q_embeds.size(0)
        
        q_embeds = F.normalize(q_embeds, dim=-1,  eps=1e-6)
        p_embeds = F.normalize(p_embeds, dim=-1,  eps=1e-6)
        logit_scale = self.get_logit_scale()
        
        if self.gather_embeddings:
            all_p_embeds = accelerator.gather(p_embeds)
            
        if self.gather_embeddings:
            score = torch.matmul(q_embeds, all_p_embeds.t()) * logit_scale  # [bs, bs * num_gpus]
            gpu_id = torch.distributed.get_rank()
            sim_targets = (gpu_id * bs + torch.arange(bs)).to(score.device)  # [bs]
        else:
            score = torch.matmul(q_embeds, p_embeds.t()) * logit_scale  # [bs, bs]
            sim_targets = torch.arange(bs).to(score.device)  # [bs]
            
        # compute loss
        loss = self.loss_function(score, sim_targets)
        _max_score, max_idxs = torch.max(score, 1)
        accuracy = (max_idxs == sim_targets).sum() / bs

        outputs = {"loss": loss, "accuracy": accuracy}
        return outputs

    def forward(self, batch, encode_batch=False):
        if encode_batch:
            return self.encode_batch(batch)
        return self.compute_inbatch_contrastive_loss(batch)
    
    def encode_batch(self, batch):
        # Compute embeddings
        id_list = batch.get("id_list")
    
        embeddings = self.encode_multimodal_input(
            batch["txt_batched"],
            batch["image_batched"],
            batch["txt_mask_batched"],
            batch["image_mask_batched"],
        )
        assert embeddings.size(0) == len(id_list), "embeddings and id_batched must have the same batch size."
        return embeddings, id_list

