import os

import argparse
import yaml
import logging
import random
import numpy as np

from omegaconf import OmegaConf
from datasets import load_dataset
from data.dataset import CustomDatasetDictDataset, CustomDatasetDictDataset_, MainCollator, Mode
from src.models.model import CLIPScoreFusion, CLIPWeightFusion, CLIPMLPFusion
import torch
from torch.optim import AdamW
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from types import SimpleNamespace
from src.utils import train_one_epoch, eval_engine

logger = logging.getLogger()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def log_results(train_stats, val_stats, test_stats, epoch=None, best_epoch=None):
    log_stats = {}
    if train_stats:
        log_stats.update({f"train_{k}": v for k, v in train_stats.items()})
    if val_stats:
        log_stats.update({f"val_{k}": v for k, v in val_stats.items()})
    if test_stats:
        log_stats.update({f"test_{k}": v for k, v in test_stats.items()})
    if epoch is not None:
        log_stats["epoch"] = epoch
    if best_epoch is not None:
        log_stats["best_epoch"] = best_epoch
    return log_stats

def filter_parameters(model, condition_fn):
    named_parameters = model.named_parameters()
    return [p for n, p in named_parameters if condition_fn(n, p) and p.requires_grad]

def create_optimizer(gain_or_bias_params, rest_params, config):
    return AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": 0.2},
        ],
        lr=float(config.trainer_config.learning_rate),
        betas=(0.9, 0.98),
        eps=1.0e-6,
    )

def save_checkpoint(model, optimizer, scheduler, epoch, scaler, config, best):
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    if best:
        checkpoint_name = f"{model_name}_epoch_best.pth"
    else:
        checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
        
    save_obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config,
        "epoch": epoch,
        "scaler": scaler.state_dict(),
    }
    checkpoint_path = os.path.join(ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(save_obj, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def train(
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    scaler,
    config,
    epoch
):
    writer = SummaryWriter()
    global_step, total_loss, best_inbatch_accuracy = (
        0,
        0.0,
        0.0,
    ) 
    best_epoch = 0
    model.zero_grad()
    best_validation_loss = float('inf')
    
    if epoch != 0:
        print(f"Resuming training from epoch {epoch}")
    
    for epoch in range(epoch, config.trainer_config.num_train_epochs):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            config.trainer_config.device,
            scheduler,
            global_step,
            scaler,
            writer,
            config,
        )
        
        eval_freq = config.evaluator.eval_freq
        if val_loader is None or epoch % eval_freq != 0:
            log_stats = log_results(train_stats, None, None, epoch, best_epoch)
        else:
            val_status = eval_engine(model, writer, val_loader, config.trainer_config.device)
            
            try:
                validation_loss = float(val_status["loss"])
            except ValueError:
                print(f"Error: Expected a number but got '{val_status['loss']}' for validation loss")
                validation_loss = float('inf')
            save_checkpoint(model, optimizer, scheduler, epoch, scaler, config, best=False)    
            if validation_loss <= best_validation_loss:
                best_validation_loss = validation_loss
                save_checkpoint(model, optimizer, scheduler, epoch, scaler, config, best=True)
                best_epoch = epoch

            log_stats = log_results(train_stats, val_status, None, epoch, best_epoch)
        
    torch.cuda.empty_cache()
    
                
def main(config):

    config.trainer_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Merge Style: {config.trainer_config.merge}")
    model=None
    # Merge Score
    if config.trainer_config.merge== "score":
        model = CLIPScoreFusion(
            model_name=config.trainer_config.pretrained_clip_model_dir,
            device = config.trainer_config.device
        )
        
    # Merge Weight
    elif config.trainer_config.merge== "weight":
        model = CLIPWeightFusion(
            model_name=config.trainer_config.pretrained_clip_model_dir,
            device = config.trainer_config.device
        )
        
    # Merge MLP
    elif config.trainer_config.merge== "mlp":
        model = CLIPMLPFusion(
            model_name=config.trainer_config.pretrained_clip_model_dir,
            device = config.trainer_config.device
        ) 
        
    model.float().to(config.trainer_config.device)

    exclude_condition = lambda n, p: p.ndim < 2 or any(sub in n for sub in ["bn", "ln", "bias", "logit_scale"])
    include_condition = lambda n, p: not exclude_condition(n, p)
    gain_or_bias_params = filter_parameters(model, exclude_condition)
    rest_params = filter_parameters(model, include_condition)
    
    optimizer = create_optimizer(gain_or_bias_params, rest_params, config)
    scaler = GradScaler() 
    
    dataset = load_dataset(config.dataset.name)

    ckpt_config = config.model_checkpoint
    ## resume train
    if ckpt_config.resume_training:
        checkpoint_path = os.path.join(config.save_dir)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        logger.info(f"loading CLIPScoreFusion checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
    
    model.train()
    
    logger.info("Preparing dataset ...")  
    logger.info(f"Loading dataset from {config.dataset.name}")
    
    img_preprocess_fn = model.get_img_preprocess_fn()
    tokenizer_fn = model.get_tokenizer()

    image_size = (model.clip_model.visual.input_resolution, model.clip_model.visual.input_resolution)
    
    train_dataset = CustomDatasetDictDataset_(
            dataset = dataset,
            dataset_dict_split="train",
            img_preprocess_fn=img_preprocess_fn,
            image_size = image_size[0],
            simclr=config.dataset.simclr,
            query_modes=["img_txt", "img_only", "txt_only"]
        )
    
    eval_dataset = CustomDatasetDictDataset_(
            dataset = dataset,
            dataset_dict_split="validation",
            img_preprocess_fn=img_preprocess_fn,
            image_size = image_size[0],
            simclr=False,
            query_modes=["img_txt"]
        )
    ## DataLoader
    collator = MainCollator(
        tokenizer=tokenizer_fn, 
        image_size=image_size, 
        device=config.trainer_config.device,
        model_name = config.trainer_config.pretrained_clip_model_dir,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.trainer_config.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.trainer_config.eval_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    # Initializing the scheduler
    t_total = (
        len(train_dataloader) // config.trainer_config.gradient_accumulation_steps * config.trainer_config.num_train_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_total, eta_min=0)
    
    epoch = 0
    
    train(
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        epoch,
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", default="./config/train.yaml", help="Path to the config file.")
    parser.add_argument("--batch_size", default="32")
    parser.add_argument("--epochs", default=10)
    parser.add_argument("--pretrained_clip_model_dir", default="ViT-L/14", help="Pretrain Model")
    parser.add_argument("--merge", default="weight", help="Merge type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)
    
    config.trainer_config.batch_size = args.batch_size
    config.trainer_config.merge = args.merge
    config.trainer_config.pretrained_clip_model_dir = args.pretrained_clip_model_dir
    config.trainer_config.num_train_epochs = args.epochs
    set_seed(config.trainer_config.seed)
    main(config)
