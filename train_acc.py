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
from src.models.biomodel import CLIPBIOcoreFusion
import torch
from torch.optim import AdamW
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from types import SimpleNamespace
from src.utils import train_one_epoch, train_one_epoch_, eval_engine # train_one_epoch_ is likely the accelerated version you'll use

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed as accelerate_set_seed

logger = logging.getLogger()

def set_seed(seed):
    accelerate_set_seed(seed)
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

# Modified save_checkpoint to use Accelerator's saving mechanism
def save_checkpoint(accelerator, config, epoch, best):
    ckpt_config = config.model.ckpt_config
    model_name = config.model.short_name.lower()
    
    if best:
        checkpoint_name = f"{model_name}_epoch_best.pth"
    else:
        checkpoint_name = f"{model_name}_epoch_{epoch}.pth"
            
    checkpoint_path = os.path.join(ckpt_config.ckpt_dir, checkpoint_name)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    accelerator.save_state(checkpoint_path)
    if accelerator.is_main_process:
        print(f"Saved checkpoint to {checkpoint_path}")

def train(
    train_loader,
    val_loader,
    model,
    optimizer,
    scheduler,
    config,
    resume_epoch,
    accelerator,
    writer=None,
):
    global_step = 0
    best_validation_loss = float("inf")
    best_epoch = 0

    if resume_epoch != 0 and accelerator.is_main_process:
        print(f"Resuming training from epoch {resume_epoch}")

    for epoch in range(resume_epoch, config.trainer_config.num_train_epochs):
        # 1. Train One Epoch
        train_stats = train_one_epoch_(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            accelerator=accelerator,
            epoch=epoch,
            writer=writer,
            config=config,
        )

        # 2. Validation
        do_validation = (val_loader is not None) and (epoch % config.evaluator.eval_freq == 0)
        if not do_validation:
            if accelerator.is_main_process:
                print(f"Epoch {epoch} finished. Train stats: {train_stats}")
            continue

        # model.eval()
        # with torch.no_grad():
        #     val_stats = evaluate(
        #         model=model,
        #         dataloader=val_loader,
        #         accelerator=accelerator,
        #         writer=writer,
        #         epoch=epoch,
        #     )
        # model.train()

        # validation_loss = val_stats.get("loss", float("inf"))

        # # 3. Checkpoint 저장
        # if accelerator.is_main_process:
        #     save_checkpoint(accelerator, config, epoch, best=False)
        #     if validation_loss <= best_validation_loss:
        #         best_validation_loss = validation_loss
        save_checkpoint(accelerator, config, epoch, best=True)
        # Save epoch number too
        epoch_info_path = os.path.join(config.model.ckpt_config.ckpt_dir, "epoch_info.pt")
        torch.save({"epoch": epoch}, epoch_info_path)
        best_epoch = epoch

        # 로그 출력
        print(f"Epoch {epoch} finished.")
        print(f"Train Stats: {train_stats}")
            # print(f"Validation Stats: {val_stats}")

    if writer and accelerator.is_main_process:
        writer.close()

    torch.cuda.empty_cache()
    
def main(config):
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"Using device: {device}")
        writer = SummaryWriter(log_dir=config.model.short_name)
    else:
        writer = None
        
    if accelerator.is_main_process:
        print(f"Merge Style: {config.trainer_config.merge}")
    
    model=None
    if config.trainer_config.merge == "score":
        model = CLIPScoreFusion(
            model_name=config.trainer_config.pretrained_clip_model_dir,
            device = config.trainer_config.device
        )
    elif config.trainer_config.merge == "weight":
        model = CLIPWeightFusion(
            model_name=config.trainer_config.pretrained_clip_model_dir,
            device = config.trainer_config.device
        )
    elif config.trainer_config.merge == "mlp":
        model = CLIPMLPFusion(
            model_name=config.trainer_config.pretrained_clip_model_dir,
            device = config.trainer_config.device
        ) 
    model.float()

    exclude_condition = lambda n, p: p.ndim < 2 or any(sub in n for sub in ["bn", "ln", "bias", "logit_scale"])
    include_condition = lambda n, p: not exclude_condition(n, p)
    gain_or_bias_params = filter_parameters(model, exclude_condition)
    rest_params = filter_parameters(model, include_condition)
    optimizer = create_optimizer(gain_or_bias_params, rest_params, config)


    dataset = load_dataset(config.dataset.name)
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
    
    t_total = (
        len(train_dataloader) // config.trainer_config.gradient_accumulation_steps * config.trainer_config.num_train_epochs
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=t_total, eta_min=0)
    
    # 2. Prepare all training components
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

    # 3. Handle resume training using accelerator's load_state
    resume_epoch = 0
    ckpt_config = config.model_checkpoint
    if ckpt_config.resume_training:
        ckpt_path = os.path.join(ckpt_config.ckpt_dir, f"{config.model.short_name.lower()}_epoch_best.pth")
        if os.path.exists(ckpt_path):
            accelerator.load_state(ckpt_path)
            epoch_info_path = os.path.join(ckpt_config.ckpt_dir, "epoch_info.pt")
            if os.path.exists(epoch_info_path):
                resume_epoch = torch.load(epoch_info_path).get("epoch", 0)
            if accelerator.is_main_process:
                print(f"Resumed from {ckpt_path}, starting at epoch {resume_epoch}")
        else:
            if accelerator.is_main_process:
                print("No checkpoint found. Starting from scratch.")
            
    # model.train() # No need to call model.train() here, train function will set it.

    # 4. Pass accelerator to the train function
    train(
        train_dataloader,
        eval_dataloader,
        model,
        optimizer,
        scheduler,
        # scaler, # Removed scaler
        config,
        resume_epoch,
        accelerator=accelerator,
        writer=writer,
    )

def parse_arguments():
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config_path", default="./config/config.yaml",help="Path to the config file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)
    
    set_seed(config.trainer_config.seed)
    main(config)