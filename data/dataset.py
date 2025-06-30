
import clip
import random
import torch
from datasets import load_dataset

from enum import Enum
from typing import Callable, List, Union, Any
from PIL import Image

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# from torchvision.transforms import GaussianBlur

class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"

class CustomDatasetDictDataset(Dataset):
    def __init__(self, dataset=None, dataset_dict_split=None, img_preprocess_fn=None, image_size=None, simclr=True, query_modes=["img_txt", "img_only", "txt_only"]):
        
        self.df = dataset
        self.data = self.df[dataset_dict_split]

        self.img_preprocess_fn = img_preprocess_fn
        self.image_size = image_size
        self.simclr_transform = self._get_simclr_pipeline_transform(self.image_size)
        self.query_modes = query_modes
        
    def _load_and_preprocess_image(self, image):
        image = self.img_preprocess_fn(image.convert("RGB"))
        return image

    def __len__(self):
        return len(self.data)
    
    def _get_simclr_pipeline_transform(self, size, s=1):
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        calculated_kernel_size = int(0.1 * size)
        if calculated_kernel_size % 2 == 0:
            calculated_kernel_size += 1
        kernel_size = max(calculated_kernel_size, 3) 
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=kernel_size),
            transforms.ToTensor()
        ])
        return data_transforms
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        image_data = sample.get("image")
        query_image_tensor = self._load_and_preprocess_image(image_data)
        pos_cand_image_tensor = query_image_tensor
        if simclr is False:
            pos_cand_image_tensor = self.simclr_transform(image_data.convert("RGB")) 

        caption = sample.get("caption")
        
        query_mode = random.choice(self.query_modes)
        questions_list = sample.get("questions")
        
        question = ""  
        query_img = None 
        
        selected_question = ""
        if isinstance(questions_list, list) and questions_list:
            valid_questions = [q.strip() for q in questions_list if isinstance(q, str) and q.strip()]
            if valid_questions:
                selected_question = random.choice(valid_questions)
        
        elif isinstance(questions_list, str) and questions_list.strip():
            selected_question = questions_list.strip()
        
        
        if query_mode == "img_txt": # (이미지+질문) -> (이미지+텍스트)
            query_txt = selected_question # 질문 사용
            query_img = self._load_and_preprocess_image(image_data) 
        elif query_mode == "img_only": # (이미지) -> (이미지+텍스트)
            query_txt = ""
            query_img = self._load_and_preprocess_image(image_data) 
        elif query_mode == "txt_only": # (질문) -> (이미지+텍스트)
            query_txt = selected_question 
            query_img = None
        else:
            raise ValueError(f"Unknown query mode: {query_mode}")
        instance = {
            "query": {
                "txt": query_txt, 
                "img": query_img,  
            },
            "pos_cand": {
                "txt": caption, 
                "img": pos_cand_image_tensor,
            },
        }

        return instance

class CustomDatasetDictDataset_(Dataset):
    def __init__(self, dataset=None, dataset_dict_split=None, img_preprocess_fn=None, image_size=None, simclr=True, query_modes=["img_txt", "img_only", "txt_only"]):
        
        self.df = dataset
        self.data = self.df[dataset_dict_split]

        self.img_preprocess_fn = img_preprocess_fn
        self.image_size = image_size
        self.simclr_transform = self._get_simclr_pipeline_transform(self.image_size)
        self.query_modes = query_modes
        self.simclr=simclr
    def _load_and_preprocess_image(self, image):
        image = self.img_preprocess_fn(image.convert("RGB"))
        return image

    def __len__(self):
        return len(self.data)
    
    def _get_simclr_pipeline_transform(self, size, s=1):
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        
        calculated_kernel_size = int(0.1 * size)
        if calculated_kernel_size % 2 == 0:
            calculated_kernel_size += 1
        kernel_size = max(calculated_kernel_size, 3) 
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=kernel_size),
            transforms.ToTensor()
        ])
        return data_transforms
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        image_data = sample.get("image")
        query_image_tensor = self._load_and_preprocess_image(image_data)
        
        pos_cand_image_tensor = query_image_tensor
        if self.simclr is False:
            pos_cand_image_tensor = self.simclr_transform(image_data.convert("RGB")) 

        caption_ = sample.get("caption")
        
        query_mode = random.choice(self.query_modes)
        query_mode_target = random.choice(self.query_modes)
        
        questions_list = sample.get("questions")
        
        caption = ""
        question = ""  
        query_img = None 
        
        selected_question = ""
        if isinstance(questions_list, list) and questions_list:
            valid_questions = [q.strip() for q in questions_list if isinstance(q, str) and q.strip()]
            if valid_questions:
                selected_question = random.choice(valid_questions)
        
        elif isinstance(questions_list, str) and questions_list.strip():
            selected_question = questions_list.strip()
        
        
        if query_mode == "img_txt": # (이미지+질문) -> (이미지+텍스트)
            query_txt = selected_question # 질문 사용
            query_img = query_image_tensor
        elif query_mode == "img_only": # (이미지) -> (이미지+텍스트)
            query_txt = ""
            query_img = query_image_tensor
        elif query_mode == "txt_only": # (질문) -> (이미지+텍스트)
            query_txt = selected_question 
            query_img = None
        else:
            raise ValueError(f"Unknown query mode: {query_mode}")
        
        if query_mode_target == "img_txt": # (이미지+질문) -> (이미지+텍스트)
            caption = caption_
            target_img = pos_cand_image_tensor 
        elif query_mode_target == "img_only": # (이미지) -> (이미지+텍스트)
            caption = ""
            target_img = pos_cand_image_tensor 
        elif query_mode_target == "txt_only": # (질문) -> (이미지+텍스트)
            caption = caption_
            target_img = None
        else:
            raise ValueError(f"Unknown query mode: {query_mode}")
        
        
        instance = {
            "query": {
                "txt": query_txt, 
                "img": query_img,  
            },
            "pos_cand": {
                "txt": caption, 
                "img": target_img,
            },
        }

        return instance


class Candidates_Dataset(Dataset):
    def __init__(self, dataset=None, dataset_dict_split=None, img_preprocess_fn=None, query_modes="img_txt"):
        
        self.df = dataset
        self.data = self.df[dataset_dict_split]
        self.img_preprocess_fn = img_preprocess_fn
        self.query_modes = query_modes
        
    def _load_and_preprocess_image(self, image):
        image = self.img_preprocess_fn(image.convert("RGB"))
        return image

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        image_data = sample.get("image")
        image_id = sample.get("id") 
        caption = ""
        img = None 
        if self.query_modes == "img_txt":
            caption = sample.get("caption")
            img = self._load_and_preprocess_image(image_data)
        elif self.query_modes == "img_only": 
            caption = ""
            img = self._load_and_preprocess_image(image_data)
        elif self.query_modes == "txt_only":
            caption = sample.get("caption")
            img = None
            
            
        instance = {
                "txt": caption, 
                "img": img,
                'id' : image_id
        }
        return instance

class Query_Dataset(Dataset):
    def __init__(self, dataset=None, dataset_dict_split=None, img_preprocess_fn=None, query_modes="img_txt"):
        
        self.df = dataset
        self.data = self.df[dataset_dict_split]
        self.img_preprocess_fn = img_preprocess_fn
        self.query_modes = query_modes
    
    def _load_and_preprocess_image(self, image):
        image = self.img_preprocess_fn(image.convert("RGB"))
        return image

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        image_data = sample.get("image")
        image_id = sample.get("id") 
        questions_list = sample.get("questions")
        
        question = ""  
        query_img = None 
        
        selected_question = ""
        if isinstance(questions_list, list) and questions_list:
            valid_questions = [q.strip() for q in questions_list if isinstance(q, str) and q.strip()]
            if valid_questions:
                selected_question = valid_questions[0]
        
        if self.query_modes == "img_txt":
            query_txt = selected_question 
            query_img = self._load_and_preprocess_image(image_data) 
        elif self.query_modes == "img_only": 
            query_txt = ""
            query_img = self._load_and_preprocess_image(image_data) 
        elif self.query_modes == "txt_only": 
            query_txt = selected_question 
            query_img = None
        else:
            raise ValueError(f"Unknown query mode: {query_mode}")
        instance = {
                "txt": query_txt, 
                "img": query_img,
                'id' : image_id
        }
        return instance
    
    
class CollatorBase(object):
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: Union[tuple, int]):
        
        self.tokenizer = tokenizer
        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.H, self.W = image_size
        self.padded_image = torch.zeros((3, self.H, self.W))  # Note: this is a black image
        self.padded_txt = "" 
    
    def _get_padded_text_with_mask(self, txt):
        return (txt, 1) if txt not in [None, ""] else (self.padded_txt, 0)
    
    def _get_padded_image_with_mask(self, img):
        return (img, 1) if img is not None else (self.padded_image, 0)
    
    def __call__(self, batch):
        raise NotImplementedError("This method should be implemented in derived classes.")

class MainCollator(CollatorBase):
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: tuple, device, mode=Mode.TRAIN, model_name = None, ):
        super().__init__(tokenizer, image_size)
        self.model_name = model_name
        self.mode = mode
        self.device = device

    def __call__(self, batch):
        txt_list, txt_mask_list, img_list, img_mask_list = [], [], [], []
        question_list, caption_list = [], []
        index_mapping = {
            "query": [[] for _ in range(len(batch))],
            "pos_cand": [[] for _ in range(len(batch))], 
        }
        instance_keys = ["query", "pos_cand"] 
        
        counter = 0
        for inst_idx, instance in enumerate(batch):
            for instance_key in instance_keys:
                items = [instance[instance_key]] if instance_key != "neg_cand_list" else instance[instance_key]  # list
                for item in items:
                    txt = item["txt"]
                    img = item["img"]

                    index_mapping[instance_key][inst_idx].append(counter)  # Track current index
                    counter += 1
                    caption_list.append(txt)
                    padded_txt, txt_mask = self._get_padded_text_with_mask(txt)
                    padded_img, img_mask = self._get_padded_image_with_mask(img)
                    txt_list.append(padded_txt)
                    img_list.append(padded_img)
                    txt_mask_list.append(txt_mask)
                    img_mask_list.append(img_mask)
        
        processed_batch = {
            "txt_batched": self.tokenizer(txt_list).to(self.device),
            "image_batched": torch.stack(img_list, dim=0).to(self.device),
            "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long).to(self.device),
            "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long).to(self.device),
            "index_mapping": index_mapping,
            "caption" : caption_list
        }
    
        # if self.mode == Mode.EVAL:
        #     if qid_list:
        #         processed_batch.update({"qid_list": qid_list})
        #     if task_id_list:
        #         processed_batch.update({"task_id_list": task_id_list})

        # if self.mode == Mode.TRAIN:
        #     if p_did_list:
        #         processed_batch.update({"p_did_list": torch.tensor(p_did_list)})

        # TODO: Fix this hack for BLIP tokenizer.
        if hasattr(processed_batch["txt_batched"], "input_ids"):
            bs = processed_batch["txt_batched"]["input_ids"].size(0)
        else:
            bs = len(processed_batch["txt_batched"])
        assert bs == processed_batch["image_batched"].size(0)
        assert bs == processed_batch["txt_mask_batched"].size(0)
        assert bs == processed_batch["image_mask_batched"].size(0)
        return processed_batch
    
class CandidatePoolCollator(CollatorBase):
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: tuple):
        super().__init__(tokenizer, image_size)

    def __call__(self, batch):
        txt_list, txt_mask_list, img_list, img_mask_list, id_list = [], [], [], [], []
        for instance in batch:
            txt = instance["txt"]
            img = instance["img"]
            padded_txt, txt_mask = self._get_padded_text_with_mask(txt)
            padded_img, img_mask = self._get_padded_image_with_mask(img)
            txt_list.append(padded_txt)
            img_list.append(padded_img)
            txt_mask_list.append(txt_mask)
            img_mask_list.append(img_mask)

            image_id = instance['id']
            id_list.append(image_id)


        processed_batch = {
            "txt_batched": self.tokenizer(txt_list),
            "image_batched": torch.stack(img_list, dim=0),
            "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long),
            "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long),
        }

        if id_list:
            processed_batch.update({"id_list": id_list})

        if hasattr(processed_batch["txt_batched"], "input_ids"):
            bs = processed_batch["txt_batched"]["input_ids"].size(0)
        else:
            bs = len(processed_batch["txt_batched"])
        assert bs == processed_batch["image_batched"].size(0)
        assert bs == processed_batch["txt_mask_batched"].size(0)
        assert bs == processed_batch["image_mask_batched"].size(0)
        return processed_batch

class QueryCollator(CollatorBase):
    def __init__(self, tokenizer: Callable[[List[str]], Any], image_size: tuple):
        super().__init__(tokenizer, image_size)

    def __call__(self, batch):
        txt_list, txt_mask_list, img_list, img_mask_list, id_list = [], [], [], [], []
        # Candidate can be indexed directly from the batch
        for instance in batch:
            txt = instance["txt"]
            img = instance["img"]
            padded_txt, txt_mask = self._get_padded_text_with_mask(txt)
            padded_img, img_mask = self._get_padded_image_with_mask(img)
            txt_list.append(padded_txt)
            img_list.append(padded_img)
            txt_mask_list.append(txt_mask)
            img_mask_list.append(img_mask)

            image_id = instance['id']
            id_list.append(image_id)

        processed_batch = {
            "txt_batched": self.tokenizer(txt_list),
            "image_batched": torch.stack(img_list, dim=0),
            "txt_mask_batched": torch.tensor(txt_mask_list, dtype=torch.long),
            "image_mask_batched": torch.tensor(img_mask_list, dtype=torch.long),
        }

        if id_list:
            processed_batch.update({"id_list": id_list})

        if hasattr(processed_batch["txt_batched"], "input_ids"):
            bs = processed_batch["txt_batched"]["input_ids"].size(0)
        else:
            bs = len(processed_batch["txt_batched"])
        assert bs == processed_batch["image_batched"].size(0)
        assert bs == processed_batch["txt_mask_batched"].size(0)
        assert bs == processed_batch["image_mask_batched"].size(0)
        return processed_batch