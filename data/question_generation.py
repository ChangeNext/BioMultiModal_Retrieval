import os
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from tqdm import tqdm
import json

df = load_dataset("eltorio/ROCOv2-radiology")
model_name = "Qwen/Qwen3-14B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = 'left' 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt_template = """
You are a radiologist analyzing a radiological image from dataset.
You always generate detailed and clinically relevant 2~3 questions.
You are given a radiological image caption, delimited by triple single quotes.
Create some questions about the radiological image caption.
Output your questions in JSON format as:
[
  {{"question": "<question1>"}},
  {{"question": "<question2>"}},
  ...
]

Radiological image caption: {caption}

Note, questions are never about any change from the last or previous radiological image.
Questions are also never about future plans; questions always focus on the radiological image itself.
Answers are very detailed and include explanations without repeating words that are in the question.
Do not include any additional commentary.
Here are the questions: 
"""

def generate_questions_batch(captions):
    messages_batch = []
    for caption in captions:
        final_prompt = prompt_template.format(caption=caption)
        messages_batch.append([{"role": "user", "content": final_prompt}])

    texts = [tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    ) for messages in messages_batch]

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id 
    )

    batch_contents = []
    input_id_lengths = [len(model_inputs.input_ids[i]) for i in range(len(texts))]

    for i in range(len(captions)):
        output_ids = generated_ids[i][input_id_lengths[i]:].tolist()
        try:
            # Qwen의 </think> 토큰 ID는 151668입니다. 
            # 만약 모델이 thinking 토큰을 출력하지 않았다면, 이 부분은 0이 됩니다.
            index = len(output_ids) - output_ids[::-1].index(151668) 
        except ValueError:
            index = 0
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        batch_contents.append(content)
    
    return batch_contents

split_list = ["train", "validation","test"]
new_splits = {}
batch_size = 64

for split in split_list:
    print(f"Processing {split} split...")
    split_dataset = df[split]
    captions = split_dataset["caption"]
    questions_list = []

    for i in tqdm(range(0, len(captions), batch_size)):
        batch_captions = captions[i:i + batch_size]
        try:
            raw_outputs = generate_questions_batch(batch_captions)

            for raw_output in raw_outputs:
                questions = []
                try:
                    parsed = json.loads(raw_output)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "question" in item:
                                q = item["question"]
                                if isinstance(q, list):
                                    questions.extend(q)
                                else:
                                    questions.append(q)
                except json.JSONDecodeError as json_e:
                    print(f"JSON parsing error for output: '{raw_output[:200]}...' => {json_e}")
                except Exception as other_e:
                    print(f"Other error during JSON processing: {other_e}")
                
                questions_list.append(questions)
        except Exception as e:
            print(f"Error for batch starting with {batch_captions[0][:50]}... => {e}")
            for _ in batch_captions:
                questions_list.append([])

    new_splits[split] = split_dataset.add_column("questions", questions_list)

df_with_questions = DatasetDict(new_splits)
df_with_questions.push_to_hub("Jiiwonn/roco2-question-dataset")