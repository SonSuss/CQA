import json
import os
import glob

from models.components.utils import disable_torch_init
from models.chart_qa_model.builder import load_pretrained_llava_model
from models.components.constants import IMAGE_TOKEN_INDEX
from models.components.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from models.components.conversation import conv_templates, SeparatorStyle
from eval.chart_qa.metric import chartqa_evaluator

import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class EvalDataset(Dataset):
    def __init__(self, data_items, image_folder, tokenizer, image_processor, conv):
        self.data_items = data_items
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_mode = conv

    def __getitem__(self, index):
        line = self.data_items[index]
        image_file = line["image"]
        qs = line["conversations"][0]["value"]
        conv = self.conv_mode.copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor= self.image_processor.preprocess(image,return_tensors='pt')['pixel_values']

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)
        
        return input_ids, image_tensor, attention_mask

    def __len__(self):
        return len(self.data_items)


def collate_fn(batch):
    input_ids, image_tensors, attention_masks = zip(*batch)
    
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    return input_ids, image_tensors, attention_masks

def create_data_loader(questions, image_folder, tokenizer, image_processor, conv, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = EvalDataset(questions, image_folder, tokenizer, image_processor, conv)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    return data_loader

def get_eval(model_path, valset_path, output_path, image_folder="", conv_mode="phi4_instruct", temperature=0.0, top_p=1.0, max_new_tokens=1024, min_new_tokens=1, num_beams=1):
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_pretrained_llava_model(model_path)
    all_data = json.load(open(valset_path, "r"))
    os.makedirs(output_path, exist_ok=True)
    answers_file = os.path.join(output_path, f"answers_{temperature}_{top_p}.json")
    ans_file = []
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.PHI4 else conv.sep
    keywords = [stop_str]
    data_loader = create_data_loader(all_data, image_folder, tokenizer, image_processor, conv, num_workers=4)
    with torch.inference_mode():
        for (input_ids, image_tensor, attention_mask), line in tqdm(zip(data_loader, all_data), total=len(all_data)):
            idx = line["id"]
            cur_prompt = line["conversations"][0]["value"]
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask.to(device='cuda', non_blocking=True),
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if temperature > 0 else False,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            ans_file.append({
                "id": idx,
                "question": cur_prompt,
                "gt_answer": line["conversations"][1]["value"],
                "final_model_answer": outputs
            })

    with open(answers_file, "w", encoding="utf-8") as f:
        json.dump(ans_file, f)
def eval_model(eval_path, output_path):
    model_file = [
        name for name in os.listdir(eval_path)
        if os.path.isdir(os.path.join(eval_path, name))
    ]
    for model_path in model_file:
        full_model_path = os.path.join(eval_path, model_path)
        answers_files = glob.glob(os.path.join(full_model_path, "*.json"))
        eval_file = os.path.join(output_path, f"eval.json")
        eval_results = []
        for answers_file in answers_files:
            eval_config=answers_file.split("answers")[-1]
            with open(answers_file, "r", encoding="utf-8") as f:
                answers = json.load(f)
            standard_accuracy, relaxed_accuracy = chartqa_evaluator(answers)
            eval_results.append({
                "eval_config": eval_config,
                "standard_accuracy": standard_accuracy,
                "relaxed_accuracy": relaxed_accuracy
            })
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_results, f)