import json
import os

from models.components.utils import disable_torch_init
from models.chart_qa_model.builder import load_pretrained_llava_model
from models.components.constants import IMAGE_TOKEN_INDEX
from models.components.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from models.components.conversation import conv_templates, SeparatorStyle

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

def RelaxedAccuracy(pred, gt):
    try:
        gt = float(gt)
        pred = float(pred)
        if gt == 0.0:
            if pred == gt:
                return 1.0
            else:
                return 0.0
        else:
            if abs(pred-gt) / gt <= 0.05:
                return 1.0
            else:
                return 0.0
    except:
        if str(gt) == str(pred):
            return 1.0
        else:
            return 0.0

def collate_fn(batch):
    input_ids, image_tensors, attention_masks = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    return input_ids, image_tensors, attention_masks

def create_data_loader(questions, image_folder, tokenizer, image_processor, conv, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = EvalDataset(questions, image_folder, tokenizer, image_processor, conv)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader

def chartqa_evaluator(data, key='final_model_answer'):
    acc = 0
    for item in data:
        item['relaxed_acc'] = RelaxedAccuracy(item[key], item['gt_answer'].split('<pot_note>')[0])
        if item['relaxed_acc'] == 1.0:
            acc += 1
    accuracy = acc/len(data)
    return data, accuracy

def get_eval(model_path, valset_path, output_path, image_folder="", conv_mode="phi4_instruct", temperature=0.0, top_p=1.0, max_new_tokens=1024, min_new_tokens=1, num_beams=1):
    disable_torch_init()
    model, tokenizer, image_processor, context_len = load_pretrained_llava_model(model_path)
    all_data = json.load(open(valset_path, "r"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    answers_file = os.path.join(output_path, "answers.json")
    ans_file = []
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.PHI4 else conv.sep
    keywords = [stop_str]
    data_loader = create_data_loader(all_data, image_folder, tokenizer, image_processor, conv, num_workers=4)
    for (input_ids, image_tensor, attention_mask), line in tqdm(zip(data_loader, all_data), total=len(all_data)):
        idx = line["id"]
        cur_prompt = line["conversations"][0]["value"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        attention_mask = attention_mask.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if temperature > 0 else False,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
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

def eval_model(eval_path):
    eval_file = os.path.join(eval_path,"answers.json")
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Evaluation file {eval_file} does not exist.")
    with open(eval_file, "r") as f:
        data = json.load(f)
        

    
    