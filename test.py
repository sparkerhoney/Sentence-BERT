from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

with open('./data/data.json', 'r') as f:
    data = json.load(f) # DB Data
with open('./data/target_data.json', 'r') as f:
    target_data = json.load(f) # Extraction Data

def preprocess_data(data_entries, method):
    processed_texts = []
    for data_entry in data_entries:
        if method == 'only_value':
            text_pieces = [str(value) for value in data_entry.values()]
            processed_texts.append(' '.join(filter(None, text_pieces)))
        elif method == 'entire_json':
            processed_texts.append(json.dumps(data_entry))
        elif method == 'field_name_and_value':
            processed_texts.append(' '.join(filter(None, [f"{key}: {value}" for key, value in data_entry.items()])))
        elif method == 'multiple_appending':
            weighted_text_pieces = []
            for key, value in data_entry.items():
                weighted_text_pieces.extend([f"{key}: {value}"] * (3 if key in ['entity'] else 1))
            processed_texts.append(' '.join(weighted_text_pieces))
    return processed_texts

def cal_combined_score(target_entity_embedding, data_entity_embedding, entity_weight=3, json_weight=1):
    entity_score = F.cosine_similarity(target_entity_embedding, data_entity_embedding, dim=1)
    combined_score = entity_score.mean() * entity_weight + json_weight
    return combined_score / (entity_weight + json_weight)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculation(mode):
    processed_target_texts = preprocess_data(target_data, mode)
    target_embeddings = []
    for text in processed_target_texts:
        encoded_target = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_target)
        target_embeddings.append(mean_pooling(model_output, encoded_target['attention_mask']))
    target_embeddings = torch.stack(target_embeddings)

    highest_score = float('-inf')
    highest_scored_value = None

    for entry in tqdm(data, desc=f"Processing mode: {mode}"):
        processed_entry_texts = preprocess_data([entry], mode)
        entry_embeddings = []
        for text in processed_entry_texts:
            encoded_entry = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_entry)
            entry_embeddings.append(mean_pooling(model_output, encoded_entry['attention_mask']))
        entry_embeddings = torch.stack(entry_embeddings)

        combined_score = cal_combined_score(target_embeddings, entry_embeddings)

        if combined_score > highest_score:
            highest_score = combined_score
            highest_scored_value = entry

    print(f"Mode: {mode} - Highest Scored DB Value:", highest_scored_value)
    print(f"Mode: {mode} - Highest Combined Similarity Score:", highest_score)

modes = ['only_value', 'entire_json', 'field_name_and_value', 'multiple_appending']
for mode in modes:
    calculation(mode)
