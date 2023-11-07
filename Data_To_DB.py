from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
import pandas as pd
from tqdm import tqdm

# Initialize Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Load Data
with open('./data/data.json', 'r') as f:
    data = json.load(f)

# Preprocess Data Function
def preprocess_data(data_entry, method):
    if method == 'only_value':
        return ' '.join([str(value) for key, value in data_entry.items() if key != 'entity'])
    elif method == 'entire_json':
        return json.dumps(data_entry)
    elif method == 'field_name_and_value':
        return ' '.join([f'{key}: {value}' for key, value in data_entry.items() if key != 'entity'])
    elif method == 'multiple_appending':
        return ' '.join([f'{key}: {value} ' * (3 if key == 'entity' else 1) for key, value in data_entry.items()])

# Mean Pooling Function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Embedding and CSV creation
embeddings_df = pd.DataFrame()

for mode in tqdm(['only_value', 'entire_json', 'field_name_and_value', 'multiple_appending'], desc='Modes'):
    mode_embeddings = []
    for entry in tqdm(data, desc=f'Processing {mode}'):
        processed_text = preprocess_data(entry, mode)
        encoded_input = tokenizer(processed_text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embedding = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy().flatten()
        mode_embeddings.append(embedding)
    embeddings_df[mode + '_embedding'] = mode_embeddings

# Add the 'entity' information as the last column
embeddings_df['entity'] = [json.dumps(entry) for entry in data]

# Convert the DataFrame to CSV
csv_path = './data/embeddings.csv'
embeddings_df.to_csv(csv_path, index=False)

print('sucess')
