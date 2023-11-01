from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
import pandas as pd
from tqdm import tqdm

# Load the JSON dataset
with open('/Users/marketdesigners/Documents/GitHub/sentencec-bert/data/lexical_entity_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to mean pool token embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# List to hold dictionary entries and embeddings
embeddings_data = []

# Loop through each entry in the JSON data
for entry in tqdm(data, desc="Processing", unit="entry"):
    # Concatenate all string values from the dictionary entry into one large string
    text_data = ' '.join([str(value) for value in entry.values() if isinstance(value, str)])
    
    # Tokenize the concatenated text
    encoded_input = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling
    sentence_embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
    
    # Convert embeddings to list and append to data
    embeddings_data.append((entry, sentence_embedding[0].tolist()))

# Convert to DataFrame
embeddings_df = pd.DataFrame(embeddings_data, columns=['Dictionary_Entry', 'Embedding'])

# Save to CSV
embeddings_df.to_csv('embeddings.csv', index=False)
