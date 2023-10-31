from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json

# extractions_sample_data load
with open('extractions_sample_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# lexical_entity_sample_data load
with open('lexical_entity_sample_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# "extractions_usage_examples" 첫번째 key의 값을 추출
extractions_usage_examples = data[0]['FoundedEntity']['usage_example']

# "usage_example" 첫번째 key의 값을 추출
extractions_usage_examples = data[0]['FoundedEntity']['usage_example']


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# # Sentences we want sentence embeddings for
# sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(usage_examples, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)
