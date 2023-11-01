from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json

# extractions_sample_data load
with open('/Users/marketdesigners/Documents/GitHub/sentencec-bert/data/extractions_sample_data.json', 'r', encoding='utf-8') as f:
    extractions_data = json.load(f)

# lexical_entity_sample_data load
with open('/Users/marketdesigners/Documents/GitHub/sentencec-bert/data/lexical_entity_sample_data.json', 'r', encoding='utf-8') as f:
    lexical_entity_data = json.load(f)

# "extractions_usage_examples" 첫번째 key의 값을 추출
extractions_usage_examples = extractions_data[0]['FoundedEntity']['usage_example']

# "usage_example" 첫번째 key의 값을 추출
lexical_entity_usage_examples = lexical_entity_data[0]['Entity']['usage_example']


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
extractions_encoded_input = tokenizer(extractions_usage_examples, padding=True, truncation=True, return_tensors='pt')
lexical_entity_encoded_input = tokenizer(lexical_entity_usage_examples, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    extractions_model_output = model(**extractions_encoded_input)
    lexical_entity_model_output = model(**lexical_entity_encoded_input)

# Perform pooling
extractions_sentence_embeddings = mean_pooling(extractions_model_output, extractions_encoded_input['attention_mask'])
lexical_entity_sentence_embeddings = mean_pooling(lexical_entity_model_output, lexical_entity_encoded_input['attention_mask'])

# Normalize embeddings
extractions_sentence_embeddings = F.normalize(extractions_sentence_embeddings, p=2, dim=1)
lexical_entity_sentence_embeddings = F.normalize(lexical_entity_sentence_embeddings, p=2, dim=1)

print("extractions_Sentence embeddings:")
print(extractions_sentence_embeddings)

print("#######################################################################")

print("lexical_entity_Sentence embeddings:")
print(lexical_entity_sentence_embeddings)

print("#######################################################################")

# Calculate cosine similarity
cosine_similarity = F.cosine_similarity(extractions_sentence_embeddings, lexical_entity_sentence_embeddings)

print(cosine_similarity.item())  # Print the cosine similarity as a Python float