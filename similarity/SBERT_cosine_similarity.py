from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences1 = ["""{
        "Entity": {
            "Entity.entity": "get the hang of",
            "sub_type": "idiom",
            "CEFR": "B1",
            "core1": "Education and Knowledge",
            "core2": "Emotions and Characteristics",
            "core3": "Mind",
            "source": "",
            "usage_example": "After a few tries, she finally got the hang of playing the piano.",
            "EXPLANATION": "to become familiar or skilled in doing something."
        }
    }"""]
sentences2 = ["""{
        "FoundedEntity": {
            "Entity.entity": "get the hang of",
            "sub_type": "idiom",
            "usage_example": "It was a bit challenging, especially the past perfect tense, but I think I'm getting the hang of it.",
            "context_from_dialogue": "The student expressed understanding and acquiring skill in using past perfect tense through practice.",
            "reason_of_score": "The idiom is used correctly within the context, indicating the process of understanding or mastering something through experience or practice."
        },
        "Score": 0.85
    }"""]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input1 = tokenizer(sentences1, padding=True, truncation=True, return_tensors='pt')
encoded_input2 = tokenizer(sentences2, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
with torch.no_grad():
    model_output1 = model(**encoded_input1)
    model_output2 = model(**encoded_input2)

# Perform pooling
sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask'])
sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask'])

# Normalize embeddings
sentence_embeddings1 = F.normalize(sentence_embeddings1, p=2, dim=1)
sentence_embeddings2 = F.normalize(sentence_embeddings2, p=2, dim=1)

# Calculate cosine similarity
cosine_similarity = F.cosine_similarity(sentence_embeddings1, sentence_embeddings2)

print(cosine_similarity.item())  # Print the cosine similarity as a Python float