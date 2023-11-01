from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences1 = ["""{
        "Entity": {
            "Entity.entity": "practice makes perfect",
            "sub_type": "idiom",
            "CEFR": "A2",
            "core1": "Education and Knowledge",
            "core2": "Education and Learning",
            "core3": "Education",
            "source": "",
            "usage_example": "Keep practicing, remember, practice makes perfect.",
            "EXPLANATION": "the more you practice, the better you become at something."
        }
    }"""]
sentences2 = ["""{
        "FoundedEntity": {
            "Entity.entity": "practice makes perfect",
            "sub_type": "idiom",
            "usage_example": "You're welcome! Remember, practice makes perfect.",
            "context_from_dialogue": "The tutor encourages the student to keep practicing to improve.",
            "reason_of_score": "The idiom is used in a motivational context, which is a common and appropriate use, encouraging continued effort and improvement."
        },
        "Score": 0.95
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

# Combine embeddings for PCA
combined_embeddings = torch.cat([sentence_embeddings1, sentence_embeddings2], dim=0)

# Convert tensor to numpy array
combined_embeddings_np = combined_embeddings.cpu().numpy()

# PCA를 사용하여 2D로 차원 축소
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(combined_embeddings_np)

# 2D 공간에 점을 그리기
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
labels = ['Sentence 1', 'Sentence 2']
for i, txt in enumerate(labels):
    plt.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]))
plt.title("PCA of Sentence Embeddings")
plt.show()
