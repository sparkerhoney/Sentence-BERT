from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
import pandas as pd
from tqdm import tqdm

with open('./data/data.json', 'r') as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# def preprocess_db_value(db_value, method):

data