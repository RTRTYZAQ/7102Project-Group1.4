import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
from torch import nn
import os
from tqdm import tqdm
from collections import Counter
import re
import torch.nn.functional as F
import math


class BertWithMLP(nn.Module):
    def __init__(self, bert, hidden_size=768, mlp_hidden_size1=1024, mlp_hidden_size2 =256, num_classes=10):
        super(BertWithMLP, self).__init__()
        self.bert = bert
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden_size2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        cls = outputs.last_hidden_state[:, 0, :]
        
        logits = self.mlp(cls)
        
        return logits


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class CrossAttention(nn.Module):
    def __init__(self, num_count, dim_count, dim_bert, hidden_size, num_head):
        super(CrossAttention, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size

        self.count_embeding = nn.Embedding(num_count, dim_count)

        self.proj_Q = nn.Linear(dim_count, hidden_size)
        self.proj_K = nn.Linear(dim_bert, hidden_size)
        self.proj_V = nn.Linear(dim_bert, hidden_size)

        self.linear = nn.Linear(hidden_size, hidden_size)

        self.attention_drop = nn.Dropout(0.1)
        self.linear_drop = nn.Dropout(0.1)

        self.layer_norm1 = LayerNorm(dim_count)
        self.layer_norm2 = LayerNorm(hidden_size)
        self.layer_norm3 = LayerNorm(hidden_size)

        self.fnn1 = nn.Linear(hidden_size, 3072)
        self.fnn2 = nn.Linear(3072, hidden_size)
        self.fnn_dropout = nn.Dropout(0.1)
    
    def forward(self, bert_output, count):
        count_token = self.count_embeding(count).unsqueeze(dim=1)

        batch_size = bert_output.shape[0]

        Q = self.proj_Q(self.layer_norm1(count_token)).view(batch_size, -1, self.num_head, int(self.hidden_size / self.num_head)).transpose(1, 2)
        K = self.proj_K(self.layer_norm2(bert_output)).view(batch_size, -1, self.num_head, int(self.hidden_size / self.num_head)).transpose(1, 2)
        V = self.proj_V(self.layer_norm2(bert_output)).view(batch_size, -1, self.num_head, int(self.hidden_size / self.num_head)).transpose(1, 2)

        temp_output = self.linear_drop(self.linear(torch.matmul(self.attention_drop(F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size / self.num_head), dim=-1)), V).transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size))).squeeze(dim=1) + bert_output[:, 0, :]

        return self.fnn2(self.fnn_dropout(nn.ReLU()(self.fnn1(self.layer_norm3(temp_output))))) + temp_output


class BertWithMLP_CrossAttention(nn.Module):
    def __init__(self, bert, hidden_size=768, mlp_hidden_size1=1024, mlp_hidden_size2 =256, num_classes=10, num_count=332):
        super(BertWithMLP_CrossAttention, self).__init__()
        self.bert = bert
        self.cross_attention = CrossAttention(num_count, 768, 768, 768, 12)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size2),
            nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(mlp_hidden_size1, mlp_hidden_size2),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden_size2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask, count):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        cls = self.cross_attention(outputs.last_hidden_state, count)

        
        logits = self.mlp(cls)
        
        return logits
    

def tokenize(tokenizer, x, max_len):

    if type(x) != list:
        x = [x]

    token = [tokenizer.encode_plus(
    text,
    truncation=True,
    add_special_tokens=True,
    max_length=max_len,            
    pad_to_max_length=True,  
    return_attention_mask=True,  
    return_tensors='pt',      
    ) for text in x]

    return token


def convert_class(x):
    return {
        'input_ids': torch.cat(tuple([_['input_ids'] for _ in x]), dim=0),
        'token_type_ids': torch.cat(tuple([_['token_type_ids'] for _ in x]), dim=0),
        'attention_mask': torch.cat(tuple([_['attention_mask'] for _ in x]), dim=0),
    }


def convert_rating(x1, x2):
    return {
        'input_ids': torch.cat(tuple([_['input_ids'] for _ in x1]), dim=0),
        'token_type_ids': torch.cat(tuple([_['token_type_ids'] for _ in x1]), dim=0),
        'attention_mask': torch.cat(tuple([_['attention_mask'] for _ in x1]), dim=0),
        'usefulcount': torch.tensor(x2),
    }


def Bert_Inference(task='class_prediction', model_path='drug_class_prediction_best_model.pth', x='Zoloft'):
    
    valid_task = {'class_prediction', 'rating_prediction'}
    if task not in valid_task:
        raise ValueError(f"mode must be one of {valid_task}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BERT = BertModel.from_pretrained("bert-base-uncased")

    if task == 'class_prediction':
        x = tokenize(tokenizer, x, 50)

        model = BertWithMLP(BERT, hidden_size=768, mlp_hidden_size1=1024, mlp_hidden_size2=256, num_classes=6).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    
        x = convert_class(x)

        outputs = model(input_ids=x['input_ids'].to(device), attention_mask=x['attention_mask'].to(device))

        class_ = torch.argmax(outputs, dim=-1)

        Num_to_Class = {
            0: 'Analgesics',
            1: 'Mood Stabilizers',
            2: 'Antibiotics',
            3: 'Antiseptics',
            4: 'Antimalarial',
            5: 'Antipiretics',
        }

        result = [Num_to_Class[int(_)] for _ in class_]
    
    if task == 'rating_prediction':
        x_ = tokenize(tokenizer, x[0], 64)

        model = BertWithMLP_CrossAttention(BERT, hidden_size=768, mlp_hidden_size1=1024, mlp_hidden_size2=256, num_classes=10, num_count=2000).to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    
        x = convert_rating(x_, x[1])

        outputs = model(input_ids=x['input_ids'].to(device), attention_mask=x['attention_mask'].to(device), count=x['usefulcount'].to(device))

        result = torch.argmax(outputs, dim=-1)


    return result

#示例：

# ##class prediction单个drug name：
# print(Bert_Inference(task='class_prediction', model_path='drug_class_prediction_best_model.pth', x='Sertraline'))
#
# ##class prediction多个drug name：
# print(Bert_Inference(task='class_prediction', model_path='drug_class_prediction_best_model.pth', x=['Sertraline', 'Mobic', 'Azithromycin']))
#
#
# ##rating prediction单个review：
# print(Bert_Inference(task='rating_prediction', model_path='rating_cross_attention_best_model.pth', x=[['It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil'], [27]]))
#
# ##rating prediction多个review：
# print(Bert_Inference(task='rating_prediction', model_path='rating_cross_attention_best_model.pth', x=[['It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil', "Abilify changed my life. There is hope. I was on Zoloft and Clonidine when I first started Abilify at the age of 15.. Zoloft for depression and Clondine to manage my complete rage. My moods were out of control. I was depressed and hopeless one second and then mean, irrational, and full of rage the next. My Dr. prescribed me 2mg of Abilify and from that point on I feel like I have been cured though I know I&#039;m not.. Bi-polar disorder is a constant battle. I know Abilify works for me because I have tried to get off it and lost complete control over my emotions. Went back on it and I was golden again.  I am on 5mg 2x daily. I am now 21 and better than I have ever been in the past. Only side effect is I like to eat a lot.", "He pulled out, but he cummed a bit in me. I took the Plan B 26 hours later, and took a pregnancy test two weeks later - - I&#039;m pregnant."], [27, 32, 5]]))