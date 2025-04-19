# %%
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
import math
import torch.nn.functional as F

# %%
def load_csv(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_reviews = train_df[train_df['Product Class'] != 'Else']['review'].tolist()
    train_ratings = train_df[train_df['Product Class'] != 'Else']['rating'].tolist()
    train_usefulcount = train_df[train_df['Product Class'] != 'Else']['usefulCount'].tolist()

    test_reviews = test_df[test_df['Product Class'] != 'Else']['review'].tolist()
    test_ratings = test_df[test_df['Product Class'] != 'Else']['rating'].tolist()
    test_usefulcount = test_df[test_df['Product Class'] != 'Else']['usefulCount'].tolist()

    return train_reviews,train_ratings, train_usefulcount, test_reviews, test_ratings, test_usefulcount

train_reviews, train_ratings, train_usefulcount, test_reviews, test_ratings, test_usefulcount = load_csv('./data/drugsComTrain_raw_addclass.csv', './data/drugsComTest_raw_addclass.csv')

# %%
print(len(torch.unique(torch.tensor(train_usefulcount))))
print(train_usefulcount)

# train_usefulcount = torch.tensor(train_usefulcount).float()
# print(((train_usefulcount - train_usefulcount.mean(dim=0, keepdim=True)) / train_usefulcount.std(dim=0, keepdim=True))[:100])

# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# %%
def tokenize(tokenizer, train_reviews, test_reviews):
    train_reviews_token = [tokenizer.encode_plus(
    text,
    truncation=True,
    add_special_tokens=True,
    max_length=128,            
    pad_to_max_length=True,  
    return_attention_mask=True,  
    return_tensors='pt',      
    ) for text in train_reviews]

    test_reviews_token = [tokenizer.encode_plus(
    text,
    truncation=True,
    add_special_tokens=True,
    max_length=128,            
    pad_to_max_length=True,  
    return_attention_mask=True,  
    return_tensors='pt',      
    ) for text in test_reviews]

    return train_reviews_token, test_reviews_token


train_reviews_token, test_reviews_token = tokenize(tokenizer, train_reviews, test_reviews)

# %%
class Review_Rating_Dataset(torch.utils.data.Dataset):
    def __init__(self, reviews_token, rating, usefulcount):
        self.review = reviews_token
        self.rating = rating
        self.usefulcount = torch.tensor(usefulcount)
 
    def __getitem__(self, idx):
        item = {k: v.squeeze(dim=0) for k, v in self.review[idx].items()}
        item["rating"] = torch.tensor(self.rating[idx] - 1)
        item['usefulcount'] = self.usefulcount[idx]
        return item
 
    def __len__(self):
        return len(self.rating)


train_dataset = Review_Rating_Dataset(train_reviews_token, train_ratings, train_usefulcount)
test_dataset = Review_Rating_Dataset(test_reviews_token, test_ratings, test_usefulcount)

# %%
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# %%
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

# %%
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

# %%
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    # total_error = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['rating'].to(device)
        counts = batch['usefulcount'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, count=counts)

        # loss = criterion(outputs, labels.float())
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # preds = torch.round(outputs)
        preds = torch.argmax(outputs, dim=-1)
        correct_predictions += torch.sum(preds == labels)
        # total_error += torch.sum(torch.abs(labels - outputs))
        total_loss += loss.item()
        
        # 更新进度条显示
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': torch.sum(preds == labels).item()/len(labels),
            # 'error': torch.mean(torch.abs(labels - outputs)).item()
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    # error = total_error.item() / len(dataloader.dataset)
    # return avg_loss, accuracy, error
    return avg_loss, accuracy


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    # total_error = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['rating'].to(device)
            counts = batch['usefulcount'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, count=counts)
            loss = criterion(outputs, labels)
            # loss = criterion(outputs, labels.float())
            
            preds = torch.argmax(outputs, dim=-1)
            correct_predictions += torch.sum(preds == labels)
            # total_error += torch.sum(torch.abs(labels - outputs))
            total_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': torch.sum(preds == labels).item()/len(labels),
                # 'error': torch.mean(torch.abs(labels - outputs)).item()
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    # error = total_error.item() / len(dataloader.dataset)
    # return avg_loss, accuracy, error
    return avg_loss, accuracy

# 4. 主训练循环
def train_and_evaluate(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    scheduler,
    criterion, 
    device, 
    epochs, 
    model_save_path,
    eval_every=1  # 每多少轮评估一次
):
    # best_val_error = 0.0
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        # 'train_error': [],
        'val_loss': [],
        'val_acc': [],
        # 'val_error': []
    }
    
    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # 训练阶段
        # train_loss, train_acc, train_error = train_epoch(
        #     model, train_loader, optimizer, criterion, device)
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        # history['train_error'].append(train_error)
        
        # print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Error: {train_error:.4f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # 验证阶段
        if epoch % eval_every == 0 and val_loader is not None:
            # val_loss, val_acc, val_error = eval_model(
            #     model, val_loader, criterion, device)
            val_loss, val_acc = eval_model(
                model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc.item())
            # history['val_error'].append(val_error)
            
            # print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Error: {val_error:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # if val_error > best_val_error:
            #     best_val_error = val_error
            #     torch.save(model.state_dict(), model_save_path)
            #     print(f"New best model saved to {model_save_path} with val_acc: {val_acc:.4f} | val_error: {val_error:.4f}")

            #     continue

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
                # print(f"New best model saved to {model_save_path} with val_acc: {val_acc:.4f} | val_error: {val_error:.4f}")
                print(f"New best model saved to {model_save_path} with val_acc: {val_acc:.4f}")
    
    return history

# %%
def main():
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BERT = BertModel.from_pretrained("bert-base-uncased")

    model = BertWithMLP_CrossAttention(BERT, hidden_size=768, mlp_hidden_size1=1024, mlp_hidden_size2=256, num_classes=10, num_count=2000)
    model.to(device)

    # 参数分组
    no_decay = ['bias', 'LayerNorm.weight']
    bert_params = []
    mlp_params = []

    for name, param in model.named_parameters():
        if 'mlp' in name or 'cross' in name:  # MLP层参数
            mlp_params.append((name, param))
        else:  # BERT参数
            bert_params.append((name, param))

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
        'lr': 2e-5},  # BERT主体较小学习率
        
        {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,
        'lr': 2e-5},
        
        {'params': [p for n, p in mlp_params if not any(nd in n for nd in no_decay)],
        'weight_decay': 0.01,
        'lr': 1e-4},  # MLP层较大学习率
        
        {'params': [p for n, p in mlp_params if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0,
        'lr': 1e-4}
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    epochs = 15

    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)  # 10%的warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    criterion = torch.nn.CrossEntropyLoss()
    model_save_path = "./rating_best_model.pth"
    
    # 创建保存目录
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 训练和验证
    history = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=epochs,
        model_save_path=model_save_path,
        eval_every=1  # 每轮都验证
    )
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    # print(f"Best validation error: {max(history['val_error']):.4f}")

if __name__ == "__main__":
    main()


