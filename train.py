from transformers import AutoTokenizer
from transformers import AutoModel
import torch
from dataset import MyDataset

# checkpoint = 'bert-base-uncased'
checkpoint = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
token = AutoTokenizer.from_pretrained(checkpoint)

def collate_fn(data):
    s1 = [i[0] for i in data]
    s2 = [i[1] for i in data]
    same = [i[2] for i in data]

    #编码
    data1 = token.batch_encode_plus(s1,
                                    truncation=True,
                                    padding=True,
                                    max_length=500,
                                    return_tensors='pt')

    data2 = token.batch_encode_plus(s2,
                                    truncation=True,
                                    padding=True,
                                    max_length=500,
                                    return_tensors='pt')

    same = torch.LongTensor(same)

    return data1, data2, same

#定义模型
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        #加载预训练模型
        self.pretrained = AutoModel.from_pretrained(checkpoint)

        #不训练,不需要计算梯度
        for param in self.pretrained.parameters():
            param.requires_grad_(False)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 2),
        )

    def get_feature(self, data):
        with torch.no_grad():
            #[b, L, 384]
            feature = self.pretrained(**data)['last_hidden_state']

        #[b, L]
        attention_mask = data['attention_mask']

        #pad位置的feature是0
        #[b, L, 384] * [b, L, 1] -> [b, L, 384]
        feature *= attention_mask.unsqueeze(dim=2)

        #所有词的feature求和
        #[b, L, 384] -> [b, 384]
        feature = feature.sum(dim=1)

        #求和后的feature除以句子的长度
        #[b, L] -> [b, 1]
        attention_mask = attention_mask.sum(dim=1, keepdim=True)

        #[b, 384] / [b, 1] -> [b, 384]
        feature /= attention_mask.clamp(min=1e-8)

        return feature

    def forward(self, data1, data2):
        feature1 = self.get_feature(data1)
        feature2 = self.get_feature(data2)

        feature = torch.cat([feature1, feature2], dim=1)

        return self.fc(feature)

#训练
def train():
    global model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # 多训练epoch 使用验证集验证
    model.train()
    for i, (data1, data2, same) in enumerate(loader):
        same = same.to(device)
        for k in data1.keys():
            data1[k] = data1[k].to(device)
            data2[k] = data2[k].to(device)
        pred = model(data1, data2)

        loss = criterion(pred, same)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if (i%1000):
            pred = pred.argmax(dim=1)
            accuracy = (pred == same).sum().item() / len(same)
            print(i, loss.item(), accuracy)

    torch.save(model.cpu(), 'models/optim_model.model')

if __name__ == 'main':
    dataset = MyDataset('train')
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=8,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)
    
    
    model = Model()
    train()
    
