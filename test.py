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


def test():
    model.eval()
    correct = 0
    total = 0

    loader_test = torch.utils.data.DataLoader(dataset=MyDataset('test'),
                                              batch_size=16,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=False)

    for i, (data1, data2, label) in enumerate(loader_test):
        with torch.no_grad():
            pred = model(data1, data2)

        pred = pred.argmax(dim=1)

        correct += (pred == label).sum().item()
        total += len(label)

        print(i)

    print(correct / total)

#使用原始数据集data.xslx 
def call():
    model.eval()
    
    loader = torch.utils.data.DataLoader(dataset=MyDataset('train'),
                                              batch_size=16,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=False)
    
    for i, (data1, data2, label) in enumerate(loader):
        with torch.no_grad():
            pred = model(data1, data2)

        pred = pred.argmax(dim=1)



if __name__ == 'main':
    model = torch.load('models/optim_model.model')
    test()