from transformers import AutoTokenizer
from transformers import AutoModel
import torch
from datasets import load_dataset

path = r".\dataset\train.csv"
# path =  r""

# 数据集定义
class MyDataset(torch.utils.data.Dataset):
    # 划分train,test,valid
    def __init__(self, split):
        dataset = load_dataset("csv", data_files=[path])['train']
        self.dataset = dataset.train_test_split(test_size=0.1)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        _, category, s1, s2, same = self.dataset[i].values()
        return s1, s2, same

