import torch
from torch.utils.data import Dataset, DataLoader
class GPTDatasetV1(Dataset):
    def __init__(self, tokenizer, text, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        ids = tokenizer.encode(text)

        for i in range(0, len(ids) - max_length, stride):
            self.input_ids.append(torch.tensor(ids[i:i+max_length]))
            self.target_ids.append(torch.tensor(ids[i+1:i+max_length+1]))

        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self,idx):
            return self.input_ids[idx], self.target_ids[idx]