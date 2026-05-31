import torch
from torch.utils.data import Dataset

# GPTDatasetV1: converts a long text into many (input, target) training pairs
# for next-token prediction.
#
# - Tokenizes the full text into a single sequence of IDs.
# - Slides a window of size `max_length` across the IDs, stepping by `stride`.
# - For each window position, stores the input chunk (length max_length) and
#   the target chunk (same chunk shifted right by one token).
# - Number of pairs ≈ (len(ids) - max_length) // stride.

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
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