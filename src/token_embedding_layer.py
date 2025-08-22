import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):

    def __init__(self, text, tokenizer, context_size, stride):
        self.input_tokenIDs = []    
        self.target_tokenIDs = []
    
    ## Tokenize the data

        tokenized_data = tokenizer.encode(text)
        end = len(tokenized_data) - context_size

        for i in range(0, end, stride):

            x = tokenized_data[i:i + context_size]
            y = tokenized_data[i + 1: i + context_size + 1]

            self.input_tokenIDs.append(torch.tensor(x))
            self.target_tokenIDs.append(torch.tensor(y))



    # Return the length of the dataset    
    def __len__(self):
        return len(self.input_tokenIDs)
    
    # Returns a single row from the dataset
    def __getitem__(self, idx):
        return self.input_tokenIDs[idx], self.target_tokenIDs[idx]



def createGPTDataLoader(text, batch_size = 4, context_size = 256, stride = 128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, context_size, stride)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader

