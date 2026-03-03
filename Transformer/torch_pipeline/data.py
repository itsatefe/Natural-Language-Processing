import torch
from torch.utils.data import Dataset, DataLoader

class TaaghcheDataset(Dataset):
    def __init__(self, tokenizer, comments, targets=None, label_list=None, max_len=128):
        self.comments = comments
        self.targets = targets
        self.has_target = targets is not None
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {label: i for i, label in enumerate(label_list)} if label_list else {}

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'comment': comment,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
        }
        if self.has_target:
            target = self.targets[idx]
            target = self.label_map.get(target, target)
            item['targets'] = torch.tensor(target, dtype=torch.long)
        return item

def create_data_loader(comments, targets, tokenizer, max_len, batch_size, label_list=None):
    ds = TaaghcheDataset(tokenizer, comments, targets, label_list, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=targets is not None)
