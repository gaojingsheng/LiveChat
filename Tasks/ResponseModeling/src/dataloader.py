
import torch
from torch.utils.data import Dataset

class LiveDataset(Dataset):
    def __init__(self, total_data, tokenizer, max_len, history_post_len=512):
        self.tokenizer = tokenizer
        self.data = total_data
        self.max_len = max_len
        self.history_post_len = history_post_len

    def __len__(self):
        return len(self.data)
    
    def tokenize(self, input_text, max_len):
        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return ids, mask, token_type_ids

    def __getitem__(self, index):
        queries = self.data[index]['query']
        reponses = self.data[index]['response']
        ids1, mask1, token_type_ids1 = self.tokenize(queries, self.max_len)
        ids2, mask2, token_type_ids2 = self.tokenize(reponses, self.max_len)
        if 'history_post' in self.data[index]:
            # history_post_id = self.data[real_index]['history_post']
            # print("history_post_id is ", history_post_id)
            history_posts = self.data[index]['history_post']
            ids3, mask3, token_type_ids3 = self.tokenize(history_posts, self.history_post_len)
            return {
                'ids': [torch.tensor(ids1, dtype=torch.long),torch.tensor(ids2, dtype=torch.long),torch.tensor(ids3, dtype=torch.long)],
                'mask': [torch.tensor(mask1, dtype=torch.long),torch.tensor(mask2, dtype=torch.long),torch.tensor(mask3, dtype=torch.long)],
                'token_type_ids': [torch.tensor(token_type_ids1, dtype=torch.long),torch.tensor(token_type_ids2, dtype=torch.long),torch.tensor(token_type_ids3, dtype=torch.long)],
            }
        else:
            return {
                'ids': [torch.tensor(ids1, dtype=torch.long),torch.tensor(ids2, dtype=torch.long)],
                'mask': [torch.tensor(mask1, dtype=torch.long),torch.tensor(mask2, dtype=torch.long)],
                'token_type_ids': [torch.tensor(token_type_ids1, dtype=torch.long),torch.tensor(token_type_ids2, dtype=torch.long)],
            }

