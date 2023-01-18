
import random
import numpy as np
import torch
import pickle as pk
from datasets import Dataset as Ddataset
import pandas as pd
import json



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    while '' in preds:
        idx=preds.index('')
        preds[idx]='。'
    return preds, labels

def load_pk(file_path):
    data = pk.load(open(file_path, 'rb'))
    results = {'query':[], 'response':[]}
    for sample in data:
        query = sample[1]
        response = sample[2]
        results['query'].append(query)
        results['response'].append(response)
    results = Ddataset.from_dict(results)
    return results

