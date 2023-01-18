
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

def load_pk(file_path):
    data = pk.load(open(file_path, 'rb'))
    results = {'streamer':[], 'audiences':[]}
    for sample in data:
        audiences = sample[1]
        streamer = sample[2]
        results['streamer'].append(streamer)
        results['audiences'].append(audiences)
    results = Ddataset.from_dict(results)
    return results

def load_pk_persona(file_path, personaId_list_path):
    id_list = json.load(open(personaId_list_path,"rb"))
    data = pk.load(open(file_path, 'rb'))
    results = {'streamer':[], 'audiences':[]}
    for sample in data:
        id_index = sample[0]
        audiences = sample[1]
        streamer = str(id_list[id_index]) + " [SEP] "+ sample[2]
        results['streamer'].append(streamer)
        results['audiences'].append(audiences)
    results = Ddataset.from_dict(results)
    return results


def load_retrive_history_post(file_path, history_post_path):
    history_post_list = json.load(open(history_post_path,"rb"))
    data = pk.load(open(file_path, 'rb'))

    results = {'history_post':[], 'streamer':[], 'audiences':[]}
    for sample in data:
        scratch_id = sample[0]
        audiences = sample[1]
        streamer = sample[2]
        results['history_post'].append(history_post_list[scratch_id])
        results['streamer'].append(streamer)
        results['audiences'].append(audiences)
    results = Ddataset.from_dict(results)
    return results


def load_retrive_history_post_and_id(file_path, history_post_path, persona_id_path):

    history_post_list = json.load(open(history_post_path,"rb"))
    id_list = json.load(open(persona_id_path,"rb"))
    data = pk.load(open(file_path, 'rb'))

    results = {'history_post':[], 'streamer':[], 'audiences':[]}
    for sample in data:
        scratch_id = sample[0]
        audiences = sample[1]
        streamer = str(id_list[scratch_id]) + " [SEP] "+ sample[2]
        
        results['history_post'].append(history_post_list[scratch_id])
        results['streamer'].append(streamer)
        results['audiences'].append(audiences)
    results = Ddataset.from_dict(results)
    return results


def compute_metrics(batch_x_emb, batch_y_emb):
    """
        recall@k for N candidates
        if batch_x_emb.dim() == 2:
            # batch_x_emb: (batch_size, emb_size)
            # batch_y_emb: (batch_size, emb_size)
        
        if batch_x_emb.dim() == 3:
            # batch_x_emb: (batch_size, batch_size, emb_size), the 1st dim is along examples and the 2nd dim is along candidates
            # batch_y_emb: (batch_size, emb_size)

    """
    batch_size = batch_x_emb.size(0)
    targets = torch.arange(batch_size, device=batch_x_emb.device)
    if batch_x_emb.dim() == 2:
        dot_products = batch_x_emb.mm(batch_y_emb.t()) # (batch_size, batch_size)
    elif batch_x_emb.dim() == 3:
        dot_products = torch.bmm(batch_x_emb, batch_y_emb.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1,2))[:, targets, targets]

    # dot_products: (batch_size, batch_size)
    sorted_indices = dot_products.sort(descending=True)[1]
    targets = np.arange(batch_size).tolist()
    recall_k = []
    if batch_size <= 10:
        ks = [1, max(1, round(batch_size*0.2)), max(1, round(batch_size*0.5))]
    elif batch_size <= 100:
        ks = [1, max(1, round(batch_size*0.1)), max(1, round(batch_size*0.5))]
    else:
        raise ValueError("batch_size: {0} is not proper".format(batch_size))
    for k in ks:
        # sorted_indices[:,:k]: (batch_size, k)
        num_ok = 0
        for tgt, topk in zip(targets, sorted_indices[:,:k].tolist()):
            if tgt in topk:
                num_ok += 1
        recall_k.append(num_ok/batch_size)
    
    # MRR
    MRR = 0
    for tgt, topk in zip(targets, sorted_indices.tolist()):
        rank = topk.index(tgt)+1
        MRR += 1/rank
    MRR = MRR/batch_size
    return recall_k, MRR

def compute_metrics_from_logits(logits, targets):
    """
        recall@k for N candidates
        
            logits: (batch_size, num_candidates)
            targets: (batch_size, )

    """
    batch_size, num_candidates = logits.shape
    
    sorted_indices = logits.sort(descending=True)[1]
    targets = targets.tolist()
    
    recall_k = []
    if num_candidates <= 10:
        ks = [1, max(1, round(num_candidates*0.2)), max(1, round(num_candidates*0.5))]
    elif num_candidates <= 100:
        ks = [1, max(1, round(num_candidates*0.1)), max(1, round(num_candidates*0.5))]
    else:
        raise ValueError("num_candidates: {0} is not proper".format(num_candidates))
    for k in ks:
        # sorted_indices[:,:k]: (batch_size, k)
        num_ok = 0
        for tgt, topk in zip(targets, sorted_indices[:,:k].tolist()):
            if tgt in topk:
                num_ok += 1
        recall_k.append(num_ok/batch_size)
    
    # MRR
    MRR = 0
    for tgt, topk in zip(targets, sorted_indices.tolist()):
        rank = topk.index(tgt)+1
        MRR += 1/rank
    MRR = MRR/batch_size
    return recall_k, MRR
