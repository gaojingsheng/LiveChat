
import sys
import argparse
import logging
import os
import numpy as np
from transformers import BertTokenizer, BertConfig
from transformers.modeling_utils import PreTrainedModel, unwrap_model

from utils import set_seed, load_retrive_history_post, load_retrive_history_post_and_id, load_pk_persona, load_pk, compute_metrics, compute_metrics_from_logits
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch import optim, cuda
from model import ThreeBert
from dataloader import LiveDataset
import torch.nn.functional as F
from sklearn import metrics
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0.4):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        cos_sim = F.cosine_similarity(output1, output2)

        loss_cos_con = torch.mean((1-label) * torch.div(torch.pow((1.0-cos_sim), 2), 4) + \
                                    (label) * torch.pow(cos_sim * torch.lt(cos_sim, self.margin), 2))
        return loss_cos_con

class BatchCosineContrastiveLoss(nn.Module):
    def __init__(self, ):
        super(BatchCosineContrastiveLoss, self).__init__()

    def forward(self, output1, output2):
        batch_size = output2.size(0)
        y_true = torch.arange(batch_size, dtype=torch.long, device=output1.device)

        cos_sim_matrix = F.cosine_similarity(output1.unsqueeze(1), output2.unsqueeze(0), dim=2)
        cos_sim_matrix = cos_sim_matrix - torch.eye(batch_size, device="cuda") * 1e-12

        loss = F.cross_entropy(cos_sim_matrix, y_true)

        return loss

def DotProDuctLoss(batch_x_emb, batch_y_emb):
    """
        if batch_x_emb.dim() == 2:
            # batch_x_emb: (batch_size, emb_size)
            # batch_y_emb: (batch_size, emb_size)
        
        if batch_x_emb.dim() == 3:
            # batch_x_emb: (batch_size, batch_size, emb_size), the 1st dim is along examples and the 2nd dim is along candidates
            # batch_y_emb: (batch_size, emb_size)
    """
    batch_size = batch_x_emb.size(0)
    targets = torch.tensor([batch_size-1 for i in range(batch_size)], device=batch_x_emb.device)

    if batch_x_emb.dim() == 2:
        dot_products = batch_x_emb.mm(batch_y_emb.t())
    elif batch_x_emb.dim() == 3:
        dot_products = torch.bmm(batch_x_emb, batch_y_emb.unsqueeze(0).repeat(batch_size, 1, 1).transpose(1,2))[:, targets, targets] # (batch_size, batch_size)
    
    # dot_products: [batch, batch]
    log_prob = F.log_softmax(dot_products, dim=1)
    loss = F.nll_loss(log_prob, targets)
    nb_ok = (log_prob.max(dim=1)[1] == targets).float().sum()

    return loss, nb_ok/batch_size
 

class Similarity(nn.Module):
    def __init__(self,):
        super(Similarity, self).__init__()

    def forward(self, output1, output2):
        if len(output1.size()) == 1 and len(output2.size()) == 1:
            output1 = output1.unsqueeze(0)
            output2 = output2.unsqueeze(0)
            cos_sim = F.cosine_similarity(output1, output2)
        else:
            cos_sim = F.cosine_similarity(output1, output2)

        return float(cos_sim)

def freeze_params(model):
    """Set requires_grad=False for each of model.parameters()"""
    for _, para in model.named_parameters():
        para.requires_grad = False

def validation(model, test_dataloader):
    model.eval()

    total_acc = []
    total_loss = []
    total_recall = []
    total_MRR = []

    with torch.no_grad():
        for _, data in enumerate(tqdm(test_dataloader, desc='Evaluating')):
            ids, mask, token_type_ids = data['ids'], data['mask'], data['token_type_ids']
            if args.history_post:
                ids = [ids[0].to(device, dtype = torch.long), ids[1].to(device, dtype = torch.long), ids[2].to(device, dtype = torch.long)]
                mask = [mask[0].to(device, dtype = torch.long), mask[1].to(device, dtype = torch.long), mask[2].to(device, dtype = torch.long)]
                token_type_ids = [token_type_ids[0].to(device, dtype = torch.long), token_type_ids[1].to(device, dtype = torch.long), token_type_ids[2].to(device, dtype = torch.long)]
            else:
                ids = [ids[0].to(device, dtype=torch.long), ids[1].to(device, dtype=torch.long)]
                mask = [mask[0].to(device, dtype=torch.long), mask[1].to(device, dtype=torch.long)]
                token_type_ids = [token_type_ids[0].to(device, dtype=torch.long), token_type_ids[1].to(device, dtype=torch.long)]

            if args.apply_interaction == True:
                logits, targets = model(ids, mask, token_type_ids, args.apply_interaction)
                loss = F.cross_entropy(logits, targets)
                acc = (targets.long() == logits.float().argmax(dim=1)).sum() / targets.size(0)
                test_recall, test_MRR = compute_metrics_from_logits(logits, targets)

            else:
                output1, output2 = model(ids, mask, token_type_ids, args.apply_interaction)
                # output1, output2 = model(ids, mask, token_type_ids)
                loss, acc = DotProDuctLoss(output1, output2)
                test_recall, test_MRR = compute_metrics(output1, output2)

            total_loss.append(float(loss))
            total_acc.append(float(acc))
            total_recall.append(test_recall)
            total_MRR.append(test_MRR)

    return np.mean(total_loss), np.mean(total_acc), np.mean(total_recall, axis=0), np.mean(total_MRR)

def save_model(args, output_dir, model, tokenizer=None):

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model checkpoint to {output_dir}")

    if not isinstance(model, PreTrainedModel):
        if isinstance(unwrap_model(model), PreTrainedModel):
            if state_dict is None:
                state_dict = model.state_dict()
            unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            if state_dict is None:
                state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    else:
        model.save_pretrained(output_dir, state_dict=model.state_dict())
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, "training_args.bin"))


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="bert-base-chinese", type=str)
    parser.add_argument("--load_model_path", default="", type=str)
    parser.add_argument("--history_post_path", default="", type=str)
    parser.add_argument("--persona_id_path", default="", type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--writer_dir", default="./outputs/runs", type=str)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--history_post", action='store_true', default=False)
    parser.add_argument("--add_id", action='store_true', default=False)
    parser.add_argument("--apply_interaction", action='store_true', default=False)
    parser.add_argument("--train_from_scratch", action='store_true', default=False)
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--max_length", default=64, type=int)
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--logging_steps", default=100, type=int)
    parser.add_argument("--max_train_samples", default=400000, type=int)
    parser.add_argument("--max_val_samples", default=10000, type=int)
    parser.add_argument("--seed", default=20, type=int)
    parser.add_argument("--data_dir", default="./dataset", type=str)
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--do_test", action='store_true', default=False)
    parser.add_argument("--freeze_plm", action='store_true', default=False)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    datasets = {}
    data_files = {}

    args.add_id = True
    print(args)
    if args.do_train:
        writer = SummaryWriter(log_dir=args.writer_dir, flush_secs=120)

    if args.do_train:
        data_files["train"] = os.path.join(args.data_dir, "train_data.pk")
    if args.do_eval:
        data_files["validation"] = os.path.join(args.data_dir, "dev_data.pk")
    if args.do_test:
        data_files["test"] = os.path.join(args.data_dir, "test_data.pk")

    for key in data_files:
        if args.history_post and args.add_id:
            print("load history_post and ID")
            history_post_path = args.history_post_path
            persona_id_path = args.persona_id_path
            datasets[key] = load_retrive_history_post_and_id(data_files[key], history_post_path, persona_id_path)

        elif args.history_post and not args.add_id:
            print("load history_post")
            history_post_path = args.history_post_path
            datasets[key] = load_retrive_history_post(data_files[key], history_post_path)

        elif args.add_id and not args.history_post:
            print("load ID")
            persona_id_path = args.persona_id_path
            datasets[key] = load_pk_persona(data_files[key], persona_id_path)
        else:
            print("load no persona!")
            datasets[key] = load_pk(data_files[key])

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    device = 'cuda' if cuda.is_available() else 'cpu'
    config = BertConfig.from_pretrained(args.model_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    if args.train_from_scratch:
        model = ThreeBert(config, train_from_scratch=True).to(device)
    else:
        model = ThreeBert(config, train_from_scratch=False).to(device)
    if args.load_model_path != "":
        model.load_state_dict(torch.load(os.path.join(args.load_model_path, "pytorch_model.bin")))
    criterion = BatchCosineContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr = args.lr )
    
    if args.freeze_plm:
        print("Freeze the pretrained model.......")
        freeze_params(model)

    if args.do_train:
        train_dataset = datasets["train"]
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))

        print("train dataset length is {}".format(len(train_dataset)))
        training_set = LiveDataset(train_dataset, tokenizer, args.max_length)
        print("LiveDataset length is {}".format(len(training_set)))
        training_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
        print("train dataset length is {}".format(len(training_loader)))

    if args.do_eval:
        eval_dataset = datasets["validation"]
        if args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(args.max_val_samples))
        testing_set = LiveDataset(eval_dataset, tokenizer, args.max_length)
        testing_loader = DataLoader(testing_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

        print("eval dataset processed over")
    # Training
    if args.do_train:
        print("Begin training")
        train_step = -1
        for epoch in range(args.epoch):
            model.train()
            for _, data in enumerate(tqdm(training_loader, desc='Training')):
                # tqdm slow is for it didn't display until next epoch
                train_step += 1
                ids, mask, token_type_ids= data['ids'], data['mask'], data['token_type_ids']
                if args.history_post:
                    ids = [ids[0].to(device, dtype=torch.long), ids[1].to(device, dtype=torch.long), ids[2].to(device, dtype=torch.long)]
                    mask = [mask[0].to(device, dtype=torch.long), mask[1].to(device, dtype=torch.long), mask[2].to(device, dtype=torch.long)]
                    token_type_ids = [token_type_ids[0].to(device, dtype=torch.long), token_type_ids[1].to(device, dtype=torch.long), token_type_ids[2].to(device, dtype=torch.long)]
                else:
                    ids = [ids[0].to(device, dtype=torch.long), ids[1].to(device, dtype=torch.long)]
                    mask = [mask[0].to(device, dtype=torch.long), mask[1].to(device, dtype=torch.long)]
                    token_type_ids = [token_type_ids[0].to(device, dtype=torch.long), token_type_ids[1].to(device, dtype=torch.long)]
                optimizer.zero_grad()
                
                if args.apply_interaction == True:
                    logits, targets = model(ids, mask, token_type_ids, args.apply_interaction)
                    loss = F.cross_entropy(logits, targets)
                    acc = (targets.long() == logits.float().argmax(dim=1)).sum() / targets.size(0)
                    if train_step % 500==0:
                        train_recall, train_MRR = compute_metrics_from_logits(logits, targets)
                        print(f'Step:{train_step}, Epoch:{epoch}, Loss:{loss.item()}, batch_acc:{acc}, batch_recall:{train_recall}, batch_MRR:{train_MRR}')
                else:
                    output1, output2 = model(ids, mask, token_type_ids, args.apply_interaction)
                    loss, acc = DotProDuctLoss(output1, output2)
                    if train_step % 500==0:
                        train_recall, train_MRR = compute_metrics(output1, output2)
                        print(f'Step:{train_step}, Epoch:{epoch}, Loss:{loss.item()}, batch_acc:{acc}, batch_recall:{train_recall}, batch_MRR:{train_MRR}')
                writer.add_scalar('Loss/train', loss.item(), train_step)
                loss.backward()
                optimizer.step() 
            test_loss, test_acc, test_recall, test_mrr = validation(model, testing_loader)
            print(f'Test Epoch:{epoch}, Test loss:{test_loss}, Test accuracy:{test_acc}, Test recall:{test_recall}, Test MRR:{test_mrr}')
            writer.add_scalar('Test loss', test_loss, epoch)
            writer.add_scalar('Test accuracy', test_acc, epoch)
            writer.add_scalar('Test MRR', test_mrr, epoch)

            save_model_path = os.path.join(args.output_dir,"epoch_{}".format(epoch))
            save_model(args, save_model_path, model, tokenizer)
            
    if args.do_eval:
        test_loss, test_acc, test_recall, test_mrr = validation(model, testing_loader)
        print(f'Test checkpoint{args.load_model_path}, Test loss:{test_loss}, Test accuracy:{test_acc}, Test recall:{test_recall}, Test MRR:{test_mrr}')
