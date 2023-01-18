
import sys
import argparse
import logging
import os
import numpy as np
from transformers import BertTokenizer, BertConfig
from transformers.modeling_utils import PreTrainedModel, unwrap_model

from utils import set_seed, load_pk_persona, load_pk, compute_metrics_from_logits
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch import optim, cuda
from model import SingleBert
from dataloader import SingleBertDataset
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            label = data['label'].to(device, dtype=torch.long)
            assert label.size(0) == 10

            logits = model(ids, mask, token_type_ids) # 
            loss = F.cross_entropy(logits, label)

            logits_softmax = F.softmax(logits, dim=1)[:,1].unsqueeze(0)
            label_softmax = torch.tensor([9], dtype=torch.long).to(device)

            acc = (label_softmax.long() == logits_softmax.float().argmax(dim=1)).sum() / label_softmax.size(0)
            test_recall, test_MRR = compute_metrics_from_logits(logits_softmax, label_softmax)

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
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--writer_dir", default="./outputs/runs", type=str)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--add_id", action='store_true', default=False)
    parser.add_argument("--train_batch_size", default=40, type=int)
    parser.add_argument("--test_batch_size", default=10, type=int)
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

    datasets = {}
    data_files = {}

    if args.do_train:
        writer = SummaryWriter(log_dir=args.writer_dir, flush_secs=120)
        data_files["train"] = os.path.join(args.data_dir, "train_data.pk")
    if args.do_eval:
        data_files["validation"] = os.path.join(args.data_dir, "dev_data.pk")
    if args.do_test:
        data_files["test"] = os.path.join(args.data_dir, "test_data.pk")

    for key in data_files:
        if args.add_id:
            print("load ID")
            persona_id_path = "./personaId_list.json"
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

    model = SingleBert(config).to(device)

    if args.load_model_path != "":
        model.load_state_dict(torch.load(os.path.join(args.load_model_path, "pytorch_model.bin")))

    optimizer = optim.Adam(model.parameters(), lr = args.lr )
    
    if args.freeze_plm:
        print("Freeze the pretrained model.......")
        freeze_params(model)
    
    if args.do_train:
        train_dataset = datasets["train"]
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))

        print("train dataset length is {}".format(len(train_dataset)))
        training_set = SingleBertDataset(train_dataset, tokenizer, args.max_length)
        print("SiameseNetworkDataset length is {}".format(len(training_set)))
        training_loader = DataLoader(training_set, batch_size=args.train_batch_size, shuffle=False, num_workers=4)
    
        print("train dataset processed over")
        print("train dataset length is {}".format(len(training_loader)))

    if args.do_eval:
        eval_dataset = datasets["validation"]
        if args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(args.max_val_samples))

        testing_set = SingleBertDataset(eval_dataset, tokenizer, args.max_length)
        testing_loader = DataLoader(testing_set, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

        print("eval dataset processed over")

    # Training
    if args.do_train:
        print("Begin training")
        train_step = -1
        for epoch in range(args.epoch):
            model.train()
            for data in tqdm(training_loader, desc='Training'):

                train_step += 1
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                label = data['label'].to(device, dtype=torch.long)

                optimizer.zero_grad()
                logits = model(ids, mask, token_type_ids)
                loss = F.cross_entropy(logits, label)
                
                if train_step % 500==0:
                    train_acc = (label.long() == logits.float().argmax(dim=1)).sum() / label.size(0)
                    print(f'Step:{train_step}, Epoch:{epoch}, Loss:{loss.item()}, batch_acc:{train_acc}')
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
