
import sys
import argparse
import logging
import numpy as np
import math
from transformers import (BertTokenizer, BartForConditionalGeneration, HfArgumentParser, DataCollatorForSeq2Seq, 
        Seq2SeqTrainingArguments) 
from datasets import Dataset

from arguments import DataTrainingArguments, ModelArguments
from utils import set_seed

from trainer.seq2seq_trainer import Seq2SeqTrainerNew 

class Predictor:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name",default="cpt",type=str)
        parser.add_argument("--model_path",default="./outputs/checkpoint",type=str)
        parser.add_argument("--lr",default=2e-5,type=float)
        parser.add_argument("--batch_size",default='32',type=str)
        parser.add_argument("--epoch",default='15',type=str)
        parser.add_argument("--data_dir",default="./dataset/",type=str)
        args = parser.parse_args()
        arg_dict=args.__dict__
        
        args=[
            '--model_name_or_path',arg_dict['model_path'],
            '--model_name',arg_dict['model_name'],
            '--output_dir',"./outputs",
            '--preprocessing_num_workers=4', 
            '--logging_steps=100',
            # '--max_train_samples=200',
            # '--max_val_samples=200',
            '--dataloader_num_workers=4',
            '--per_device_train_batch_size',arg_dict['batch_size'],
            '--per_device_eval_batch_size',arg_dict['batch_size'],
            '--overwrite_output_dir',
            '--max_source_length=64',
            '--val_max_target_length='+'64',
            '--predict_with_generate=1',
            '--seed',str(1000*1),
            '--num_train_epochs',arg_dict['epoch'],
            '--save_strategy','epoch',
            '--save_total_limit', '3',
            '--evaluation_strategy','epoch',
            '--learning_rate',str(arg_dict['lr']),
        ]

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
        model_args, self.data_args, self.training_args = parser.parse_args_into_dataclasses(args)
        set_seed(self.training_args.seed)

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        self.tokenizer=BertTokenizer.from_pretrained(model_args.model_name_or_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)


        self.model.config.max_length=self.data_args.val_max_target_length

        self.max_target_length = self.data_args.val_max_target_length
        self.padding=False
        
        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        print("Initialize our Trainer")
        self.trainer = Seq2SeqTrainerNew(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=None,
        )

    def preprocess_function(self, examples):
        queries = examples['query']
        reponses = examples['response']
        
        # inputs = ["[SEP]".join(b) + "[SEP]" + a for a, b in zip(documents, contexts)]
        model_inputs = self.tokenizer(queries, max_length=self.data_args.max_source_length, padding=self.padding, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(reponses, max_length=self.max_target_length, padding=self.padding, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def predict(self, query):
        results={'query':[],'response':[]}

        results['response'].append('')
        results['query'].append(query)
        results=Dataset.from_dict(results)
        test_dataset = results
        column_names = test_dataset.column_names

        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=column_names,
        )
            
        output, scores = self.trainer.predict(test_dataset, metric_key_prefix="predict") # 使用自定义trainer predict函数
        predictions = output.predictions
        test_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True,
        )
        test_preds = [pred.strip() for pred in test_preds]
        test_pred = "".join([x for x in test_preds[0] if x != ' '])
        return test_pred, scores[0].cpu().numpy()


if __name__ == "__main__":
    pred = Predictor()
    query_list = []
    results = []
    
    for query in query_list:
        test_pred, score = pred.predict(query)
        results.append([query, test_pred, math.exp(float(score))])
        
        print(query, test_pred)
