
import sys
import argparse
import logging
import os
import numpy as np
from metrics import cal_bleu, cal_dist, cal_rouge, cal_ppl

import transformers
from transformers import (BertTokenizer, BartForConditionalGeneration, \
    HfArgumentParser, DataCollatorForSeq2Seq,Seq2SeqTrainer, Seq2SeqTrainingArguments)
from transformers.trainer_utils import is_main_process

from arguments import DataTrainingArguments, ModelArguments
from utils import set_seed, postprocess_text, load_pk
import re

def preprocess_function(examples):

    queries = examples['query']
    reponses = examples['response']
    # [CLS] [SEP]
    model_inputs = tokenizer(queries, max_length=data_args.max_source_length, padding=padding, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(reponses, max_length=max_target_length, padding=padding, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def freeze_params(model):
    """Set requires_grad=False for each of model.parameters()"""
    for name, para in model.named_parameters():
        # print(name, para)
        if name != "persona_embdding.weight":
            para.requires_grad = False

def compute_metrics(eval_preds):

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    # print("decoded_preds is ", decoded_preds)
    # print("decoded_labels is ", decoded_labels)
    dist_score = cal_dist(decoded_preds)
    bleu_score = cal_bleu(decoded_preds, decoded_labels)
    rouge_score = cal_rouge(decoded_preds, decoded_labels)

    result = rouge_score
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result.update(dist_score)
    result.update(bleu_score)
    result = {k: round(v, 4) for k, v in result.items()} 
    return result

def de_prefix(sentence):
    # 实现对闲聊模块输出的句子，去除该句子复述用户query的前缀
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|？|。|、|；|‘|’|【|】|·|！| |…|（|）'
    flag = True
    for idx in re.finditer(pattern, sentence):
        idx = idx.span()[1]
        temp = sentence[idx:]
        flag = False
        break
    if flag:
        return sentence
    else:
        if idx >= len(sentence)-1:
            return sentence
        else:
            return temp

def eval_prefix_result(generate_corpus, reference_corpus):

    generate_corpus = [de_prefix(gen) for gen in generate_corpus]
    reference_corpus = [de_prefix(ref) for ref in reference_corpus]
    # print(generate_corpus)
    # print(reference_corpus)
    generate_corpus = [" ".join([s for s in gen]) for gen in generate_corpus]
    reference_corpus = [" ".join([s for s in ref]) for ref in reference_corpus]

    results = {}
    bleu_result = cal_bleu(generate_corpus, reference_corpus)
    dist_result = cal_dist(generate_corpus, reference_corpus)
    rouge_result = cal_rouge(generate_corpus, reference_corpus)

    results.update(bleu_result)
    results.update(dist_result)
    results.update(rouge_result)

    return results

def eval_keep_result(generate_corpus, reference_corpus):
    
    generate_corpus = [" ".join([s for s in gen]) for gen in generate_corpus]
    reference_corpus = [" ".join([s for s in ref]) for ref in reference_corpus]

    results = {}
    bleu_result = cal_bleu(generate_corpus, reference_corpus)
    dist_result = cal_dist(generate_corpus, reference_corpus)
    rouge_result = cal_rouge(generate_corpus, reference_corpus)

    results.update(bleu_result)
    results.update(dist_result)
    results.update(rouge_result)

    return results

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="fnlp/bart-base-chinese", type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--with_persona", action='store_true', default=False)
    parser.add_argument("--persona_embedding", action='store_true', default=False)
    parser.add_argument("--persona_traits", action='store_true', default=False)
    parser.add_argument("--cls_embedding", action='store_true', default=False)
    parser.add_argument("--history_post", action='store_true', default=False)
    parser.add_argument("--batch_size", default='96', type=str)
    parser.add_argument("--epoch", default='30', type=str)
    parser.add_argument("--data_dir", default="./dataset", type=str)
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_eval", action='store_true', default=False)
    parser.add_argument("--do_predict", action='store_true', default=False)
    parser.add_argument("--freeze_plm", action='store_true', default=False)

    args = parser.parse_args()
    arg_dict=args.__dict__

    logger = logging.getLogger(__name__)

    args=[
        '--model_name_or_path',arg_dict['model_path'],
        '--do_train={}'.format(arg_dict['do_train']),
        '--do_eval={}'.format(arg_dict['do_eval']),
        '--do_predict={}'.format(arg_dict['do_predict']),
        '--train_file', os.path.join(arg_dict['data_dir'], "train_data.pk"),
        '--validation_file', os.path.join(arg_dict['data_dir'],"dev_data.pk"),
        '--test_file', os.path.join(arg_dict['data_dir'],"test_data.pk"),
        '--output_dir', arg_dict["output_dir"],
        '--preprocessing_num_workers=3', 
        '--logging_steps=100',
        '--max_train_samples=400000',
        '--max_val_samples=10000',
        '--dataloader_num_workers=3',
        '--per_device_train_batch_size', arg_dict['batch_size'],
        '--per_device_eval_batch_size', arg_dict['batch_size'],
        '--overwrite_output_dir',
        '--max_source_length=64',
        '--val_max_target_length='+'64',
        '--predict_with_generate=1',
        '--seed', str(1000*1),
        '--num_train_epochs', arg_dict['epoch'],
        '--save_strategy','epoch',
        '--save_total_limit', '10',
        '--evaluation_strategy', 'epoch',
        '--learning_rate', str(arg_dict['lr']),
    ]

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)
    # model_args: ModelArguments; data_args: DataTrainingArguments; training_args: Seq2SeqTrainingArguments
    set_seed(training_args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    datasets = {}
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file

    for key in data_files:

        print("load no persona!")
        datasets[key] = load_pk(data_files[key])

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
    tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path)
    model = BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)

    model.config.max_length = data_args.val_max_target_length

    if arg_dict["freeze_plm"]:
        print("Freeze the pretrained model.......")
        freeze_params(model)

    column_names = datasets["train"].column_names
    max_target_length = data_args.val_max_target_length
    padding = False

    if training_args.do_train:
        train_dataset = datasets["train"]

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        print("process train dataset......................")
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    print("train dataset processed over")

    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            # batch_size=training_args.per_device_eval_batch_size,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        test_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            eval_dataset = test_dataset.select(range(data_args.max_predict_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            # batch_size=training_args.per_device_eval_batch_size,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        
    print("eval dataset processed over")
    
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    print("Initialize our Trainer")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:

        eval_dataloader = trainer.get_eval_dataloader(eval_dataset) # make list into tensor

        print("begin evaluating")
        eval_result = trainer.predict(eval_dataset, metric_key_prefix="predict") 
        metrics = eval_result.metrics
        
        loss_list = []
        for index, inputs in enumerate(eval_dataloader):

            loss, logits, labels = trainer.prediction_step(trainer.model, inputs, prediction_loss_only=True)
            loss_list.append(float(loss.cpu()))

        ppl_result = round(cal_ppl(loss_list),4)
        metrics.update({"ppl": ppl_result})
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        
        test_dataloader = trainer.get_eval_dataloader(test_dataset) # make list into tensor
        loss_list = []
        for index, inputs in enumerate(test_dataloader):
            loss, logits, labels = trainer.prediction_step(trainer.model, inputs, prediction_loss_only=True)
            loss_list.append(float(loss.cpu()))

        ppl_result = round(cal_ppl(loss_list),4)
        test_result = trainer.predict(test_dataset, metric_key_prefix="predict") 
        metrics = test_result.metrics
        metrics.update({"ppl": ppl_result})

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)