## How to run

Preprocessing the original data, and copy the data_path into --datadir

The file tree of datadir should be like:

```
.
+-- LiveChat
|   +-- train_data.pk
|   +-- test_data.pk
|   +-- dev_data.pk
```

## Train

```bash
python train.py --model_path fnlp/bart-base-chinese --output_dir ./outputs --data_dir ./LiveChat/ --do_train --do_eval 
```

## Test

Similar to the training process

```bash
python train.py --model_path <checkpoint_path> --output_dir ./outputs --data_dir ./LiveChat/ --do_eval 
```
