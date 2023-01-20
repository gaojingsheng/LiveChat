## How to run

Preprocessing the original data, and copy the data_path into --datadir

The file tree of datadir should be like:

```
.
+-- dataset
|   +-- train_data.pk
|   +-- test_data.pk
|   +-- dev_data.pk
```

## Train

```bash
python train.py --history_post --add_id --train_from_scratch --do_train --do_eval --output_dir ./outputs_history_ID --writer_dir ./outputs_history_ID/runs
```

remove the --history and --add_id mean remove the input of text profiles and basic profiles

## Test

Similar to the training process

```bash
python train.py --history_post --add_id --do_eval --load_model_path ./outputs_history_ID/epoch_i --output_dir ./outputs_history_ID --writer_dir ./outputs_history_ID/runs 
```
