# Lateral LoRA

This repository contains the implementation of **Lateral LoRA**

## Overview

Lateral LoRA is a continual learning framework that enables knowledge transfer between previously learned LoRA adapters through trainable lateral connections.

This implementation is adapted from the experimental framework of the O-LoRA repository. The original task-sequential training pipeline and benchmark configuration are retained, while the learning mechanism has been replaced by the proposed Lateral LoRA method.

---

## Requirements

- Python 3.10 or later
- CUDA-enabled GPU (recommended)

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Pretrained Model

The experiments use the pretrained **T5-Base** model from Hugging Face.

No manual download is required. The model will be downloaded automatically during the first execution when

```text
--model_name_or_path t5-base
```

is specified.

---

# Running Experiments

## Step 1. Configure the experiment

Open

```
src/run_uie_lora_laterallora.py
```

and modify the following variables:

```python
FIRST_TASK = "mnli"

ADAPTER_ROOT = r"C:\lateral_nli_out"
```

where

- `FIRST_TASK` specifies the first task in the continual learning sequence. Example: Amazon → SST-2 uses "amazon", QQP → MRPC uses "qqp"
- `ADAPTER_ROOT` specifies the directory where adapters generated during continual learning are saved. Consistent with the run script in Step 2.


---

## Step 2. Train the first task

Example (MNLI)

```bash
python src/run_uie_lora_laterallora.py ^
 --do_train ^
 --do_predict ^
 --predict_with_generate ^
 --model_name_or_path t5-base ^
 --data_dir CL_Benchmark ^
 --task_config_dir configs/lateral_nl_configs/MNLI ^
 --instruction_file configs/instruction_config.json ^
 --instruction_strategy single ^
 --output_dir C:\lateral_nli_out\1-mnli ^
 --per_device_train_batch_size 2 ^
 --per_device_eval_batch_size 4 ^
 --gradient_accumulation_steps 4 ^
 --learning_rate 1e-3 ^
 --num_train_epochs 3 ^
 --run_name lateral_mnli_task1 ^
 --max_source_length 256 ^
 --max_target_length 50 ^
 --generation_max_length 20 ^
 --add_task_name True ^
 --add_dataset_name True ^
 --overwrite_output_dir ^
 --overwrite_cache ^
 --logging_steps 10 ^
 --evaluation_strategy no ^
 --save_strategy no ^
 --report_to none
 --seed 42
```

---

## Step 3. Train the next task

Example (RTE)

```bash
python src/run_uie_lora_laterallora.py ^
 --do_train ^
 --do_predict ^
 --predict_with_generate ^
 --model_name_or_path C:\lateral_nli_out\1-mnli\adapter ^
 --data_dir CL_Benchmark ^
 --task_config_dir configs/lateral_nl_configs/RTE ^
 --instruction_file configs/instruction_config.json ^
 --instruction_strategy single ^
 --output_dir C:\lateral_nli_out\2-rte ^
 ...
```

---

## Changing to another task

When changing to another continual learning task sequence, update the following accordingly:

| Item | Description |
|------|-------------|
| `FIRST_TASK` | The name of the first task in `run_uie_lora_laterallora.py`. |
| `ADAPTER_ROOT` | Directory used to save and load adapters for the current task sequence. |
| `--task_config_dir` | Configuration file of the current task (e.g., `configs/lateral_nl_configs/MNLI`, `configs/lateral_nl_configs/RTE`, `configs/lateral_sentiment_configs/amazon`). |
| `--output_dir` | Directory where the adapter for the current task is saved. For subsequent tasks, this directory is also used to load the adapter generated from the previous task. |

For example, for the Amazon → SST-2 sequence:

- `FIRST_TASK = "amazon"`
- `ADAPTER_ROOT = C:\lateral_sentiment_out`
- Task 1:
  - `--task_config_dir configs/lateral_sentiment_configs/amazon`
  - `--output_dir C:\lateral_sentiment_out\1-amazon`
- Task 2:
  - `--model_name_or_path C:\lateral_sentiment_out\1-amazon\adapter`
  - `--task_config_dir configs/lateral_sentiment_configs/SST2`
  - `--output_dir C:\lateral_sentiment_out\2-sst2`

---

## Acknowledgement

This implementation is adapted from the experimental framework of the O-LoRA repository. The task-sequential training pipeline and benchmark configuration are reused, while the proposed Lateral LoRA replaces the original O-LoRA learning mechanism.

Original O-LoRA repository:

https://github.com/cmnfriend/O-LoRA