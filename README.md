# Lateral LoRA

This repository contains the implementation of **Lateral LoRA**

Lateral LoRA is a continual learning framework that enables knowledge transfer between previously learned LoRA adapters through trainable lateral connections.


---

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Pretrained Model

The experiments utilised the pre-trained **T5-Base** model from Hugging Face. When the following code is specified, the model is automatically downloaded upon the first execution:

```text
--model_name_or_path t5-base
```


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

- `FIRST_TASK` specifies the first task in the continual learning sequence. Example: Amazon → SST-2 uses "amazon", QQP → MRPC uses "qqp".
- `ADAPTER_ROOT` specifies the directory where adapters generated when continual learning are saved. Consistent with "output_dir" used in the script from Step 2.

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
| `ADAPTER_ROOT` | The directory used to save and load adapters required for the current task sequence. |
| `--task_config_dir` | The configuration directory for the current task (e.g., `configs/lateral_nl_configs/MNLI`, `configs/lateral_nl_configs/RTE`, `configs/lateral_sentiment_configs/amazon`). |
| `--output_dir` | The directory where the adapter for the current task is saved. For subsequent tasks, this directory is also used to load the adapter generated from the previous task. |

---

## Acknowledgement

This implementation is adapted from the experimental framework of the O-LoRA repository. It reuses the task-sequence training pipeline and benchmarking configurations while the proposed Lateral LoRA replaces the original O-LoRA learning mechanism.

Original O-LoRA repository:

https://github.com/cmnfriend/O-LoRA
