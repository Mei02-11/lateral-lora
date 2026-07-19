#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add

from uie_collator import DataCollatorForUIE
from uie_dataset_lora import gen_cache_path

from uie_trainer_lora_pure import UIETrainer, DenserEvalCallback, skip_instructions
from compute_metrics import compute_metrics, compute_grouped_metrics
from model.llama import LlamaForCausalLM_with_lossmask

##### Lateral lora
import torch
from torch.utils.data import DataLoader

print("\n==========  LATERAL LORA  ==========\n")

##### User Configuration
# Modify these values before running a new task sequence
FIRST_TASK = "mnli"
ADAPTER_ROOT = r"C:\lateral_nli_out"

##### Multiple adapter
base_model_path = "t5-base"
def get_real_adapter_name(model, adapter_name):

    if adapter_name in model.peft_config:
        return adapter_name

    if "default" in model.peft_config:
        return "default"

    return adapter_name

# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    # added for AutoCL
    lora_dim: Optional[int] = field(
        default=8,
        metadata={
            "help": "Intrinsic dimension of the latent space."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    instruction_strategy: Optional[str] = field(
        default='single', metadata={
            "help": "How many different instructions to use? Support 'single' and 'multiple' mode."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )


@dataclass
class UIETrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})
    ########## Pure Lora ########## 

##### Stage 7.7
##### Dataset-wide averaged task embedding

def build_task_embedding(
    model,
    tokenizer,
    dataset,
    device,
    max_batches=10
):
    loader=DataLoader(dataset,batch_size=4,shuffle=False)
    collected_embeddings=[]

    model.eval()

    with torch.no_grad():
        for batch_idx,batch in enumerate(loader):
            if batch_idx>=max_batches:
                break

            ##### Extract source texts
            texts=batch["Instance"]["sentence"]

            ##### Tokenize
            inputs=tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            ##### Encoder hidden states
            base_encoder=(
                model.base_model.model.encoder
                if hasattr(model.base_model,"model")
                else model.base_model.encoder
            )
            encoder_outputs=base_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True
            )

            ##### Mean pooled embeddings
            embedding=encoder_outputs.last_hidden_state.mean(dim=1)
            collected_embeddings.append(embedding.mean(dim=0))

    ##### Dataset-wide centroid
    final_embedding=torch.stack(collected_embeddings).mean(dim=0)

    ##### Normalize embedding
    final_embedding=final_embedding/final_embedding.norm(p=2)

    return final_embedding

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UIETrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ##### Dynamic current adapter name
    current_adapter_name=os.path.basename(
        os.path.normpath(data_args.task_config_dir)
    ).replace(".json","").lower()

    current_adapter=current_adapter_name

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_cache_dir = gen_cache_path(training_args.output_dir, data_args)

    # Get the UIE dataset
    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "uie_dataset_lora.py"),
        data_dir=data_args.data_dir,
        task_config_dir=data_args.task_config_dir,
        instruction_file=data_args.instruction_file,
        instruction_strategy=data_args.instruction_strategy,
        cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        num_examples=data_args.num_examples
    )
    raw_datasets.cleanup_cache_files()

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if 'adapter' in model_args.model_name_or_path: # load lora-config
        #config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        ##### Multiple adapter
        config = AutoConfig.from_pretrained(base_model_path)
        if 'llama' in model_args.model_name_or_path.lower():
            tokenizer = transformers.LlamaTokenizer.from_pretrained(config.base_model_name_or_path)
            config.bos_token_id = 1
            config.eos_token_id = 2
            config.pad_token_id = 1
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            tokenizer.pad_token_id = 1
        else:
            ##### Multiple adapter
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    elif 'llama' in model_args.model_name_or_path.lower():
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.bos_token_id = 1
        config.eos_token_id = 2
        config.pad_token_id = 1
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir = model_args.cache_dir,
            use_fast = model_args.use_fast_tokenizer,
            revision = model_args.model_revision,
            use_auth_token = True if model_args.use_auth_token else None,
        )
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1
    else: # load original config
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if 'llama' in model_args.model_name_or_path.lower():  # add llama
        model_class = LlamaForCausalLM_with_lossmask
        tokenizer.padding_side = 'left'
    else: 
        model_class = AutoModelForSeq2SeqLM

    ##### Multiple adapter
    ##### AUTO MULTI-ADAPTER STANDARD LORA

    ##### Current task name
    current_adapter=os.path.basename(data_args.task_config_dir).lower()
    print(f"\n[CURRENT ADAPTER] {current_adapter}")

    ##### Adapter root
    adapter_root=ADAPTER_ROOT

    ##### Find previous adapters
    available_adapters=[]

    if os.path.exists(adapter_root):
        for folder in os.listdir(adapter_root):
            adapter_path=os.path.join(adapter_root,folder,"adapter")

            if os.path.exists(adapter_path):
                ##### Case 1: single-adapter structure (task1)
                single_config=os.path.join(adapter_path,"adapter_config.json")

                if os.path.exists(single_config):
                    adapter_name=folder.split("-")[-1]

                    if adapter_name not in [x[0] for x in available_adapters]:
                        available_adapters.append((adapter_name,adapter_path))

                ##### Case 2: multi-adapter structure (task2+)
                else:
                    for adapter_name in os.listdir(adapter_path):
                        sub_adapter_path=os.path.join(adapter_path,adapter_name)
                        config_path=os.path.join(sub_adapter_path,"adapter_config.json")

                        if os.path.exists(config_path):
                            if adapter_name not in [x[0] for x in available_adapters]:
                                available_adapters.append((adapter_name,sub_adapter_path))

    print(f"\n[FOUND ADAPTERS] {[x[0] for x in available_adapters]}")

    ##### Build base model
    model=model_class.from_pretrained(
        ##### Multiple adapter
        base_model_path if 'adapter' in model_args.model_name_or_path else model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    ##### PEFT CONFIG
    peft_config=LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=model_args.lora_dim,
        lora_alpha=32,
        lora_dropout=0.1
    )

    ##### TASK 1
    ##### No adapters yet
    if len(available_adapters)==0:
        print("\n[CREATING FIRST ADAPTER]")
        model=get_peft_model(model,peft_config)

    ##### TASK 2+
    ##### Load old adapters
    else:
        print("\n[LOADING OLD ADAPTERS]")

        ##### Load first adapter
        first_adapter_name,first_adapter_path=available_adapters[0]
        model=PeftModel.from_pretrained(
            model,
            first_adapter_path,
            adapter_name=first_adapter_name
        )
        print(f"[LOADED] {first_adapter_name}")

        ##### Load remaining adapters
        for adapter_name,adapter_path in available_adapters[1:]:
            model.load_adapter(adapter_path,adapter_name=adapter_name)
            print(f"[LOADED] {adapter_name}")

        ##### Add current adapter if missing
        if current_adapter not in model.peft_config:
            model.add_adapter(current_adapter,peft_config)
            print(f"[ADDED NEW ADAPTER] {current_adapter}")

    ##### Activate current adapter
    real_adapter_name=get_real_adapter_name(model,current_adapter)
    model.set_adapter(real_adapter_name)
    print(f"\n[ACTIVE ADAPTER] {current_adapter}")

    ##### Train ONLY current adapter
    real_adapter_name=get_real_adapter_name(model,current_adapter)

    for name,param in model.named_parameters():
        if f".{real_adapter_name}." in name:
            param.requires_grad=True
        else:
            param.requires_grad=False

    ##### DEBUG
    model.print_trainable_parameters()

    trainable=0
    total=0

    for name,param in model.named_parameters():
        total+=param.numel()
        if param.requires_grad:
            trainable+=param.numel()

    print(f"\nTrainable params: {trainable}")
    print(f"Total params: {total}")
    print(f"Trainable %: {100*trainable/total:.6f}%")
    
    model.resize_token_embeddings(len(tokenizer))

    if 'llama' in model_args.model_name_or_path.lower():
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = 1
        
    # fix lora_A/B (bases of previous LoRA parameters, loaded in "load_adapter"[peft_momdel.py])
    # fine-tune loranew_A/B (initialized in "update_layer"[lora.py])
    # optional: lora_A/B is trainable but should not move too far from lorapre_A/B
    # (constrained in "training_step"[uie_trainer_lora.py])

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        
        ##### Small validation subset for routing evaluation
        routing_eval_raw=train_dataset.select(range(min(64,len(train_dataset))))
        routing_eval_dataset=[]

        for sample in routing_eval_raw:
            ##### Input text
            input_text=sample["Instance"]["sentence"]

            ##### Target text
            target_text=sample["Instance"]["label"]

            ##### Tokenize input
            model_inputs=tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=data_args.max_source_length
            )

            ##### Tokenize target
            labels=tokenizer(
                target_text,
                truncation=True,
                padding="max_length",
                max_length=data_args.max_target_length
            )

            routing_eval_dataset.append({
                "input_ids":model_inputs["input_ids"],
                "attention_mask":model_inputs["attention_mask"],
                "labels":labels["input_ids"]
            })

    ##### Stage 7.7
    ##### Build dataset-wide semantic embedding
    current_embedding=build_task_embedding(
        model=model,
        tokenizer=tokenizer,
        dataset=train_dataset,
        device=model.device
    )

    ##### Store current task embedding
    model.current_task_embedding=current_embedding
    print(f"\n[CURRENT TASK EMBEDDING BUILT] {current_adapter_name}")

    ##### Load old adapter embeddings
    embedding_dir=os.path.join(
        "logs_and_outputs",
        "lateral-lora",
        "shared_embeddings"
    )

    os.makedirs(embedding_dir,exist_ok=True)

    for adapter_name in model.peft_config.keys():
        if adapter_name!=current_adapter_name:
            embedding_path=os.path.join(
                embedding_dir,
                f"{adapter_name}.pt"
            )

            if os.path.exists(embedding_path):
                embedding=torch.load(embedding_path).to(model.device)
                model.set_adapter_embedding(adapter_name,embedding)
                print(f"\n[EMBEDDING LOADED] {adapter_name}")

    ##### Save current embedding
    current_embedding_path=os.path.join(
        embedding_dir,
        f"{current_adapter_name}.pt"
    )

    torch.save(current_embedding.cpu(),current_embedding_path)
    print(f"\n[EMBEDDING SAVED] {current_adapter_name}")

    ##### Stage 7
    ##### Dynamic Top-K routing
    if hasattr(model,"select_topk_adapters"):
        selected_adapters,adapter_scores=model.select_topk_adapters(
            current_adapter=current_adapter_name,
            eval_dataset=routing_eval_dataset,
            k=3,
            min_improvement=-1
        )

    selected_adapters=[]

    ##### Propagate similarities to LoRA layers - did not used anymore
    if hasattr(model,"set_old_adapters"):
        model.set_old_adapters(selected_adapters)

    model.selected_old_adapters=selected_adapters

    ##### Save routing for current task
    if not hasattr(model,"task_routing_history"):
        model.task_routing_history={}

    ##### Preserve old routing history
    routing_save_dir=os.path.join(
        adapter_root,
        f"uie-lora-{current_adapter_name}",
        "adapter"
    )

    os.makedirs(routing_save_dir,exist_ok=True)
    routing_path=os.path.join(routing_save_dir,"routing.json")

    existing_history={}
    if os.path.exists(routing_path):
        with open(routing_path,"r") as f:
            existing_history=json.load(f)

    model.task_routing_history.update(existing_history)

    ##### Save current task routing
    model.task_routing_history[current_adapter_name]={
        "selected":selected_adapters,
        "scores":{
            k:float(v)
            for k,v in adapter_scores.items()
        }
    }

    with open(routing_path,"w") as f:
        json.dump(
            model.task_routing_history,
            f,
            indent=2
        )

    print(f"\n[ROUTING FILE SAVED] {routing_path}")

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForUIE(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file
    )
    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    # Metric
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=dataset["Task"])
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                      groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result

    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainer = UIETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge_metrics,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )

    ##### Stage 5
    ##### Freeze old adapters AFTER trainer init
    for old_adapter in selected_adapters:
        if old_adapter==current_adapter:
            continue

        if old_adapter=="default":
            continue

        for module in model.modules():
            if hasattr(module,"freeze_old_adapters"):
                module.freeze_old_adapters(old_adapter)

    ##### Collect ONLY trainable params
    lora_params=[]

    real_adapter_name=get_real_adapter_name(
        model,
        current_adapter
    )

    for name,param in model.named_parameters():
        ##### Current adapter LoRA trainable
        if (
            f".{real_adapter_name}." in name
            and ("lora_A" in name or "lora_B" in name)
        ):
            param.requires_grad=True
            lora_params.append(param)

        ##### Lateral modules ALWAYS trainable only when is selected
        elif (
            "lateral_A" in name
            or "lateral_B" in name
            or "transfer_gate" in name
        ):
            param.requires_grad=False

            for old_adapter in selected_adapters:
                if old_adapter in name:
                    param.requires_grad=True
                    lora_params.append(param)
                    break

        ##### Everything else frozen
        else:
            param.requires_grad=False

    print(f"Collected {len(lora_params)} trainable params")

    ##### MANUALLY override optimizer
    trainer.create_optimizer()

    trainer.optimizer.param_groups[0]["params"] = lora_params
    print("\n[AFTER TRAINER INIT]")
    model.print_trainable_parameters()

    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        ##### Enable lateral transfer during TRAINING
        for module in model.modules():
            if hasattr(module,"enable_lateral_transfer"):
                module.enable_lateral_transfer=True

        checkpoint=None
        if training_args.resume_from_checkpoint is not None:
            checkpoint=training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint=last_checkpoint

        train_result=trainer.train(resume_from_checkpoint=checkpoint)

        ##### Checking Learned gate values after training
        print("\n========== LEARNED GATES ==========\n")

        for module_name,module in model.named_modules():
            if hasattr(module,"transfer_gate"):
                for gate_name,gate in module.transfer_gate.items():
                    value=torch.sigmoid(gate).item()
                    print(module_name,gate_name,value)

        peft_model_id = training_args.output_dir + "/adapter"
        trainer.model.save_pretrained(peft_model_id)  
        tokenizer.save_pretrained(peft_model_id)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Metrics {metrics}")
        all_metrics.update(metrics)

    # Evaluation
    results = {}
    # in case the batch is shorter than max length, the output should be padded
    max_new_tokens = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    repetition_penalty = data_args.repetition_penalty

    if training_args.do_predict:
        logger.info("*** Prediction ***")
        logger.info("*** Multi-Adapter Evaluation ***")

        if data_args.max_predict_samples is not None:
            predict_dataset=predict_dataset.select(range(data_args.max_predict_samples))

        ##### Collect available adapters
        available_adapter_names=list(model.peft_config.keys())
        print(f"\n[AVAILABLE ADAPTERS] {available_adapter_names}")

        ##### Evaluate each adapter separately
        for adapter_name in available_adapter_names:
            print(f"\n[EVALUATING ADAPTER] {adapter_name}")

            ##### Internal adapter name (PEFT)
            internal_adapter_name=get_real_adapter_name(model,adapter_name)
            model.set_adapter(internal_adapter_name)

            ##### External dataset name
            dataset_name=FIRST_TASK if adapter_name=="default" else adapter_name

            print("[CURRENT DATASET NAME]",dataset_name)
            print("[ALL DATASETS IN PREDICT]",sorted(set(predict_dataset["Dataset"])))

            task_dataset=predict_dataset.filter(
                lambda x:
                x["Dataset"].lower().replace("-","")
                ==
                dataset_name.lower().replace("-","")
            )

            ##### Skip empty datasets
            if len(task_dataset)==0:
                print(f"[SKIPPED] No samples for {adapter_name}")
                continue

            ##### Predict
            predict_results=trainer.predict(
                task_dataset,
                metric_key_prefix=f"predict_{adapter_name}",
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id
            )

            metrics=predict_results.metrics
            metrics["predict_samples"]=len(task_dataset)

            trainer.log(metrics)
            trainer.log_metrics(f"predict_{adapter_name}",metrics)
            trainer.save_metrics(f"predict_{adapter_name}",metrics)

            all_metrics.update(metrics)

    return results


if __name__ == "__main__":
    main()
