import load_data
from transformers import LayoutLMTokenizerFast, TrainingArguments, Trainer
import torch
import finetuning_utils
import sroie
import numpy as np
import os
import pandas as pd

df = load_data.get_data()

msk = np.random.rand(len(df)) < 0.8

train_df = df[msk].reset_index(drop=True)
val_df = df[~msk].reset_index(drop = True)

tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")

train_data = sroie.SROIE_Dataset(train_df, tokenizer)
val_data = sroie.SROIE_Dataset(val_df, tokenizer) #not final

training_args = TrainingArguments(
    output_dir = '/scratch/fs1493/mlu_project',
    num_train_epochs = 3,
    per_device_train_batch_size = 8,
    evaluation_strategy = "epoch"
    )
trainer = Trainer(
    model_init = finetuning_utils.model_init,
    train_dataset = train_data,
    eval_dataset = val_data,
    tokenizer = tokenizer,
    compute_metrics = finetuning_utils.compute_metrics,
    args = training_args
    )
trainer.train()
