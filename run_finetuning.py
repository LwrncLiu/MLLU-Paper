import load_data
from transformers import LayoutLMTokenizerFast, TrainingArguments, Trainer
import torch
import finetuning_utils
import sroie
import numpy as np
import os
import pandas as pd

df = load_data.get_data()

tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")

sroie_dataset = sroie.SROIE_Dataset(df, tokenizer)

train_size = int(0.8*len(sroie_dataset))
test_size = len(sroie_dataset)-train_size
train_dataset, test_dataset = torch.utils.data.random_split(sroie_dataset, [train_size, test_size])

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
