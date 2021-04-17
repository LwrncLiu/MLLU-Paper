import load_data
from transformers import LayoutLMTokenizerFast, TrainingArguments, Trainer
import torch
import finetuning_utils
import sroie
import os
import pandas as pd


train_df = load_data.get_train_data()
val_df = load_data.get_train_data() #need a new function to get validation data

tokenizer = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")

train_data = sroie.SROIE_Dataset(train_df, tokenizer)
val_data = sroie.SROIE_Dataset(val_df, tokenizer) #not final

training_args = TrainingArguments(
    output_dir = '/scratch/ll3492/hw3/out',
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
