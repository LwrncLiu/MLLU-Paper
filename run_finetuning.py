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

aug_params = {
    "copies" : 3,
    "p_lines" : 1.0, #100%
    "p_char" : 1.0, #100%
}

sroie_dataset = sroie.SROIE_Dataset(df, tokenizer,augmentation = aug_params)

train_size = int(0.7*len(sroie_dataset))
test_size = int(0.2*len(sroie_dataset))
valid_size = len(sroie_dataset) - train_size - test_size
train_data, test_data, valid_data = torch.utils.data.random_split(sroie_dataset, [train_size, test_size, valid_size])

training_args = TrainingArguments(
    output_dir = '/scratch/kl2487/mllu',
    num_train_epochs = 5,
    per_device_train_batch_size = 16,
    evaluation_strategy = "epoch"
    )
trainer = Trainer(
    model_init = finetuning_utils.model_init,
    train_dataset = train_data,
    eval_dataset = valid_data,
    tokenizer = tokenizer,
    compute_metrics = finetuning_utils.compute_metrics,
    args = training_args
    )
trainer.train()
print(trainer.predict(test_data))
