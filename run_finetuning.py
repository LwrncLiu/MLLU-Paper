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

test_data = sroie.SROIE_Dataset(df.iloc[-200:].reset_index(drop = True), tokenizer, augmentation = None)

for epoch in [3,5,7]:

    for n in [25, 50, 75, 100, 200, 300, 400]:

        current_df = df.iloc[:n].reset_index(drop = True)

        #loop for augmentation parameter
        for k in [[0,0,0], [2,0.8,0.8], [3,0.8,0.8]]:
            print("augmentation parameters: ", k)
            aug_params = {
            "copies" : k[0],
            "p_lines" : k[1], #100%
            "p_char" : k[2], #100%
            }

            train_data = sroie.SROIE_Dataset(current_df, tokenizer, augmentation = aug_params)

            #loop for n sample size
                #train_size = n
                #test_size = 0.15*len(sroie_dataset)
                #val_size = len(sroie_dataset) - train_size - test_size
                #train_data, test_data, extra_data = torch.utils.data.random_split(sroie_dataset, [train_size, test_size, val_size])

            training_args = TrainingArguments(
                    output_dir = '/scratch/kl2487/mllu',
                    num_train_epochs = epoch,
                    per_device_train_batch_size = 16,
                    evaluation_strategy = "epoch",
                    do_predict=True
                    )
            trainer = Trainer(
                    model_init = finetuning_utils.model_init,
                    train_dataset = train_data,
                    eval_dataset = test_data,
                    tokenizer = tokenizer,
                    compute_metrics = finetuning_utils.compute_metrics,
                    args = training_args
                    )
            print("fine-tuning with ", n, "samples")
            trainer.train()
                #predictions=trainer.predict(valid_data)
                #predictions.metrics
                #print(predictions)
