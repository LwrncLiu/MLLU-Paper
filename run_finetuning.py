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



#augmentation type loop
for k in [[0,0,0], [2,0.8,0.8], [3,0.8,0.8]]:
    print("augmentation parameters: ", k)
    aug_params = {
    "copies" : k[0],
    "p_lines" : k[1], #100%
    "p_char" : k[2], #100%
    }
    
    sroie_dataset = sroie.SROIE_Dataset(df, tokenizer, augmentation = aug_params)

    #sample size loop
    for n in [25, 50, 75, 100, 200, 300, 400]:
        train_size = n
        test_size = 0.15*len(sroie_dataset)
        train_data, test_data = torch.utils.data.random_split(sroie_dataset, [train_size, test_size])

        training_args = TrainingArguments(
            output_dir = '/scratch/kl2487/mllu',
            num_train_epochs = 5,
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
