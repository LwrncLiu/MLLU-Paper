# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:05:45 2021

@author: lwrnc
"""
from torch.utils.data import Dataset
import load_data

class SROIE_Dataset(Dataset):
    """SROIE data loading"""
    
    def __init__(self, dataframe, tokenizer, max_seq_length = 256,augmentation = None):
        self.data = dataframe
        self.encoded = load_data.encode_data(self.data, tokenizer, max_seq_length = max_seq_length,augmentation = None)

    
    def __len__(self):
        """
        Returns
        -------
        len(dataset)

        """
        return self.data.shape[0]
    
    def __getitem__(self, i):
        """
        Returns:
            Dict{
                
            }
        of dataset[0]
        """
        return {
        'bbox':self.encoded[i]['bbox'].squeeze(),
        'labels':self.encoded[i]['label'].squeeze(),
        'input_ids':self.encoded[i]['input_ids'].squeeze(),
        'attention_mask':self.encoded[i]['attention_mask'].squeeze(),
        }
