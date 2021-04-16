# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:05:45 2021

@author: lwrnc
"""
from torch.utils.data import Dataset
from load_data import read_label_file, read_ocr_tagged_file, encode_data
import os
import pandas as pd

class SROIE_Dataset(Dataset):
    """SROIE data loading"""
    
    def __init__(self, tokenizer, max_seq_length = 256):
        path_1 = "/scratch/fs1493/mlu_project/SROIE/0325updated.task1train(626p)/"
        path_2 = "/scratch/fs1493/mlu_project/SROIE/0325updated.task2train(626p)/"

        #list of all files .txt & .jpg from task1 (images + bounding boxes)
        files = [name for name in os.listdir(path_1) if os.path.isfile(path_1 + name)]
        #txt files with bounding box information
        ocr_tagged_filenames = set([name[:-4] for name in files if name[-3:] == "txt" and name[-5] != ")"])
        #list of all files .txt & .jpg from task2 (images + ground truth)
        files_2 = [name for name in os.listdir(path_2) if os.path.isfile(path_2 + name)]
        #all images files from task2
        image_filenames = set([name[:-4] for name in files if name[-3:] == "jpg" and name[-5] != ")"])
        #all ground truth files from task 2
        label_filenames = set([name[:-4] for name in files if name[-3:] == "txt" and name[-5] != ")"])
        
        #files names of all txt ids that appear in task1 & task2
        names = ocr_tagged_filenames.intersection(label_filenames)

        labels = []
        df2 = pd.DataFrame(inex = names, columns = {'ocr_output'})
        for name in names:
            df2.at[name, 'ocr_output'] = read_ocr_tagged_file(path_1 + name + ".txt") 
            labels += [read_label_file(path_2 + name + ".txt")]

        df1 = pd.DataFrame(labels,index=names)

        self.data = pd.concat([df1, df2], axis=1).rename_axis('file_name').reset_index()
        self.encoded_data = encode_data(self.data, tokenizer, max_seq_length = max_seq_length)

    
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
        return self.data.iloc[i].to_dict()