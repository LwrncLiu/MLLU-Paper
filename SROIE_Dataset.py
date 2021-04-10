import torch
from torch.utils.data import Dataset

import json
import pandas as pd


def read_ocr_tagged_file(filepath):
    """
    Returns:
        Dict {
            bbox-top-left  -> [x,y]
            bbox-top-right -> [x,y]
            bbox-bot-right -> [x,y]
            bbox-bot-left  -> [x,y]
            text : text
        }
    """
    with open(filepath,"r") as file:
        lines = file.readlines()
        lines = [line.split(',') for line in lines]
        lines = [{'bbox-top-left':line[:2],
                'bbox-top-right':line[2:4],
                'bbox-bot-right':line[4:6],
                'bbox-bot-left':line[6:8],
                'text':','.join(line[8:])
                } for line in lines]
        # not yet processed with a tokenizer
    return lines

def read_img_file(filepath):
    """
    NOT IMPLEMENTED
    Returns:
        None
    """
    return None

def read_label_file(filepath):
    """
    Returns:
        dictionary object with labels
    """
    with open(filepath,"r") as file:
        text = file.read()
        label_output = json.loads(text)
    return label_output


class SROIE_Dataset(Dataset):
    """SROIE"""
    
    def __init__(self, tokenizer):
        path_1 = "/scratch/fs1493/mlu_project/SROIE/0325updated.task1train(626p)/"
        path_2 = "/scratch/fs1493/mlu_project/SROIE/0325updated.task2train(626p)/"

        files = [name for name in os.listdir(path_1) if os.path.isfile(path_1 + name)]
        ocr_tagged_filenames = set([name[:-4] for name in files if name[-3:] == "txt" and name[-5] != ")"])

        files_2 = [name for name in os.listdir(path_2) if os.path.isfile(path_2 + name)]
        
        image_filenames = set([name[:-4] for name in files if name[-3:] == "jpg" and name[-5] != ")"])

        label_filenames = set([name[:-4] for name in files if name[-3:] == "txt" and name[-5] != ")"])

        names = ocr_tagged_filenames.intersection(label_filenames)

        ocr_tagged_data = []
        labels = []
        for name in names:
            ocr_tagged_data += [read_ocr_tagged_file(path1 + name + ".txt")] 
            labels += [read_label_file(path2 + name + ".txt")]

        df1 = pd.DataFrame(labels,index=names)
        df2 = pd.DataFrame(ocr_tagged_data,index=names)

        self.data = pandas.concat([df1, df2], axis=1).rename_axis('example_name').reset_index()


    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, i):
        """
        Returns:
            Dict{
                
            }
        """
        return self.data.iloc[i].to_dict()