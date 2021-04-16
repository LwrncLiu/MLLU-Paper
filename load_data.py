# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 14:03:23 2021

@author: lwrnc
load data functions
"""
import torch
import json
from transformers import LayoutLMTokenizerFast

def read_ocr_tagged_file(filepath):
    """
    Returns:
        Dict {
            bbox-top-left  -> [x,y]
            bbox-bot-right -> [x,y]
            text : text
        }
    """
    with open(filepath,"r") as file:
        lines = file.readlines()
        lines = [line[:len(line) - 1] for line in lines]
        lines = [line.split(',') for line in lines]
        lines = [{'bbox-top-left':line[:2],
                  'bbox-bot-right':line[4:6],
                  'text':','.join(line[8:])
                    } for line in lines]
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


def process_for_encode(dataset):
    """
    Parameters
        dataset
    Returns
        processed_passage: list of strings from receipt
        processed_question: list of keys and answers"""
    processed_passage = []
    processed_question = []
    for i in dataset.index:
        words = [' '+element['text'] for element in dataset['ocr_output'][i]]
        text = ''.join(words)
        processed_passage.append(text)
        
        keys = dataset.columns.to_list()[1:]
        company = dataset[keys[0]][i]
        date = dataset[keys[1]][i]
        address = dataset[keys[2]][i]
        total = dataset[keys[3]][i]
        value_str = " ".join([company, date, address, total])
        i_input = keys[0] + " : " + keys[1] + " : " + keys[2] + " : " + keys[3] + " : " + value_str
        processed_question.append(i_input)
    return processed_passage, processed_question

def encode_data(dataset, tokenizer, max_seq_length=64):
    """
    Args:
        dataset
        max_seq_length
        tokenizer

    Returns:
        batch encoding with input_ids, attention_mask and token_type_ids
    """
    
    passages, questions = process_for_encode(dataset)
        
    batch_encoding = tokenizer(questions, passages, padding = "max_length", truncation = True, max_length = max_seq_length, return_tensors = "pt")
    
    return batch_encoding


# def extract_labels(dataset):
#     """Converts labels into numerical labels.

#   Args:
#     dataset: A Pandas dataframe containing the labels in the column 'label'.

#   Returns:
#     labels: A list of integers corresponding to the labels for each example,
#       where 0 is False and 1 is True.
#   """
#     ## TODO: Convert the labels to a numeric format and return as a list.
  
#     label_str = dataset['label'].to_list() #gets labels from df as list
#     labels = [1 if i == True else 0 for i in label_str] #1 if label is True,0 if False
    
#     return labels