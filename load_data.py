import json
import os
import pandas as pd
import torch
import re
from difflib import SequenceMatcher
from PIL import Image
import numpy as np
rng = np.random.default_rng(12345)

def read_ocr_tagged_file(filepath):
    """
    Returns:
        Dict {
            bbox -> [x0,y0,x1,y1]
            text : text
        }
    """
    with open(filepath,"r") as file:
        lines = file.readlines()
        lines = [line[:len(line) - 1] for line in lines]
        lines = [line.split(',') for line in lines]
        lines = [{'bbox': torch.tensor([int(i) for i in line[:2] + line[4:6]]),
                  'text':','.join(line[8:])
                 } for line in lines]
    return lines

def read_img_file(filepath):
    """
    Returns:
        dictionary object with width and height of image
    """
    image = Image.open(filepath)
    width,height = image.size
    return {'width':width,'height':height}

def read_label_file(filepath):
    """
    Returns:
        dictionary object with labels
    """
    with open(filepath,"r") as file:
        text = file.read()
        label_output = json.loads(text)
    return label_output

def get_data():
    #change the path here if you are not FRED
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
    names = ocr_tagged_filenames.intersection(label_filenames).intersection(image_filenames)
    
    labels = []
    sizes = []
    df2 = pd.DataFrame(index = names, columns = {'ocr_output'})
    for name in names:
        df2.at[name, 'ocr_output'] = read_ocr_tagged_file(path_1 + name + ".txt") 
        labels += [read_label_file(path_2 + name + ".txt")]
        sizes += [read_img_file(path_2 + name + ".jpg")]
    df1 = pd.DataFrame(labels,index=names)
    df3 = pd.DataFrame(sizes,index = names)
    train_df = pd.concat([df1, df2, df3], axis=1).rename_axis('file_name').reset_index()
    return train_df

def assign_line_label(line: str, entities):
    line_set = list(filter(None, re.split(r"[ ,/()\[\]]", line)))#line.replace(",", "").strip().split()
    thresholds = {'company': .5 + .5/(1+len(line_set)), 'date': 0.90, 'address': 0.70, 'total': 0.90}
    match = "O"
    for k, v in entities.iteritems():
        entity_set = list(filter(None, re.split(r"[ ,/()\[\]]", v)))
        matches_count = 0
        for l in line_set:
            if any(SequenceMatcher(None, a=l, b=b).ratio() > thresholds[k] for b in entity_set):
                matches_count += 1
        if matches_count == len(line_set) or matches_count == len(entity_set):
            match = k.upper()
    return match

def raw_labels(data):
    labels = []
    for i in data.index:
        line_labels = []
        for line in data['ocr_output'][i]:
            line_labels.append(assign_line_label(line['text'], data.iloc[i,1:5]))
        labels.append(line_labels)
    return labels

def process_labels(labels):
    for i, row in enumerate(labels):
        #take first date
        l = len(row)
        
        try:
            first_date = row.index('DATE')
            last_date = l - 1 - row[::-1].index('DATE')
            if first_date != last_date:
                labels[i] = [x if (x!='DATE' or j == first_date) else 'O' for j,x in enumerate(labels[i])]
        except:
            print("There are no DATES detected")

        try:
            first_total = row.index('TOTAL')
            last_total = l - 1 - row[::-1].index('TOTAL')
            if first_total != last_total:
                labels[i] = [x if (x!='TOTAL' or j == last_total) else 'O' for j,x in enumerate(labels[i])]
        except:
            print("There are no TOTALS")
            
        labels_dict = {'O': 0, 'DATE': 1, 'TOTAL': 2, 'COMPANY':3, 'ADDRESS':4}
        labels[i] = [labels_dict[x] for x in labels[i]]

    return labels        

def replace_char_with_same_type(char,percent_change = 1, percent_insert = 0, percent_delete = 0):
    if percent_change != 1 and rng.random() > percent_change:
        # no change
        return char 
    else:    
        if 65 <= ord(char) <= 90: #ord("A") <= ord(char) <= ord("Z")
            return chr(rng.integers(low=65, high=90))
        elif 97 <= ord(char) <= 122: #ord("a") <= ord(char) <= ord("z")
            if rng.random() < percent_delete:
                return ""
            elif rng.random() < percent_insert:
                return chr(rng.integers(low=97, high=122)) + chr(rng.integers(low=97, high=122))
            else:
                return chr(rng.integers(low=97, high=122))
            return chr(rng.integers(low=97, high=122))
        elif 48 <= ord(char) <= 57: #ord("0") <= ord(char) <= ord("9")
            if rng.random() < percent_delete:
                return ""
            elif rng.random() < percent_insert:
                return chr(rng.integers(low=48, high=57)) + chr(rng.integers(low=48, high=57))
            else:
                return chr(rng.integers(low=48, high=57))
        elif char in " .,":
            return " .,"[rng.integers(low=0, high=2)]
        else:
            return char


def augment_data(line, label, augmentation):
    """
    Input: words is string representing one line of ocr output
    label: what we label that line (0-Nothing/1-Date/2-Total/3-Company/4-Address)
    """
    p_lines = augmentation["p_lines"] if (augmentation != None and "p_lines" in augmentation) else 1
    p_characters = augmentation["p_char"] if (augmentation != None and "p_char" in augmentation) else 1
    p_insert = augmentation["p_insert"] if (augmentation != None and "p_insert" in augmentation) else 0
    p_delete = augmentation["p_delete"] if (augmentation != None and "p_delete" in augmentation) else 0

    if rng.random() > p_lines:
        return line
    else:
        new_line = ""
        for char in line:
            new_line += replace_char_with_same_type(char,percent_change = p_characters,percent_insert = p_insert,percent_delete = p_delete)
        return new_line

def boiler_plate(dataset, tokenizer, max_seq_length,augmentation=None):
    encoded = []
    bboxes = []
    labels = process_labels(raw_labels(dataset))
    adj_labels=[]

    copies = augmentation["copies"] if (augmentation != None and "copies" in augmentation) else 1

    for x in range(copies + 1):
        for i in dataset.index:
            words = [element['text'] for element in dataset['ocr_output'][i]]
            width = dataset['width'][i]
            height = dataset['height'][i]
            scaling_factor = torch.tensor([1000/width,1000/height,1000/width,1000/height])
            bbox = [(element['bbox'] * scaling_factor).long() for element in dataset['ocr_output'][i]]
            label = labels[i]

            if augmentation != None and x != 0:
                words = augment_data(words,label,augmentation)
            token_boxes = []
            label_list = []
            for j, (word, box) in enumerate(zip(words, bbox)):
                word_tokens = tokenizer.tokenize(word)
                token_boxes.extend([box] * len(word_tokens))
                label_list.extend([label[j]] * len(word_tokens))
            
            token_boxes = [[0,0,0,0]] + token_boxes + [[1000,1000,1000,1000]] + [[0,0,0,0]]*(max_seq_length - len(token_boxes) - 2)
            label_list = [0] + label_list + [0] + [0]*(max_seq_length - len(label_list) - 2)

            if len(token_boxes) > max_seq_length: #truncation of token boxes
                token_boxes = token_boxes[:max_seq_length - 1] + [[1000,1000,1000,1000]]
                label_list = label_list[:max_seq_length - 1] + [0]
            
            encoding = tokenizer(' '.join(words), padding = "max_length", truncation = True, max_length = max_seq_length, return_tensors = "pt")
            encoded.append(encoding)
            bboxes.append(torch.tensor([token_boxes])) 
            adj_labels.append(torch.tensor([label_list]))

    return encoded, bboxes, adj_labels

def encode_data(dataset, tokenizer, max_seq_length=64,augmentation = None):
    """
    Args:
        dataset
        max_seq_length
        tokenizer

    Returns:
        batch encoding with input_ids, attention_mask and token_type_ids
    """

    encoded, bboxes, labels = boiler_plate(dataset, tokenizer, max_seq_length,augmentation = augmentation)
    
    for i in range(len(encoded)):
        encoded[i]['bbox'], encoded[i]['label'] = bboxes[i], labels[i]
    
    return encoded
