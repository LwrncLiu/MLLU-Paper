import json
import os
import pandas as pd
import torch
import re
from difflib import SequenceMatcher

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
    names = ocr_tagged_filenames.intersection(label_filenames)
    
    labels = []
    df2 = pd.DataFrame(inex = names, columns = {'ocr_output'})
    for name in names:
        df2.at[name, 'ocr_output'] = read_ocr_tagged_file(path_1 + name + ".txt") 
        labels += [read_label_file(path_2 + name + ".txt")]
        df1 = pd.DataFrame(labels,index=names)
    
    train_df = pd.concat([df1, df2], axis=1).rename_axis('file_name').reset_index()
    return train_df

def raw_labels(data):
    def assign_line_label(line: str, entities: pd.DataFrame):
        line_set = list(filter(None, re.split(r"[ ,/()\[\]]", line)))#line.replace(",", "").strip().split()
        thresholds = {'company': .5 + .5/len(line_set), 'date': 0.90, 'address': 0.70, 'total': 0.90}
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


def boiler_plate(dataset, tokenizer, max_seq_length):
    encoded = []
    bboxes = []
    labels = process_labels(raw_labels(dataset))
    adj_labels=[]
    for i in dataset.index:
        words = [element['text'] for element in dataset['ocr_output'][i]]
        bbox = [element['bbox'] for element in dataset['ocr_output'][i]]
        label = labels[i]
        
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

def encode_data(dataset, tokenizer, max_seq_length=64):
    """
    Args:
        dataset
        max_seq_length
        tokenizer

    Returns:
        batch encoding with input_ids, attention_mask and token_type_ids
    """
    
    encoded, bboxes, labels = boiler_plate(dataset, tokenizer, max_seq_length)
    
    for i in range(len(encoded)):
        encoded[i]['bbox'], encoded[i]['label'] = bboxes[i], labels[i]
    
    return encoded
