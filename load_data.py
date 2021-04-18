import json
import os
import pandas as pd

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
        lines = [{'bbox': [int(i) for i in line[:2] + line[4:6]],
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

def get_train_data():
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


def boiler_plate_for_encoding(dataset, tokenizer, max_seq_length):
    """
    Parameters
        dataset, tokenizer, max_seq_length
    Returns
        encoded: list of tokenized
        bboxes: list of bboxes
    """
    encoded = []
    bboxes = []
    for i in dataset.index:
        words = [element['text'] for element in dataset['ocr_output'][i]]
        bbox = [element['bbox'] for element in dataset['ocr_output'][i]]
        
        token_boxes = [] #tokenize by word/phrase, and adjust number of bounding box copies accordingly
        for word, box in zip(words, bbox):
            word_tokens = tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))
        
        token_boxes = [[0,0,0,0]] + token_boxes + [[1000,1000,1000,1000]] + [[0,0,0,0]]*(max_seq_length - len(token_boxes) - 2)
        
        encoding = tokenizer(' '.join(words), padding = "max_length", truncation = True, max_length = max_seq_length, return_tensors = "pt")
        encoded.append(encoding)
        bboxes.append(torch.tensor([token_boxes]))

    return encoded, bboxes

def encode_data(dataset, tokenizer, max_seq_length=512):
    """
    Args:
        dataset
        max_seq_length
        tokenizer

    Returns:
        batch encoding with input_ids, attention_mask and token_type_ids
    """
    
    encoded, bboxes = boiler_plate_for_encoding(dataset, tokenizer, max_seq_length)
       
    for i in range(len(encoded)): #attach bbox to each encoded value
        encoded[i]['bbox'] = bboxes[i]
        
    return encoding

    
#     return labels
