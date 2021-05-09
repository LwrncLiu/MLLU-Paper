# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:54:05 2021

@author: lwrnc
"""
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import LayoutLMForTokenClassification

def compute_metrics(eval_pred):
    """Computes accuracy, f1, precision, and recall from a 
    transformers.trainer_utils.EvalPrediction object.
    """
    labels = eval_pred.label_ids.flatten()
    preds = eval_pred.predictions.argmax(-1).flatten()

    ## TODO: Return a dictionary containing the accuracy, f1, precision, and recall scores.
    ## You may use sklearn's precision_recall_fscore_support and accuracy_score methods.
    
    precision, recall, f1_score, support = precision_recall_fscore_support(y_true = labels, y_pred = preds)
    accuracy = accuracy_score(y_true = labels, y_pred = preds, normalize = True)
    
    metrics_dict = {
        "accuracy": accuracy,
        "f1": f1_score,
        "precision": precision,
        "recall": recall
        }
    
    return metrics_dict

def model_init():
    """Returns an initialized model for use in a Hugging Face Trainer."""
    
    model = LayoutLMForTokenClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased",
        num_labels=5,
        from_tf = True
        )
    
    model = model.to('cuda')
    
    return model
