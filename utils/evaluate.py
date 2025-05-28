import os
import torch
import numpy as np
from tqdm import tqdm
import sklearn
from typing import Any, Tuple, Union

import pandas as pd

from utils.predict import pred_for_seqs
from utils.plot import plot_confusion_matrix


def preprocess_logits_for_metrics(logits:Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    return logits


def compute_metrics_details(eval_pred):
    logits, labels = eval_pred
    
    if not isinstance(logits, list):
        logits = [logits]
        
    predictions = []
    scores = {}
    is_id = []
    for i in range(len(logits)):
        if logits[i].ndim == 3:
            logits[i] = logits[i].reshape(-1, logits[i].shape[-1])
        
        predictions.append(np.argmax(logits[i], axis=-1))
        # if label = -1, it is ood and set the value in is_id is 0, else 1
        is_id.append(labels[:, i] != -100)

        if not all(is_id[i]):
            scores[i] = {}
            logits[i] = torch.tensor(logits[i])
            prob = torch.nn.functional.softmax(logits[i], dim=-1)
            
            scores[i]['max_prob'] = np.max(prob.numpy(), axis=-1)
            scores[i]['neg_ent'] = torch.sum(prob * torch.nn.functional.log_softmax(logits[i], dim=-1), dim=-1).numpy()
            
    predictions = np.stack(predictions, axis=-1)
    is_id = np.stack(is_id, axis=-1)
        
    # calculate metrics for each level
    results = {}
    for i in range(len(logits)):
        results[f"level_{i}"] = calculate_metric_with_sklearn(predictions[:, i], labels[:, i])
        
        if not all(is_id[:, i]):
            for metric, value in scores[i].items():
                ood_result = calculate_ood_metric_with_sklearn(value, is_id[:, i], name=metric)
                results[f"level_{i}"] = {**results[f"level_{i}"], **ood_result}
                
    return results


"""
Manually calculate the auroc, aupr with sklearn.
"""
def calculate_ood_metric_with_sklearn(scores: np.ndarray, labels: np.ndarray, name="max_p"):
    # return the metric for ood detection
    return {
        f'auroc-{name}': sklearn.metrics.roc_auc_score(labels, scores),
        f'aupr-{name}': sklearn.metrics.average_precision_score(labels, scores),
    }


"""
Manually calculate the accuracy, matthews_correlation, f1, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray, average: str = "macro", zero_division=0):
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    assert all(valid_labels >= 0), "Labels contain padding tokens."
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average=average, zero_division=zero_division
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average=average, zero_division=zero_division
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average=average, zero_division=zero_division
        ),
    }


def calculate_metric(labels, predictions, scores, tags, tax_ranks):
    taxo = dict()
    ood = dict()
    
    if len(tags) > 0: 
        taxo_tag = []
        ood_tag = []
    
    for i, tax_rank in enumerate(tax_ranks):
        is_id = (labels[:, i] != -100)
        
        # Evaluate the performance of taxonomic classfication
        taxo[tax_rank] = calculate_metric_with_sklearn(predictions[:, i], labels[:, i])
        
        # Evaluate the performance of novelty detection 
        if not all(is_id) and any(is_id):
            ood[tax_rank] = {}
            for metric, value in scores.items():
                ood_result = calculate_ood_metric_with_sklearn(value[:, i], is_id, name=metric)
                ood[tax_rank] = {**ood[tax_rank], **ood_result}
        
        if len(tags) > 0:  
            
            # Evaluate the performance of taxonomic classfication for each marker
            for tag in np.unique(tags):
                idx = (tags == tag)
                
                # Evaluate the performance of taxonomic classfication
                taxo_results = calculate_metric_with_sklearn(predictions[idx, i], labels[idx, i], zero_division=np.nan)
                for key, value in taxo_results.items():
                    taxo_tag.append((tax_rank, tag, key, value))
        
                # Evaluate the performance of novelty detection 
                if not all(is_id[idx]) and any(is_id[idx]):
                    for metric, value in scores.items():
                        ood_result = calculate_ood_metric_with_sklearn(value[idx, i], is_id[idx], name=metric)
                        for key, value in ood_result.items():
                            ood_tag.append((tax_rank, tag, metric, key, value))
    
    results = {
        'Taxo': taxo,
        'Novelty': ood,
    }
    
    if len(tags) > 0:
        results_tag = {
            'Taxo-tag': pd.DataFrame(taxo_tag, columns=['tax_rank', 'tag', 'metric', 'value']),
            'Novelty-tag': pd.DataFrame(ood_tag, columns=['tax_rank', 'tag', 'score', 'metric', 'value']),
        }
        
        results = {**results, **results_tag}
                    
    return results


def get_outputs_plant(id_data_loader, ood_data_loader, bert, hlp, return_markers=False):
    hlp.eval()
    bert.eval()
    
    predictions = []
    labels = []
    scores = {
        'max_prob': [],
        'neg_ent': [],
    }
    tags = []
 
    # get logits, labels, markers for ID test set 
    for id_batch in tqdm(id_data_loader, desc="[Evaluate-ID dataset]"):
        output = pred_for_seqs((bert, hlp), id_batch)
        
        predictions.append(output['pred_label'])
        labels.append(id_batch[3])
        scores['max_prob'].append(output['max_prob'])
        scores['neg_ent'].append(-output['novelty_score'])
        if return_markers:
            tags.append(id_batch[4])
        
        # if len(predictions) >= 2:
        #     break
        # break
    
    # get logits, labels, markers for OOD test set  
    for ood_batch in tqdm(ood_data_loader, desc="[Evaluate-OOD dataset]"):
        output = pred_for_seqs((bert, hlp), ood_batch)
        
        predictions.append(output['pred_label'])
        labels.append(ood_batch[3])
        scores['max_prob'].append(output['max_prob'])
        scores['neg_ent'].append(-output['novelty_score'])
        if return_markers:
            tags.append(ood_batch[4])
            
        # if len(predictions) >= 2:
        #     break
        # break
        
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    scores = {key: torch.cat(value, dim=0) for key, value in scores.items()}
    if return_markers:
        tags = np.concatenate(tags, axis=0)
        
    # for ood label, we use -100 to represent
    for i, tax_rank in enumerate(hlp.tax_ranks):
        ood_idx = list(hlp.labels_maps[tax_rank].values()).index('ood')
        labels[labels[:, i] == ood_idx, i] = -100

    return labels, predictions, scores, tags


def test(id_test_data_loader, ood_test_data_loader, bert, hlp, return_markers=False, 
         plot=False, results_path=None):
    
    if os.path.exists(f'{results_path}/outputs.pt'):
        outputs = torch.load(f'{results_path}/outputs.pt')
        predictions = outputs['predictions']
        labels = outputs['labels']
        scores = outputs['scores']
        tags = outputs['tags']
    else:
        labels, predictions, scores, tags = get_outputs_plant(id_test_data_loader, 
                                                              ood_test_data_loader, 
                                                              bert, hlp, 
                                                              return_markers=return_markers,)
        
        # save labels, predictions, scores, tags
        if results_path is not None:
            os.makedirs(results_path, exist_ok=True)
            torch.save({'labels': labels, 'predictions': predictions, 'scores': scores, 'tags': tags}, 
                    f'{results_path}/outputs.pt')
            
    # evaluate
    results = calculate_metric(labels, predictions, scores, tags, hlp.tax_ranks)
    
    # confusion matrix
    if plot:
        figsize = [15, 15, 30, 50]
        os.makedirs(f'{results_path}/figs/confusion_matrix/', exist_ok=True)
        
        for i, tax_rank in enumerate(hlp.tax_ranks):
            classes = list(hlp.labels_maps[tax_rank].values())
            classes.remove('ood')
            is_id = (labels[:, i] != -100)
            plot_confusion_matrix(figsize[i], labels[is_id, i].cpu().numpy(), predictions[is_id, i], 
                                    list(hlp.labels_maps[tax_rank].keys())[:-1], list(hlp.labels_maps[tax_rank].values())[:-1], 
                                    save_path=f'{results_path}/figs/confusion_matrix/{tax_rank}.png')

    return results

# from utils.predict import pred_for_seqs

# def test_bord(data_loader, model, tax_ranks):
#     predictions = []

#     for batch in tqdm(data_loader):
#         outputs = pred_for_seqs(model, batch)
#         predictions.append(outputs['pred_label'])
    
#     predictions = torch.cat(predictions, dim=0)
#     labels = torch.tensor(data_loader.dataset.labels)
    
#     results = dict()

#     for i, tax_rank in enumerate(tax_ranks):
#         # Evaluate the performance of taxonomic classfication
#         results[tax_rank] = calculate_metric_with_sklearn(predictions[:, i], labels[:, i])    
    
#     return results


# def test_bord(data_loader, model, tax_ranks):
#     model.eval()
    
#     predictions = []
#     with torch.no_grad():
#         for input in tqdm(data_loader):
                
#             input_ids, token_type_ids, attention_mask = (
#                 input["input_ids"].to(model.device),
#                 input["token_type_ids"].to(model.device),
#                 input["attention_mask"].to(model.device),
#             )
        
#             model_outputs = model.forward(
#                 input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#             )
#             logits = model_outputs.logits

#             pred = []
#             for i in range(len(logits)):
#                 pred.append(torch.argmax(logits[i], dim=-1))
                
#             predictions.append(torch.stack(pred, dim=1))
            
    
#     torch.cuda.empty_cache()
    