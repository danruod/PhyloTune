def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import argparse
import os
from sklearn.preprocessing import normalize
import csv
import sys
import numpy as np
import pandas as pd
from collections import Counter
import time
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm

import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from model.bert_layers import BertModel

csv.field_size_limit(sys.maxsize)
csv.field_size_limit(sys.maxsize)

def get_embedding(dna_sequences, 
                  model, 
                  species, 
                  task_name="clustering",
                  post_fix="",
                  test_model_dir="./test_model",):

    batch_size = 32
    
    embedding_dir = f"embeddings/{species}/{task_name}{post_fix}"
    embedding_file = os.path.join(embedding_dir, "dnaberts.npy")
          
    if os.path.exists(embedding_file):
        print(f"Load embedding from file {embedding_file}")
        embedding = np.load(embedding_file)
    
    else:
        print(f"Calculate embedding for {model} {species} {task_name}")
        
        embedding = calculate_llm_embedding(dna_sequences,  
                                            model_name_or_path=test_model_dir, 
                                            model_max_length=5000,
                                            batch_size=batch_size,)
        
        print(f"Save embedding to file {embedding_file}")
        os.makedirs(embedding_dir, exist_ok=True)
        np.save(embedding_file, embedding)
        
    return embedding

def calculate_llm_embedding(dna_sequences, model_name_or_path, model_max_length=400, batch_size=20):
    # reorder the sequences by length
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=None,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

    is_hyenadna = "hyenadna" in model_name_or_path
    is_nt = "nucleotide-transformer" in model_name_or_path
    
    if is_nt:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        ) 
    else:
        model = BertModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = nn.DataParallel(model)
        
    model.to("cuda")

    train_loader = util_data.DataLoader(dna_sequences, batch_size=batch_size*n_gpu, shuffle=False, num_workers=2*n_gpu)
    for j, batch in enumerate(tqdm.tqdm(train_loader)):
        with torch.no_grad():
            token_feat = tokenizer.batch_encode_plus(
                    batch, 
                    max_length=model_max_length, 
                    return_tensors='pt', 
                    padding='longest', 
                    truncation=True
                )
            input_ids = token_feat['input_ids'].cuda()
            attention_mask = token_feat['attention_mask'].cuda()
            if is_hyenadna:
                model_output = model.forward(input_ids=input_ids)[0].detach().cpu()
            else:
                model_outputs = model.forward(input_ids=input_ids, 
                                              attention_mask=attention_mask,
                                              output_all_encoded_layers=True)
                model_output = model_outputs[0].detach().cpu()
                model_attention = model_outputs[-1][-1].detach().cpu()
                
                attn_score = model_attention[:,:,0,1:-1].sum(0).sum(1)
                
            attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
            embedding = torch.sum(model_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
            # embedding = (model_output*attention_mask)[:, 0, :]
            
            if j==0:
                embeddings = embedding
            else:
                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().cpu())
    
    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings

def main(args, train_samples=100):
    all_data = pd.read_csv(f'{args.data_path}/id.csv')
    ood_data = pd.read_csv(f'{args.data_path}/ood.csv')
    
    # generate embedding
    X = get_embedding(all_data['seq'].values, args.test_model_dir.split('/')[-1], args.data_path.split("/")[-2], 
                      test_model_dir=args.test_model_dir, task_name=args.data_path.split("/")[-1], )
    X = normalize(X)
    X = StandardScaler().fit_transform(X)
    print('**Original X.shape:', X.shape)
    
    X_ood = get_embedding(ood_data['seq'].values, args.test_model_dir.split('/')[-1], args.data_path.split("/")[-2],
                          test_model_dir=args.test_model_dir, post_fix = '-ood',
                          task_name=args.data_path.split("/")[-1])
    X_ood = normalize(X_ood)
    X_ood = StandardScaler().fit_transform(X_ood)[:5]
    print('**Original X_ood.shape:', X_ood.shape)
    
    X_ood = np.concatenate([X_ood, X[all_data[args.col_name] == 'OOD']])
    X = X[all_data[args.col_name] != 'OOD']
    print('**Adjusted X.shape, X_ood.shape:', X.shape, X_ood.shape)
    
    # remove 'OOD'
    labels = all_data[args.col_name].values
    labels = labels[labels != 'OOD']
    
    label2id = {l: i for i, l in enumerate(set(labels))}
    y = np.array([label2id[l] for l in labels])
    
    assert len(X) >= train_samples, "Not enough samples for training!"
    
    # print imbalance info for each class
    print("****** Data info:")
    for l, c in zip(*np.unique(y, return_counts=True)):
        print(f"  {l}: {c} ({c/len(y)*100:.2f}%)")
    
    results = {
        'mcc': [],
        'f1': [],
        'recall': [],
        'precision': [],
        'accuracy': [],
        'auroc': [],
        'aupr': [],
    }
    
    param_grid = {
        'C': [1e-3, 1e-2, 1e-1, 1, 1e2, 1e3],
        # 'solver': ['liblinear', 'saga']
        # 'C': [1e4, 1e5],
    }
    
    all_time = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
        start = time.time()
        print(f"Fold {fold+1}:")
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        if train_samples < len(X_train):  
            X_train, _, y_train, _ = train_test_split(X_train, y_train, 
                                                    train_size=train_samples, 
                                                    stratify=y_train, 
                                                    random_state=42)
        else:
            print(f"Train samples set to {len(X_train)}")

        print(f"  # Train: {len(X_train)}")
        for l, c in zip(*np.unique(y_train, return_counts=True)):
            print(f"    {l}: {c} ({c/len(y_train)*100:.2f}%)")
            
        print(f"  # Test: {len(X_test)}")
        for l, c in zip(*np.unique(y_test, return_counts=True)):
            print(f"    {l}: {c} ({c/len(y_test)*100:.2f}%)")
        
        # perform classification
        print("Training SVC...")
        # model = SVC(kernel='linear', class_weight='balanced', C=0.01)
        
        grid = GridSearchCV(SVC(kernel='linear', class_weight='balanced'),
                            param_grid, scoring='f1_weighted', cv=5)
        
        grid.fit(X_train, y_train)
        print("Best parameters:", grid.best_params_)
        print("Best weighted F1:", grid.best_score_)
        
        model = SVC(kernel='linear', class_weight='balanced', C=grid.best_params_['C'])
        
        model.fit(X_train, y_train)
        preds_test = model.predict(X_test)
        preds_train = model.predict(X_train)
        
        f1_train = sklearn.metrics.f1_score(y_train, preds_train, average="macro", zero_division=np.nan)
        # loss_train = sklearn.metrics.log_loss(y_train, preds_test.predict_proba(X_train))
                        
        mcc = sklearn.metrics.matthews_corrcoef(y_test, preds_test)
        f1 = sklearn.metrics.f1_score(y_test, preds_test, average="macro", zero_division=np.nan)
        recall = sklearn.metrics.recall_score(y_test, preds_test, average="macro", zero_division=np.nan)
        precision = sklearn.metrics.precision_score(y_test, preds_test, average="macro", zero_division=np.nan)
        accuracy = sklearn.metrics.accuracy_score(y_test, preds_test)
        mcc = sklearn.metrics.matthews_corrcoef(y_test, preds_test)
        
        results['mcc'].append(mcc)
        results['f1'].append(f1)
        results['recall'].append(recall)
        results['precision'].append(precision)
        results['accuracy'].append(accuracy)
        
        print(f"\n----- Fold {fold+1} classification report -----")
        print(classification_report(y_test, preds_test))
    
        print(f"train f1: {f1_train} test MCC: {mcc} f1: {f1} recall: {recall} precision: {precision} accuracy: {accuracy}")
        
        test_scores = model.decision_function(X_test)
        ood_scores = model.decision_function(X_ood)
        
        if test_scores.ndim == 2:
            test_scores = np.abs(test_scores).max(axis=1)
            ood_scores = np.abs(ood_scores).max(axis=1)
        else:
            test_scores = np.abs(test_scores)
            ood_scores = np.abs(ood_scores)
            
        # concat
        y_true = np.concatenate([np.ones(len(test_scores)), np.zeros(len(ood_scores))])
        # y_true = np.concatenate([np.zeros(len(test_scores)), np.ones(len(ood_scores))])
        scores = np.concatenate([test_scores, ood_scores])
        auroc = sklearn.metrics.roc_auc_score(y_true, scores)
        # aupr = sklearn.metrics.average_precision_score(y_true, scores)
        aupr = sklearn.metrics.average_precision_score(1-y_true, -scores)
        
        results['auroc'].append(auroc)
        results['aupr'].append(aupr)
        
        print(f"AUROC: {auroc} AUPR: {aupr}")
        all_time.append(time.time() - start)
        
        print('Done!')
          
    print(f"Average MCC: {np.mean(results['mcc'])} f1: {np.mean(results['f1'])} recall: {np.mean(results['recall'])} precision: {np.mean(results['precision'])} accuracy: {np.mean(results['accuracy'])}")
    print(f"Average AUROC: {np.mean(results['auroc'])} AUPR: {np.mean(results['aupr'])}")
    
    return results, all_time
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate clustering')
    parser.add_argument('--data_path', type=str, default="./datasets/simulated/1_train_test", help='Data directory')
    parser.add_argument('--test_model_dir', type=str, default="/mnt/drdeng/models/dnaberts", help='Directory to save trained models to test')
    parser.add_argument('--task_name', type=str, default="simulated", help='Task name')
    parser.add_argument('--balance', action='store_true', help='Balance data')
    parser.add_argument('--col_name', type=str, default="clade", help='Column name for label')
    
    args = parser.parse_args()
    with open('result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['simulate', 'samples', 'mcc', 'f1', 'recall', 'precision', 'accuracy', 'auroc', 'aupr', 'time'])


    data_path = args.data_path
    for file in os.listdir(data_path):
        for k in range(90, 0, -10):
            args.data_path = f"{data_path}/{file}"
            results, all_time = main(args, k)
            
            assert len(all_time) == 10, "Not enough folds!"
    
            for i in range(len(results['f1'])):
                with open('result.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([file, k, results['mcc'][i], results['f1'][i], results['recall'][i], results['precision'][i], results['accuracy'][i], results['auroc'][i], results['aupr'][i], all_time[i]])