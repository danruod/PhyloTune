import os
import argparse
import time
import pandas as pd

import torch
from torch.utils.data import DataLoader

import transformers
from transformers import BertModel
from transformers import default_data_collator, set_seed

from model.probe import *
from model.bert_layers import BertForHierarchicalClassification

from utils.evaluate import test, preprocess_logits_for_metrics, compute_metrics_details
from utils.cuda import get_max_available_gpu
from utils.dataset import SequenceDatasetMultiRank, HierarchicalDataset

import warnings
warnings.filterwarnings('ignore')

def main(args):
    default_dtype = torch.float64
    torch.set_default_dtype(default_dtype)
    
    if torch.cuda.is_available():
        try:
            device_id, _ = get_max_available_gpu()
        except:
            device_id = 0
        print(f"Using GPU: {device_id}")
    else:
        print("Using CPU")
    
    device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")
    
    results_path = os.path.join(args.result_path, f'{args.dataset}/eval_taxa')
    os.path.join(args.result_path, f'{args.dataset}/bertphylo')
    
    os.makedirs(results_path, exist_ok=True)
            
    if args.dataset == 'Plant':
        
        data_path = os.path.join(args.data_path, f'{args.dataset}/1_train_test/dnabert')
        bert_path = os.path.join(args.model_path, 'Plant_dnabert/bert')
        probe_path = os.path.join(args.model_path, 'Plant_dnabert/hlps.pt')

        # load dataset
        print(f'############# Loading {args.dataset} #############')
        
        id_test_dataset = torch.load(os.path.join(data_path, "id_test_dataset.pt"))
        ood_test_dataset = torch.load(os.path.join(data_path, "ood_test_dataset.pt"))
        assert id_test_dataset.labels_maps == ood_test_dataset.labels_maps, f'The labels_map in ID test set and OOD test set are inconsistent. Please check.'
        
        id_test_data_loader = DataLoader(id_test_dataset, batch_size=args.batch_size, shuffle=False)
        ood_test_data_loader = DataLoader(ood_test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # get labels_maps and tax_ranks
        labels_maps = id_test_dataset.labels_maps
        tax_ranks = id_test_dataset.tax_ranks
        print(f'------ Label information -----')
        for tax_rank in tax_ranks:
            print(f'{tax_rank}: {list(labels_maps[tax_rank].values())}')
        print('---------------')
        
        print(f'Number of sequences in test set: {len(id_test_dataset)} (ID), {len(ood_test_dataset)} (OOD)')
        
        # load model
        print(f'############# Loading Model #############')
        ## load BERT module
        bert = BertModel.from_pretrained(bert_path)
        bert.to(device)
        
        ## load hierarchical linear probes
        hlp_config_dict = {'device': device, 
                           'default_dtype': default_dtype,
                           'model_name': 'bertphylo',
                           'tax_ranks': tax_ranks,
                           'labels_maps': labels_maps,
                        } 
        hlp = LinearProbeMultiRank(**hlp_config_dict)
        hlp.to(device)
        hlp.load_state_dict(torch.load(probe_path, map_location=device))
        
        print(f'############# Testing #############')
        start = time.time()
        results = test(id_test_data_loader, ood_test_data_loader, bert, hlp, 
                       return_markers=args.return_markers, plot=args.plot, 
                       results_path=results_path
                       )
        print(f'Time: {time.time()-start:.2f}s')
        
    elif args.dataset == 'Bordetella':
        
        data_path = os.path.join(args.data_path, f'{args.dataset}/1_train_test')
        model_path = os.path.join(args.model_path, 'Bordetella_dnaberts')
        
        # load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=5000,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    
        # load dataset 
        labels_maps = torch.load(os.path.join(data_path, "labels_maps.pt"))
        tax_ranks = list(labels_maps.keys())
        test_dataset = HierarchicalDataset(tokenizer=tokenizer, 
                                            data_path=[os.path.join(data_path, "test.csv"), 
                                                        os.path.join(data_path, "ood.csv")
                                                        ], 
                                            input_col='seq',
                                            label_cols=tax_ranks,
                                            labels_maps=labels_maps,)
                
        # load model
        model = BertForHierarchicalClassification.from_pretrained(
            model_path,
            cache_dir=None,
            trust_remote_code=True,
            labels_maps=labels_maps,
        )
        
        # evaluate
        print(f'############# Testing #############')
        start = time.time()
        trainer = transformers.Trainer(
            model=model, 
            tokenizer=tokenizer,
            args=transformers.TrainingArguments(
                per_device_eval_batch_size=args.batch_size,
                output_dir=results_path,
                report_to="none",
                ),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            compute_metrics=compute_metrics_details,
            data_collator=default_data_collator,
        )
        results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        print(f'Time: {time.time()-start:.2f}s')
        
        new_results = {'novelty': {}, 'taxo': {}}
        for key, value in results.items():
            if isinstance(value, dict):
                new_results['taxo'][key] = {}
                new_results['novelty'][key] = {}
                for sub_key, sub_value in value.items():
                    if 'au' in sub_key:
                        new_results['novelty'][key][sub_key] = sub_value
                    else:
                        new_results['taxo'][key][sub_key] = sub_value
            
        results = new_results
    
    os.makedirs(results_path, exist_ok=True)
    
    with pd.ExcelWriter(os.path.join(results_path, f'{args.exp_name}_{time.strftime("%y%m%d-%H%M%S")}.xlsx')) as writer:
        for key, value in results.items():
            pd.DataFrame(value).to_excel(writer, sheet_name=key)
            print(f'{key}: ', pd.DataFrame(value))
                
    print('Done!')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Phylogeny Task')

    parser.add_argument('--dataset', type=str, default='Plant', choices=["Plant", "Bordetella"], required=True, help='Set sequences')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--exp_name', type=str, default='test', required=False)
    parser.add_argument('--return_markers', action='store_true', help='Return tag in results')
    parser.add_argument('--plot', action='store_true', help='Plot confusion matrix')
    parser.add_argument('--data_path', type=str, default='./datasets', help='Path for data')
    parser.add_argument('--model_path', type=str, default='./checkpoints', help='Path for model')
    parser.add_argument('--result_path', type=str, default='./results', help='Path for saving results')
    
    args = parser.parse_args()
    
    main(args)
