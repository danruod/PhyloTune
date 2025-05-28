import os
import argparse
import random
import numpy as np
import pandas as pd

import torch
import transformers
from transformers import BertModel, BertTokenizer

from model.probe import *
from model.bert_layers import BertForHierarchicalClassification

from utils.cuda import get_max_available_gpu
from utils.dataset import SequenceDatasetMultiRank, HierarchicalDataset, seq2kmers
from utils.predict import pred_for_seqs, calculate_attention

import warnings
warnings.filterwarnings('ignore')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def identify_taxonomy(args, labels_maps, tax_ranks, model, seq_ipt):
    outputs = pred_for_seqs(model, seq_ipt)
    
    unknown = outputs['novelty_score'] > args.threshold
    
    select = torch.vstack([unknown[:, i] != unknown[:, i+1] for i in range(unknown.shape[1] - 1)]).T
    
    tax_units = []
    for i in range(unknown.shape[0]):
        idx = torch.where(select[i] == True)[0]
        if len(idx) == 0:
            if unknown[i, 0] == True:
                level_id = -100
                tax_units.append(['outgroup', 'outgroup'])
            else:
                level_id = -1
                level = tax_ranks[-1]
                label = outputs['pred_label'][i, -1].item()
                try:
                    tax_units.append([level, labels_maps[level][label]])
                except:
                    # label is value, find the key
                    tax_units.append([level, next(key for key, value in labels_maps[level].items() if value == label)])
        else:
            level_id = min(idx).item()
            level = tax_ranks[min(idx).item()]
            label = outputs['pred_label'][i, min(idx).item()].item()
            try:
                tax_units.append([level, labels_maps[level][label]])
            except:
                # label is value, find the key
                tax_units.append([level, next(key for key, value in labels_maps[level].items() if value == label)])
        
        print(f'****** Result for seq {i} ******')
        print(f'# taxonomic unit: {tax_units[-1]}')
        if level_id == -100:
            print(f'# novelty scores at {tax_ranks[0]}: {outputs["novelty_score"][i][0]}')
        else:
            if level_id == -1:
                print(f'# novelty scores at {level}: {outputs["novelty_score"][i][-1]}')
            else:
                print(f'# novelty scores at {level} and {tax_ranks[min(idx).item()+1]}: {outputs["novelty_score"][i][level_id:level_id+2]}')
            print(f'# max probability for classification at {level}: {outputs["max_prob"][i][level_id]}')
        print('-----------------------------')
    return tax_units
    

def aggregate_tax_units(tax_ranks, tax_units, taxo_structure):
    final_tax_units = {}
    new_seqs_ids = {}
    if ['outgroup', 'outgroup'] in tax_units:
        final_tax_units[0] = ['outgroup', 'outgroup']
        new_seqs_ids[0] = list(range(len(tax_units)))
    else:
        j = 0
        for i, tax_unit in enumerate(tax_units):
            taxo_info = taxo_structure[taxo_structure[tax_unit[0]] == tax_unit[1]]
            for tax_rank in tax_ranks:
                if tax_unit[0] != tax_rank and taxo_info[tax_rank].values[0] in set(np.array(tax_units)[:,-1]):
                    if [tax_rank, taxo_info[tax_rank].values[0]] not in final_tax_units.values():
                        final_tax_units[j] = [tax_rank, taxo_info[tax_rank].values[0]]
                        new_seqs_ids[j] = [i]
                        j += 1
                    else:
                        idx = final_tax_units.index([tax_rank, taxo_info[tax_rank].values[0]])
                        new_seqs_ids[idx].append(i)
                    break
                if tax_unit[0] == tax_rank:
                    if tax_unit not in final_tax_units.values():
                        final_tax_units[j] = tax_unit
                        new_seqs_ids[j] = [i]
                        j += 1
                    else:
                        idx = list(final_tax_units.values()).index(tax_unit)
                        new_seqs_ids[idx].append(i)
                    break

    print(f'****** Final taxonomic units for all seqs ******')
    for key, value in final_tax_units.items():
        print(f'{key}: {value}')
        print(f'seq ids: {new_seqs_ids[key]}')
    
    return final_tax_units, new_seqs_ids


def extract_high_attention_regions(args, device, model, tokenizer, original_seqs_set, final_tax_units, new_seqs_ids):        
    for key, tax_unit in final_tax_units.items():
        print(f'****** Taxonomic unit: {tax_unit} ******')
        
        if 'marker' in original_seqs_set.columns:
            seqs_info = original_seqs_set.query(f"marker == '{args.marker}'")
        elif 'tag' in original_seqs_set.columns:
            seqs_info = original_seqs_set.query(f"tag == '{args.marker}'")
        else:
            raise ValueError('Please check the column name for marker or tag.')
        
        if tax_unit[0] != 'outgroup':
            seqs_info = seqs_info[seqs_info[tax_unit[0]] == tax_unit[1]]
        
        seqs = list(seqs_info['seq'])
        print(f'new seqs: {new_seqs_ids[key]}')
        for i in new_seqs_ids[key]:
            seqs.append(args.seqs[i])
            
        print(f'# Number of sequences: {len(seqs)}')
        avg_scores = calculate_attention(args.dataset, model, tokenizer, device, seqs, num_segs=args.num_segs)
    
        # small -> large    
        sorted_idx = torch.argsort(torch.mean(avg_scores, dim=0))
        # large -> small
        sorted_idx = sorted_idx.flip(0)
        selected_idx = sorted_idx[:args.k]
        
        print(f'###### Sequence is divided into {args.num_segs} equal segments ######')
        if args.k == 1:
            print(f'# Area of ​​highest attention: {(selected_idx + 1).tolist()} ######')
        else:
            print(f'# Areas of top {args.k} highest attention: {(selected_idx + 1).tolist()} ######')
        
        print('-----------------------------')
        
        selected_idx = selected_idx.sort()[0].tolist()
        os.makedirs(args.result_path, exist_ok=True)
        with open(os.path.join(args.result_path, f'{tax_unit[1]}_{args.marker}_high_attn_{args.k}_{args.num_segs}.fasta'), 'w') as f:
            for i in new_seqs_ids[key]:
                seq = ''
                for idx in selected_idx:
                    seq += args.seqs[i][idx * len(args.seqs[i]) // args.num_segs : (idx + 1) * len(args.seqs[i]) // args.num_segs]
                f.write(f'>{tax_unit[1]}_new\n{seq}\n')
                
            # read ech row for seqs_info (pandas)
            for row in seqs_info.iterrows():
                seq = ''
                for idx in selected_idx:
                    seq += row[1]['seq'][idx * len(row[1]['seq']) // args.num_segs : (idx + 1) * len(row[1]['seq']) // args.num_segs]
                # seq = row[1]['seq']
                # seq = seq[max_idx * len(seq) // args.num_segs : (max_idx + 1) * len(seq) // args.num_segs]
                
                if args.dataset == 'Plant':
                    name = row[1][tax_unit[0]]
                    species = row[1]['species']                
                    taxid = row[1]['taxid']
                    
                    f.write(f'>{name}_{species}_{taxid}\n{seq}\n')
                elif args.dataset == 'Bordetella':
                    species = row[1]['species']
                    individual = row[1]['individual']
                    f.write(f'>{species}_{individual}\n{seq}\n')
                
                else:
                    raise ValueError('Please check the dataset name.')


def main(args):
    set_seed(0)
    
    default_dtype = torch.float64
    torch.set_default_dtype(default_dtype)
    
    assert args.k <= args.num_segs, f'k should be less than or equal to num_segs.'
    
    args.result_path = os.path.join(args.result_path, f'{args.dataset}/bertphylo')
    
    if torch.cuda.is_available():
        try:
            device_id, _ = get_max_available_gpu()
        except:
            device_id = 0
        print(f"Using GPU: {device_id}")
    else:
        print("Using CPU")
    
    device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    if args.dataset == 'Plant':
        
        data_path = os.path.join(args.data_path, f'{args.dataset}/1_train_test')
        bert_path = os.path.join(args.model_path, 'Plant_dnabert/bert')
        probe_path = os.path.join(args.model_path, 'Plant_dnabert/hlps.pt')
        
        original_seqs_set = pd.read_csv(os.path.join(data_path, "id_train.tsv"), sep='\t')
        
        assert args.marker in list(original_seqs_set['tag'].unique()), f'The marker only support {list(original_seqs_set["tag"].unique())}'
    
        # load taxonomic hierarchy
        print(f'# Loading taxonomic hierarchy of the Plant... ')
            
        # get labels_maps and tax_ranks
        labels_maps = torch.load(os.path.join(data_path, 'labels_maps.pt'))
        tax_ranks = list(labels_maps.keys())

        print(f'------ Label information -----')
        for tax_rank in tax_ranks:
            print(f'{tax_rank}: {list(labels_maps[tax_rank].values())}')
        print('---------------')
        
        # load tokenizer
        tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
        
        # load model
        print(f'# Loading BERTPhylo .... ')
        ## load BERT module
        bert = BertModel.from_pretrained(bert_path)
        bert.to(device)
        
        ## load hierarchical linear probes
        hlp_config_dict = {'device': device, 
                           'default_dtype': default_dtype,
                           'tax_ranks': tax_ranks,
                           'labels_maps': labels_maps,
                           } 
        hlp = LinearProbeMultiRank(**hlp_config_dict)
        hlp.to(device)
        hlp.load_state_dict(torch.load(probe_path, map_location=device))
        
        model = (bert, hlp)
        
        for tax_rank in tax_ranks:
            original_seqs_set[tax_rank] = original_seqs_set[tax_rank].apply(lambda x: x.split("'")[1] if "'" in x else x)
        original_seqs_set['species'] = original_seqs_set['species'].apply(lambda x: x.split("'")[1] if "'" in x else x)
        
        taxo_structure = original_seqs_set[tax_ranks].drop_duplicates()
        
        if len(args.seqs) > 0:
            seq_kmers = []
            for seq in args.seqs:
                seq_kmers.append(seq2kmers(seq))
                
            seq_ipt = tokenizer(seq_kmers, padding=True, truncation=True, max_length=512, return_tensors="pt",)

        if args.threshold is None:
            args.threshold = 0.1
            
    elif args.dataset == 'Bordetella':
        
        data_path = os.path.join(args.data_path, f'{args.dataset}/1_train_test')
        model_path = os.path.join(args.model_path, 'Bordetella_dnaberts')
        
        original_seqs_set = pd.read_csv(os.path.join(data_path, "train.csv"))
        assert args.marker in list(original_seqs_set['marker'].unique()), f'The marker only support {list(original_seqs_set["marker"].unique())}'
        
        # load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=5000,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
  
        # load model
        labels_maps = torch.load(os.path.join(data_path, "labels_maps.pt"))
        tax_ranks = list(labels_maps.keys())
        model = BertForHierarchicalClassification.from_pretrained(
            model_path,
            cache_dir=None,
            trust_remote_code=True,
            labels_maps=labels_maps,
        ).to(device)

        taxo_structure = original_seqs_set[tax_ranks].drop_duplicates()
        
        if len(args.seqs) > 0:
            seq_ipt = tokenizer.batch_encode_plus(
                args.seqs, 
                max_length=5000, 
                return_tensors='pt', 
                padding='longest', 
                truncation=True
            )
        
        if args.threshold is None:
            args.threshold = 0.7
        
    else:
        raise ValueError('Please check the dataset name.')
    
    if n_gpu > 1:
        model = nn.DataParallel(model)
                        
    if len(args.seqs) > 0:
        # identify smallest taxonomic units
        print(f'############# Identify the smallest taxonomic units #############')
        tax_units = identify_taxonomy(args, labels_maps, tax_ranks, model, seq_ipt)

        print(f'****** Merge overlapping clades ..... ')
        final_tax_units, new_seqs_ids = aggregate_tax_units(tax_ranks, tax_units, taxo_structure)

    # Extract high-attention regions
    print(f'############# Extract high-attention regions #############')
  
    extract_high_attention_regions(args, device, model, tokenizer, original_seqs_set, final_tax_units, new_seqs_ids)    
            
    print(f'############# Task completed #############')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Phylogeny Task')

    parser.add_argument('--dataset', type=str, default='Plant', required=True, help='Set sequences')
    parser.add_argument('--seqs', type=str, nargs='+', required=True, help='Set sequences')
    parser.add_argument('--marker', type=str, required=True, help='Set marker')
    parser.add_argument('--threshold', type=float, help='Set threshold for novelty score')
    parser.add_argument('--num_segs', type=int, default=3, help='Set number of segments')
    parser.add_argument('--k', type=int, default=1, help='Set k segments with high attention scores')
    parser.add_argument('--data_path', type=str, default='./datasets', help='Path for data')
    parser.add_argument('--model_path', type=str, default='./checkpoints', help='Path for model')
    parser.add_argument('--result_path', type=str, default='./results', help='Path for saving results')
                        
    args = parser.parse_args()
    
    main(args)
