import torch
import argparse
import os
import numpy as np
import scipy
import pandas as pd

import transformers
from transformers import BertTokenizer, BertModel

from model.probe import *
from model.bert_layers import BertForHierarchicalClassification

from analysis.attn import rearrange_attn_score
from analysis.hetero import calculate_hetero
from analysis.fst import calculate_fst_by_attn
from analysis.dxy import calculate_dxy_by_attn
from analysis.sub_rate import diff_bases, calculate_sub_rate
from analysis.utils import read_fasta

from utils.predict import calculate_attention
from utils.plot import plot_attn_analysis


def get_scores(scores, seqs, seq_name_list, align_seqs, align_seq_name_list, con_seq, args, result_path, marker):
    results_dir = f"{result_path}/{marker}"
    os.makedirs(results_dir, exist_ok=True)
    results_dict = {}
    recalc_attn = False
    recalc_pearson = False
    
    for metric in args.metrics:                
        file_name = f'{results_dir}/{metric}-64.csv'
            
        if not os.path.exists(file_name):
            recalc_attn = True
            break
    
    if recalc_attn or not os.path.exists(f'{results_dir}/attn_scores.csv') or not os.path.exists(f'{results_dir}/ave_scores.csv'):
        print("### Calculate avgerate attention scores ###")
        # rearrange attn based on aligned seqs   
        attn_scores, valid_bases, seq_name, filter_align_seqs, filter_con_seq = rearrange_attn_score(scores, seqs, seq_name_list, 
                                                                                                     align_seqs, align_seq_name_list, 
                                                                                                     con_seq, gap_thres=args.gap_thres, 
                                                                                                     attn_sort=True)
        
        if args.ignore_gap:
            ave_scores = attn_scores.sum(0) / valid_bases.sum(0)
        else:
            ave_scores = attn_scores.mean(0)
        
        results_dict['attn'] = ave_scores
        
        pd.DataFrame(attn_scores, index=seq_name).to_csv(f'{results_dir}/attn_scores.csv')
        pd.DataFrame(ave_scores).to_csv(f'{results_dir}/ave_scores.csv')
        
    else:
        attn_scores = pd.read_csv(f'{results_dir}/attn_scores.csv', index_col=0).values
        results_dict['attn'] = pd.read_csv(f'{results_dir}/ave_scores.csv').iloc[:, 1:].values.reshape(-1)

    # calculate heterozygosity, fst, dxy, sub_rate
    for metric in args.metrics:                
        file_name = f'{results_dir}/{metric}-64.csv'
            
        if os.path.exists(file_name):
            print(f"### Load {metric} from file {file_name}")
            results = pd.read_csv(file_name).iloc[:, 1:]
            if results.shape[-1] == 1:
                results_dict[metric] = results.values.reshape(-1)
            else:
                # to {column1: [value], column2: [value]}
                results_dict[metric] = results.to_dict(orient='list')
        else:
            print(f"### Calculate {metric}")
            if metric == 'hetero':
                results_dict[metric] = calculate_hetero(filter_align_seqs, ignore_gap=args.ignore_gap)
            elif metric == 'fst':
                results_dict[metric] = calculate_fst_by_attn(filter_align_seqs, ignore_gap=args.ignore_gap)
            elif metric == 'dxy':
                results_dict[metric] = calculate_dxy_by_attn(filter_align_seqs, filter_con_seq, use_condition=False, ignore_gap=args.ignore_gap)
            elif metric == 'sub_rate':
                if os.path.exists(f'{results_dir}/values.npz'):
                    print("Loading diff_values and valid_values ...")                    
                    values = np.load(f'{results_dir}/values.npz', allow_pickle=False)
                    diff_values = values['diff_values']
                    valid_values = values['valid_values']
                else:
                    print("Calculating diff_values and valid_values ...")
                    # the nucleatide different per site between two sequences. similar to hamming distance
                    # diff_values/total_values: valid_length * #seq * #seq4
                    diff_values, valid_values = diff_bases(filter_align_seqs)
                    
                    # store _values, valid_values
                    np.savez_compressed(f'{results_dir}/values.npz', diff_values=diff_values, valid_values=valid_values, allow_pickle=False)
                    # np.save(f'{results_dir}/diff_values.npy', diff_values)
                    # np.save(f'{results_dir}/valid_values.npy', valid_values)
                    
                # remove the compare with unvalid_bases 
                diff_values = diff_values * valid_values
                results_dict[metric] = calculate_sub_rate(diff_values, valid_values, window=10, step=1, ignore_gap=args.ignore_gap)
            else:
                raise ValueError(f"Unknown metric: {metric}")   
                
            pd.DataFrame(results_dict[metric]).to_csv(file_name)
            recalc_pearson = True

    recalc_pearson = True
     
    if not recalc_pearson and os.path.exists(f'{results_dir}/pearson_metric.csv'):
        print(f"### Load pearson_metric from file {results_dir}/pearson_metric.csv")
        pearson_metric = pd.read_csv(f'{results_dir}/pearson_metric.csv').drop(columns=['Unnamed: 0'])
    else:
        print("### Calculate pearson_metric")
        pearson_metric = []
        results_keys = list(results_dict.keys())
        # results_keys = ['attn', "dxy", "fst", "hetero", "sub_rate"]
        for i, key in enumerate(results_keys):
            value = results_dict[key]
            # if value is dict
            if isinstance(value, dict):
                for key1, value1 in value.items():
                    for key2 in results_keys[i+1:]:
                        value2 = results_dict[key2]
                        if isinstance(value2, dict):
                            for key3, value3 in value2.items():
                                pearson_result = scipy.stats.pearsonr(value1, value3)
                                pearson_metric.append([marker, f'{key}-{key1}', f'{key2}-{key3}', pearson_result[0], pearson_result[1]])
                        else:
                            pearson_result = scipy.stats.pearsonr(value1, value2)
                            pearson_metric.append([marker, f'{key}-{key1}', key2, pearson_result[0], pearson_result[1]])
            
            else:
                for key2 in results_keys[i+1:]:
                    value2 = results_dict[key2]
                    if isinstance(value2, dict):
                        for key3, value3 in value2.items():
                            pearson_result = scipy.stats.pearsonr(value, value3)
                            pearson_metric.append([marker, key, f'{key2}-{key3}', pearson_result[0], pearson_result[1]])
                    else:
                        pearson_result = scipy.stats.pearsonr(value, value2)
                        pearson_metric.append([marker, key, key2, pearson_result[0], pearson_result[1]])
   
        pearson_metric = pd.DataFrame(pearson_metric, columns=['marker', 'key1', 'key2', 'coeff', 'pvalue'])
        pearson_metric.to_csv(f'{results_dir}/pearson_metric-64.csv')
        
        print("pearson_metric", pd.DataFrame(pearson_metric))
                 
    return attn_scores, results_dict, pearson_metric


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    result_path = os.path.join(args.result_path, f'{args.dataset}/attn_analysis')
    
    attention_dir = f"{result_path}/attention-64"
    os.makedirs(attention_dir, exist_ok=True)
        
    save_dir = f"{result_path}/gap{args.gap_thres}"
    if args.ignore_gap:
        save_dir = f"{save_dir}_ATCG" 
    os.makedirs(save_dir, exist_ok=True)
    
    if args.plot:
        plot_dir = f"{save_dir}/plot"
        os.makedirs(plot_dir, exist_ok=True)
        
    if args.dataset == 'Plant':
        
        data_path = os.path.join(args.data_path, f'{args.dataset}/2_attn_analysis')
        bert_path = os.path.join(args.model_path, 'Plant_dnabert/bert')
        probe_path = os.path.join(args.model_path, 'Plant_dnabert/hlps.pt')
        
        labels_maps = torch.load(os.path.join(args.data_path, f'{args.dataset}/1_train_test/labels_maps.pt'))
        tax_ranks = list(labels_maps.keys())
        
        # load tokenizer
        tokenizer = BertTokenizer.from_pretrained(bert_path, do_lower_case=False)
        
        # load model
        bert = BertModel.from_pretrained(bert_path)
        bert.to(device)
        hlp_config_dict = {'device': device, 
                           'default_dtype': bert.dtype,
                           'tax_ranks': tax_ranks,
                           'labels_maps': labels_maps,
                           } 
        hlp = LinearProbeMultiRank(**hlp_config_dict)
        hlp.to(device)
        hlp.load_state_dict(torch.load(probe_path, map_location=device))
        
        model = (bert, hlp)

    elif args.dataset == 'Bordetella':

        data_path = os.path.join(args.data_path, f'{args.dataset}/2_attn_analysis')
        model_path = os.path.join(args.model_path, 'Bordetella_dnaberts')
        
        # load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=5000,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
  
        # load model
        labels_maps = torch.load(os.path.join(args.data_path, f'{args.dataset}/1_train_test/labels_maps.pt'))
        tax_ranks = list(labels_maps.keys())
        model = BertForHierarchicalClassification.from_pretrained(
            model_path,
            cache_dir=None,
            trust_remote_code=True,
            labels_maps=labels_maps,
        ).to(device)
        
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    if args.metrics is None:
        args.metrics = ["sub_rate", "hetero", "fst", "dxy"]
    
    print(f"Metrics: {args.metrics}")
    print(f'Ignore loci with gaps more than {args.gap_thres}')
    print(f'markers: {[file_name.split(".fasta")[0] for file_name in os.listdir(f"{data_path}/original")]}')
    
    pearson_metric_all = []
    for file_name in os.listdir(f'{data_path}/original'):
        
        # if file_name == '28s.fasta':
        #     continue

        print(f"********** READ {file_name} **********")
        
        attention_file = os.path.join(attention_dir, file_name.replace(".fasta", ".csv"))
        
        if os.path.exists(attention_file):
            print(f"### Load attention score from {attention_file} ### ")
            # read scores_pd from file
            scores_pd = pd.read_csv(attention_file)
            # remove the first column with Unnamed
            if 'Unnamed: 0' in scores_pd.columns:
                scores_pd = scores_pd.drop(columns=['Unnamed: 0'])
            
            seq_name_list = scores_pd.get('seq_name').values.tolist()
            seqs = scores_pd.get('seq').values.tolist()
            scores = scores_pd.iloc[:, 2:].values
        else:

            ########## read seqs ##########
            # read original seqs to obtain attn
            seqs, _, seq_name_list = read_fasta(f'{data_path}/original/{file_name}')
                        
            assert len(np.unique(seq_name_list)) == len(seq_name_list)

            ########## get attention ##########
            scores = calculate_attention(args.dataset, model, tokenizer, device, seqs, 
                                         seq_name_list=seq_name_list, 
                                         attention_file=attention_file)
            
        # read aligned seqs to rearrange attn
        # Note: align seqs order may not be the same as seqs order in scores
        align_seqs, align_length, align_seq_name_list = read_fasta(f'{data_path}/align/{file_name}', 
                                                                    return_array=True)
        assert len(np.unique(align_seq_name_list)) == len(align_seq_name_list)
        assert len(np.unique(align_length)) == 1
        align_length = np.unique(align_length)[0] 
        
        # read consensus seq
        if os.path.exists(f'{data_path}/consensus/{file_name}'):
            con_seq, con_length, con_seq_name_list = read_fasta(f'{data_path}/consensus/{file_name}')
            assert len(con_length) == 1
            assert con_length[0] == align_length
            con_seq = con_seq[0]
            
        # './consensus.fasta' exists
        elif os.path.exists(f'{data_path}/consensus.fasta'):
            con_seqs, con_length, con_seq_name_list = read_fasta(f'{data_path}/consensus.fasta')
            
            if len(con_seqs) > 1:
                idx = con_seq_name_list.index(file_name.split(".fasta")[0])
            else:
                idx = 0
                
            con_seq = con_seqs[idx]
            assert con_length[idx] == align_length
                
        else:
            con_seq = None
        
        ########## get scores ##########
        marker = file_name.split('.fasta')[0]
        
        attn_scores, results_dict, pearson_metric = get_scores(scores, seqs, seq_name_list, align_seqs, align_seq_name_list, con_seq, args, save_dir, marker)
        pearson_metric_all.append(pearson_metric)
        
        if args.plot:
            plot_attn_analysis(attn_scores, results_dict, pearson_metric, plot_dir, marker)
    
    # merge pearson_metric 
    if len(pearson_metric_all) > 0:
        pd.concat(pearson_metric_all).to_csv(f'{save_dir}/pearson_results-64.csv')
        
        pearson_metric_all = pd.concat(pearson_metric_all)
        pearson_metric_avg = pearson_metric_all.groupby(['key1', 'key2']).agg({'coeff': ['mean', 'std'], 'pvalue': ['mean', 'std']}).reset_index()
        
        with pd.ExcelWriter( f"{save_dir}/pearson_results-64.xlsx") as writer:
            pearson_metric_all.to_excel(writer, sheet_name='all')
            pearson_metric_avg.to_excel(writer, sheet_name='avg')
            
    print("Done!")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Plant', required=True, help='Set sequences')
    parser.add_argument('--data_path', type=str, default='./datasets', help='Path for data')
    parser.add_argument('--model_path', type=str, default='./checkpoints', help='Path for model')
    parser.add_argument('--result_path', type=str, default='./results', help='Path for saving results')
    parser.add_argument("--gap_thres", default=0.9, type=float, help="filter loci with too much gap")
    parser.add_argument("--metrics", nargs='+', type=str,)
    parser.add_argument("--ignore_gap", action='store_true',)
    parser.add_argument("--plot", action='store_true',)
    
    args = parser.parse_args()
    
    main(args)