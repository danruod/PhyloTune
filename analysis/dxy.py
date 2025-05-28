import numpy as np
from analysis.utils import valid_bases
from tqdm import tqdm

################### D_XY ###################  

def calculate_dxy_by_attn(seqs, con_seq, use_condition=True, ignore_gap=False, gap_thres=1.0):
    assert seqs.shape[-1] == len(con_seq), f'seq.shape[-1] {seqs.shape[-1]} != len(con_seq) {len(con_seq)}'
    
    dxy_values = []
    
    seqs_class = seqs[:int(seqs.shape[0] / 2), :]
    seqs_rest = seqs[int(seqs.shape[0] / 2):, :]
    
    num_gap = 0
    seq = []
    for i in tqdm(range(seqs.shape[1]), desc='Calculating Dxy'):
        if (seqs[:, i] == '-').sum() / seqs.shape[0] <= gap_thres:
            seq.append(con_seq[i])
            
            if ignore_gap:
                if con_seq[i] not in list(valid_bases):
                    prob_class = 1
                    prob_reset = 1
                else:
                    # find the frequency of con_seq[i] in seq_class[:, i] and seq_rest[:, i]    
                    prob_class = (np.char.upper(seqs_class[:, i]) == con_seq[i].upper()).sum() / np.isin(np.char.upper(seqs_class[:, i]), list(valid_bases)).sum()
                    prob_reset = (np.char.upper(seqs_rest[:, i]) == con_seq[i].upper()).sum() / np.isin(np.char.upper(seqs_class[:, i]), list(valid_bases)).sum()
                    # for nan and inf
                    prob_class = 0 if np.isnan(prob_class) or np.isinf(prob_class) else prob_class
                    prob_reset = 0 if np.isnan(prob_reset) or np.isinf(prob_reset) else prob_reset
            else:
            
                if con_seq[i] == '-':
                    num_gap += 1
                
                # find the frequency of con_seq[i] in seq_class[:, i] and seq_rest[:, i]    
                prob_class = (np.char.upper(seqs_class[:, i]) == con_seq[i].upper()).mean()
                prob_reset = (np.char.upper(seqs_rest[:, i]) == con_seq[i].upper()).mean()

            if use_condition:
                class_unique = np.unique(np.char.upper(seqs_class[:, i]))
                rest_unique = np.unique(np.char.upper(seqs_rest[:, i]))
                
                if (class_unique[:2] == rest_unique[:2]).all():
                    dxy = prob_class * (1 - prob_reset) + prob_reset * (1 - prob_class)
                else:
                    dxy = prob_class * prob_reset +  (1 - prob_class) * (1 - prob_reset)
            else:
                dxy = prob_class * (1 - prob_reset) + prob_reset * (1 - prob_class)
            dxy_values.append(dxy)
        
    dxy_values = np.asarray(dxy_values).T
    
    print(f'num_gap: {num_gap}')
        
    return dxy_values
