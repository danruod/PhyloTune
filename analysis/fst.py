import numpy as np
from analysis.hetero import calculate_heter_s
from tqdm import tqdm

################### F_ST ###################  

def calculate_fst_by_attn(seqs, ignore_gap=False, gap_thres=1.0):
    fst_values = []
    
    seqs_class = seqs[:int(seqs.shape[0] / 2), :]
    seqs_rest = seqs[int(seqs.shape[0] / 2):, :]
    
    for i in tqdm(range(seqs.shape[1]), desc='Calculating Fst'):
        if (seqs[:, i] == '-').sum() / seqs.shape[0] <= gap_thres:    
        
            heter_s_class = calculate_heter_s(seqs_class[:, i], ignore_gap=ignore_gap)
            heter_s_reset = calculate_heter_s(seqs_rest[:, i], ignore_gap=ignore_gap)
        
            heter_s_avg = (heter_s_class + heter_s_reset) / 2
            
            heter_t = calculate_heter_s(seqs[:, i])

            fst = 1 - heter_s_avg / heter_t if heter_t != 0 else 0
            fst_values.append(fst)
        
    fst_values = np.asarray(fst_values).T
        
    return fst_values
