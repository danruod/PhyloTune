import numpy as np
from analysis.utils import valid_bases
from tqdm import tqdm

################### Heterozygosity ###################       

def calculate_heter_s(seq, ignore_gap=False):
    seq_upper = np.char.upper(seq)
    if ignore_gap:
        seq_upper = seq_upper[np.isin(seq_upper, list(valid_bases))]
    
    _, count = np.unique(seq_upper, return_counts=True)
    heter_s = 1 - np.sum((count / np.sum(count)) ** 2) if count.size > 0 else 0
    
    # for nan and inf
    heter_s = 0 if np.isnan(heter_s) or np.isinf(heter_s) else heter_s
    
    return heter_s


def calculate_hetero(seqs, ignore_gap=False):
    hetero_values = []
    for i in tqdm(range(seqs.shape[1]), desc='Calculating heterozygosity'):
        hetero = calculate_heter_s(seqs[:, i], ignore_gap=ignore_gap)
        hetero_values.append(hetero)

    return np.asarray(hetero_values).T

