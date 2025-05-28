import numpy as np
from analysis.utils import valid_bases
from tqdm import tqdm

# diff -> the nucleatide different per site between two sequences. similar to hamming distance
# diff shape: seq_len * #seq * #seq
# valid -> The value is 1 only if the bases at the paired positions in both sequences are valid nucleotides.
# valid shape: seq_len * #seq * #seq

# def diff_bases(seq):
#     diff = np.zeros((seq.shape[1], seq.shape[0], seq.shape[0]))
#     valid = np.zeros((seq.shape[1], seq.shape[0], seq.shape[0]))
    
#     for i in range(seq.shape[0]):                      
#         for j in range(i+1, seq.shape[0]):
#             # seq[:, i] in valid_bases and seq[:, j] in valid_bases
#             valid[:, i, j] = np.array([base.upper() in valid_bases for base in seq[i, :]]) & np.array([base.upper() in valid_bases for base in seq[j, :]])
#             diff[:, i, j] = (np.char.upper(seq[i, :]) !=  np.char.upper(seq[j, :])).astype(int)

#     return diff, valid

def diff_bases(seq):
    seq_upper = np.char.upper(seq)
    
    seq_valid = np.zeros_like(seq_upper, dtype=bool)
    for i in range(seq.shape[0]):
        seq_valid[i] = np.array([base in valid_bases for base in seq_upper[i, :]])
    
    diff = np.zeros((seq.shape[1], seq.shape[0], seq.shape[0]))
    valid = np.zeros((seq.shape[1], seq.shape[0], seq.shape[0]))
    
    for i in tqdm(range(seq.shape[0])):                      
        for j in range(i+1, seq.shape[0]):
            # seq[:, i] in valid_bases and seq[:, j] in valid_bases
            valid[:, i, j] = seq_valid[i] & seq_valid[j]
            diff[:, i, j] = (seq_upper[i] != seq_upper[j]).astype(int)

    return diff, valid


def calculate_sub_rate(diff_values, valid_values, window=1, step=1, ignore_gap=False):
    if (step != 1) and (window != step):
        raise ValueError('step should be 1 if window is not equal to step')

    sub_rates = np.zeros(diff_values.shape[0])
    for i in tqdm(range(0, diff_values.shape[0]-window+1, step), desc='Calculating substitution rate'):
        diff_window = diff_values[i:i+window].sum(0)
        if ignore_gap:
            valid_window = valid_values[i:i+window].sum(0)
        else:
            valid_window = window
        
        # if the value of valid_window is 0, the value of sub_rate is 0
        p = np.nan_to_num(diff_window / valid_window, nan=0, posinf=0, neginf=0)
        p = np.clip(p, 0, 0.75)  # Ensure p is within the valid range
        sub_rate = - (3/4) * np.log(1 - (4/3) * p)
        sub_rate = np.nan_to_num(sub_rate, nan=0, posinf=0, neginf=0)
        sub_rates[i: i+step] = sub_rate.mean()
    
    sub_rates[i:] = sub_rate.mean()
    
    return sub_rates
