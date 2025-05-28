import numpy as np
from Bio.SeqIO import parse
import pandas as pd
import os
from pathlib import Path

valid_bases = {'A', 'T', 'C', 'G'}


def seq2kmers(seq, k=3, stride=3, pad=True, to_upper=True):
    """transforms sequence to k-mer sequence.
    If specified, end will be padded so no character is lost"""
    if (k == 1 and stride == 1):
        # for performance reasons
        return seq
    kmers = str()
    for i in range(0, len(seq) - k + 1, stride):
        kmer = seq[i:i+k]
        if to_upper:
            kmers += kmer.upper()
        else:
            kmers += kmer
        if i + k < len(seq):
            kmers += ' '
    if (pad and len(seq) - (i + k)) % k != 0:
        kmers += ' ' + seq[i+k:].ljust(k, 'N')
    return kmers


def read_fasta(file, return_array=False):
    original_seqs = []
    seq_name_list = []
    length = []

    
    # if file is directory, read each file 
    if Path(file).is_dir():
        for file_name in os.listdir(file):
            if file_name.endswith(".fasta"):
                for r in parse(open(Path(file) / file_name), 'fasta'):
                    if return_array:
                        original_seqs.append(list(str(r.seq)))
                    else:
                        original_seqs.append(str(r.seq))
                    seq_name_list.append(f'{str(r.id)}-{file_name.split(".")[0]}')
                    
                    length.append(len(str(r.seq)))
    else:
        for r in parse(open(file), 'fasta'):
            if return_array:
                original_seqs.append(list(str(r.seq)))
            else:
                original_seqs.append(str(r.seq))
            seq_name_list.append(str(r.id.split(".fasta")[0]))
            
            length.append(len(str(r.seq)))
    
    if return_array:
        original_seqs = np.array(original_seqs)
        
    return original_seqs, length, seq_name_list


def read_fasta_by_filter(file, kmer, stride, filter_name=None, return_kmer=True, return_array=False):
    original_seqs = []
    seqs = []
    seq_name_list = []
    length = []
    num_kmers = []
    for r in parse(open(file), 'fasta'):
        if filter_name is not None and filter_name not in str(r.id):
            continue
        
        if return_array:
            original_seqs.append(list(str(r.seq)))
        else:
            original_seqs.append(str(r.seq))
        seq_name_list.append(str(r.id))
        if return_kmer:
            seq_kmer, num_kmer = seq2kmers(str(r.seq), k=kmer, stride=stride)
            seqs.append(seq_kmer)
            num_kmers.append(num_kmer)
        
        length.append(len(str(r.seq)))
    
    if return_array:
        original_seqs = np.array(original_seqs)
        
    # print("length:", length)
    if return_kmer:
        # print("num_kmers:", num_kmers)
        return seqs, original_seqs, length, seq_name_list
    else:
        return original_seqs, length, seq_name_list


def get_tsv_dataset(file_path):
    pd_dataframe = pd.read_csv(file_path, sep='\t')
        
    if "Unnamed: 0" in pd_dataframe.columns:
        pd_dataframe = pd_dataframe.drop(columns=["Unnamed: 0"])
    
    return pd_dataframe


