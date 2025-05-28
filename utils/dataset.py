import torch
import os
import json
import logging
import pandas as pd
import numpy as np

import transformers
from torch.utils.data import Dataset
from typing import Dict, List

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.DEBUG)


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


"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, seqs: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in seqs]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer


class SequenceDatasetMultiRank(torch.utils.data.Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, tax_ranks, labels_id, labels_name, 
                 labels_taxid, labels_maps, seq_tag):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.tax_ranks = tax_ranks
        self.labels_id = labels_id
        self.labels_name = labels_name
        self.labels_taxid = labels_taxid
        self.labels_maps = labels_maps
        self.seq_tag = seq_tag
        
    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        token_type_id = self.token_type_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.labels_id[idx]
        
        tag = self.seq_tag[idx]
        return input_id, token_type_id, attention_mask, label, tag
    
    def __len__(self):
        return len(self.labels_id)
    

class HierarchicalDataset(Dataset):
    """HierarchicalDataset for supervised fine-tuning

        :param data_path: CSV/TSV file path
        :param input_column: column name where the seq is located
        :param label_columns: class label column names, arranged from coarse to fine
        :param delimiter: file delimiter (default is comma, suitable for CSV files, TSV files use `\t`)
        """

    def __init__(self, 
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1,
                 input_col: str = 'seq',
                 label_cols: list = ['species'],
                 labels_maps: dict = None,
                 ood_id: int = -100,
                 balance: bool = False,
                 shuffle: bool = False,
                 sample_num: int = 50):

        super(HierarchicalDataset, self).__init__()

        # load data from the disk
        if isinstance(data_path, list):
            data = pd.concat([pd.read_csv(path, sep='\t') if path.endswith(".tsv") else pd.read_csv(path) for path in data_path])
        else:
            data = pd.read_csv(data_path, sep='\t') if data_path.endswith(".tsv") else pd.read_csv(data_path)    

        # shuffle the data, and reset the index
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
            
        assert input_col in data.columns, f"Input column {input_col} not found in the data."
        assert all([col in data.columns for col in label_cols]), f"Label columns {label_cols} not found in the data."
        
        if balance:
            sample_num = min(data[label_cols].value_counts().min(), sample_num)
            logger.info(f"Balancing the data with {sample_num} samples for each class.")
            data = data.groupby(label_cols).apply(lambda x: x.sample(sample_num)).reset_index(drop=True)
        
        seqs = list(data[input_col])
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            seqs = load_or_generate_kmer(data_path, seqs, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            seqs,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
    
        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.token_type_ids = output["token_type_ids"] if "token_type_ids" in output else None
        
        if labels_maps is None:
            self.labels_maps = {col: {label: idx for idx, label in enumerate(data[col].unique())} for col in label_cols}
        else:
            # if labels_maps.values is not ascending, adapt labels_maps
            self.labels_maps = dict()
            for col in label_cols:
                label_map = labels_maps[col]
                if list(label_map.values()) != list(range(len(label_map))):
                    # find key in label_map based on value
                    self.labels_maps[col] = {list(label_map.keys())[list(label_map.values()).index(idx)]: idx for idx in range(len(label_map))}
                else:
                    self.labels_maps[col] = label_map
            
            self.labels_maps = labels_maps

        # for label not in self.labels_maps, return ood_id
        self.labels = data[label_cols].apply(lambda row: [self.labels_maps[col].get(row[col], ood_id) for col in label_cols], axis=1).tolist()
        
        # Calculate label weights, exclude ood_id
        self.label_weight = []
        for col in label_cols:
            num_classes = 0
            label_samples = []
            for label in self.labels_maps[col].keys():
                if label in data[col].value_counts():
                    if data[col].value_counts().get(label) > 0:
                        num_classes += 1
                        label_samples.append(data[col].value_counts().get(label))
                    else:
                        logger.warning(f"No samples found for {label} in {col}.")
                        label_samples.append(1e-10)
            
            if num_classes == 0:
                raise ValueError(f"No in-distritbuion label found in {col}, please check the labels_maps.")
            
            self.label_weight.append((sum(label_samples) / num_classes) / label_samples)
            
            # print label_weight
            logger.info(f"Label weight for {col}: {self.label_weight[-1]}")
        
        self.ood_id = ood_id
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # return input_ids, attention_mask, token_type_ids, labels
        return dict(input_ids=self.input_ids[i], 
                    attention_mask=self.attention_mask[i], 
                    token_type_ids=self.token_type_ids[i] if self.token_type_ids is not None else None,
                    labels=self.labels[i])
    
    def print_info(self):
        # print #seq for each label based on the self.labels
        label_count = {}
        for i, col in enumerate(self.labels_maps.keys()):
            unique, counts = np.unique(np.array(self.labels)[:, i], return_counts=True)
            seqs_count = dict(zip(unique, counts))
            for name, value in self.labels_maps[col].items():
                if value in list(seqs_count.keys()):
                    label_count[name] = {'count': seqs_count.get(value, 0), 'level': col}
            if self.ood_id in list(seqs_count.keys()):
                label_count[f'ood-{col}'] = {'count': seqs_count.get(self.ood_id, 0), 'level': col}
        logger.info(pd.DataFrame(label_count).T)
        
        return
