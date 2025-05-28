import torch
import numpy as np
import pandas as pd
import torch.utils.data as util_data
import tqdm

from utils.dataset import seq2kmers

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        # squeezed.append(layer_attention.squeeze(0).detach().cpu())
        squeezed.append(layer_attention.detach().cpu())
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)


def get_attention_for_batch(model, input_ids, attention_mask, token_type_ids, device):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
            token_type_ids=token_type_ids.to(device),
            output_hidden_states=False,
            )
        attention = outputs[-1]
    
    return attention


def get_attention_dnabert(model, tokenizer, device, seqs, batch_size=128):    
    model.eval()
    
    seqs = []
    for raw_seq in seqs:
        seqs.append(seq2kmers(raw_seq))
            
    inputs = tokenizer(seqs, padding=True, truncation=True, max_length=512, return_tensors="pt",)
    
    attn_score_list = []
    
    for i in range(0, len(seqs), batch_size):
        # print(i)
        
        input_ids = inputs['input_ids'][i:min(i+batch_size, len(seqs))].to(device)
        attention_mask = inputs['attention_mask'][i:min(i+batch_size, len(seqs))].to(device)
        token_type_ids = inputs['token_type_ids'][i:min(i+batch_size, len(seqs))].to(device)
        
        attention = get_attention_for_batch(model, input_ids, attention_mask, token_type_ids, device)
        
        # layer * batch_size * head * seq_len * seq_len
        attn = format_attention(attention).detach().cpu()
        
        # batch_size * seq_len
        attn_score = attn[9,:,:,0,1:-1].sum(0).sum(1)
        attn_score_list.append(attn_score)
            
    # num_seqs * seq_len
    attn_score = torch.vstack(attn_score_list)
       
    return attn_score


def get_attention_dnaberts(model, tokenizer, dna_sequences, model_max_length=5000, batch_size=64):
    # reorder the sequences by length
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    all_attn_scores = []

    train_loader = util_data.DataLoader(dna_sequences, batch_size=batch_size*n_gpu, shuffle=False, num_workers=2*n_gpu)
    for batch in tqdm.tqdm(train_loader):
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

            model_outputs = model.forward(input_ids=input_ids, 
                                            attention_mask=attention_mask)
            model_attention = model_outputs[-1][-1].detach().cpu()
            
            # get special tokens
            special_tokens = tokenizer.special_tokens_map
            
            # get vocab with key as the id and value as the token
            token_to_id = tokenizer.get_vocab()
            id_to_token_length = {v: 0 if k in special_tokens.values() else len(k) for k, v in token_to_id.items()}
            id_to_token_length[0] = 1 # [UNK] token
            
            model_attention = model_attention[:,:,0,:].sum(1)
            for i, seq_ids in enumerate(input_ids):
                seq_ids = seq_ids.detach().cpu().numpy()
                attn_scores = []
                for s, id in enumerate(seq_ids):
                    attn_scores += [model_attention[i, s].item()] * id_to_token_length[id]
                
                all_attn_scores.append(attn_scores)
                
        all_attn_scores = [np.array(attn_scores) for attn_scores in all_attn_scores]
        
        scores = np.zeros([len(all_attn_scores), max(lengths)])
        for i in range(len(all_attn_scores)):
            scores[i][:len(all_attn_scores[i])] = all_attn_scores[i]
        
    return scores


def get_real_score(attention, kmer, stride):
    counts = np.zeros([attention.shape[1]*stride+kmer-stride])
    real_scores = np.zeros([attention.shape[0], attention.shape[1]*stride+kmer-stride])

    for i in range(attention.shape[1]):
        idx = i*stride
        for j in range(kmer):
            counts[idx+j] += 1.0
            real_scores[:, idx+j] += attention[:, i]

    real_scores = real_scores/counts

    return real_scores


def rearrange_attn_score(scores, seqs, seq_name_list, align_seqs, align_seq_name_list, 
                         con_seqs, gap_thres=1.0, attn_sort=False):
    """
    Rearrange the attention scores to match the alignment sequences.
    
    gap_thres: the threshold to filter out the columns with too many gaps.
    Attn_sort: sort the sequences based on the attention scores.
    """
    assert scores.shape[0] == len(seq_name_list), f'wrong input'
    assert len(seq_name_list) == len(align_seq_name_list), f'wrong input'
    assert scores.shape[0] == align_seqs.shape[0], f'wrong input'
    
    if len(align_seqs.shape) == 2:
        align_scores = np.zeros_like(align_seqs, dtype=float)
        valid_bases = np.zeros_like(align_seqs, dtype=int)
    else:
        align_scores = np.zeros([align_seqs.shape[0], len(align_seqs[0])])
        valid_bases = np.zeros([align_seqs.shape[0], len(align_seqs[0])])
        
    seqs = np.asarray(seqs)
    align_index = []
    for i, seq_name in enumerate(seq_name_list):
        k = 0
        idx = align_seq_name_list.index(seq_name)
        align_index.append(idx)
        for j, s in enumerate(align_seqs[idx]):
            if s != '-':
                if s.upper() != seqs[i][k].upper():
                    print(f"seq_name: {seq_name}, idx: {idx}, j: {j}, s: {s}, seqs[i, k]: {seqs[i][k]}")
                
                align_scores[i, j] = scores[i, k]
                valid_bases[i, j] = 1
                k += 1
    
    # sort align_seqs to be same as seq_name_list            
    align_seqs = align_seqs[align_index]         
    assert list(np.asarray(align_seq_name_list)[align_index]) == seq_name_list, f'wrong input'
    
    # remove the columns with too many gaps
    filter_align_seqs = []
    filter_con_seqs = [] if con_seqs is not None else None
    if gap_thres < 1.0:
        scores = []
        valids = []
        for i in range(align_scores.shape[1]):
            if (align_seqs[:, i] == '-').sum() / align_seqs.shape[0] <= gap_thres:
                scores.append(align_scores[:, i])
                valids.append(valid_bases[:, i])
                filter_align_seqs.append(align_seqs[:, i])
                if con_seqs is not None:
                    filter_con_seqs.append(con_seqs[i])
        
        align_scores = np.asarray(scores).T
        valid_bases = np.asarray(valids).T
        filter_align_seqs = np.asarray(filter_align_seqs).T
    else:
        filter_align_seqs = np.asarray(align_seqs)
        
    # sort the sequences based on the attention scores
    if attn_sort:            
        order_attn = np.zeros([align_scores.shape[0], 2])
        order_attn[np.argsort(align_scores[:, :int(align_scores.shape[-1] / 3)].mean(-1)), 0] = np.arange(align_scores.shape[0], 0, -1)
        order_attn[np.argsort(align_scores[:, int(align_scores.shape[-1] * 2 / 3):].mean(-1)), 1] = np.arange(align_scores.shape[0])
        order_attn = order_attn.mean(-1)
        attn_sort_idx = order_attn.argsort()
        
        align_scores = align_scores[attn_sort_idx]
        valid_bases = valid_bases[attn_sort_idx]
        seq_name_list = list(np.asarray(seq_name_list)[attn_sort_idx])
        if gap_thres < 1.0:
            filter_align_seqs = filter_align_seqs[attn_sort_idx]
                                     
    return align_scores, valid_bases, seq_name_list, filter_align_seqs, filter_con_seqs
    

def avg_attn_by_category(scores, seqs_name_list, database, tax_ranks, include_all=False):

    attn = dict()
    avg_attn = dict()
    
    seqs_taxid = pd.DataFrame(seqs_name_list, columns=['taxid'])['taxid'].str.split('_').str[0].astype(int).to_frame()
    seqs_taxid = seqs_taxid.join(database.set_index('taxid'), on='taxid')
    
    for tax_rank in tax_ranks:
        class_list = list(seqs_taxid[tax_rank].drop_duplicates())
        if include_all:
            avg_attn[f'{tax_rank}-all'] = scores.mean(0)
        # print(seqs_taxid[tax_rank].value_counts())

        for class_name in class_list:
            idx = (seqs_taxid[tax_rank] == class_name)
            attn[f'{tax_rank}-{class_name}'] = scores[idx]
            avg_attn[f'{tax_rank}-{class_name}'] = scores[idx].mean(0)
        
    return avg_attn, attn

