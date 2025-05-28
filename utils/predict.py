import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from utils.dataset import seq2kmers

def pred_for_seqs(model, input):
    
    if isinstance(model, tuple):
        bert, probe = model
    
        bert.eval()
        probe.eval()
    
        if isinstance(input, list):
            input_ids, token_type_ids, attention_mask = (
                input[0].to(probe.device),
                input[1].to(probe.device),
                input[2].to(probe.device),
            )
        else:
            input_ids, token_type_ids, attention_mask = (
                input["input_ids"].to(probe.device),
                input["token_type_ids"].to(probe.device),
                input["attention_mask"].to(probe.device),
            )
    
        with torch.no_grad():
            # obtain logits and transformed for input data
            outputs = bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                output_attentions=False,
            )
            hidden_states = outputs[2]
            sequence_output = (
                hidden_states[probe.layer_num].to(probe.device).to(probe.default_dtype)
            )[:, 0, :]
            
            logits = probe(sequence_output)
    else:
        model.eval()
        
        input_ids, token_type_ids, attention_mask = (
            input["input_ids"].to(model.device),
            input["token_type_ids"].to(model.device),
            input["attention_mask"].to(model.device),
        )
        
        with torch.no_grad():
            model_outputs = model.forward(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = model_outputs.logits


    outputs = {
        'pred_label': [],
        'novelty_score': [],
        'max_prob': [],
    }
    for i in range(len(logits)):
        prob = torch.softmax(logits[i], dim=-1)
        outputs['pred_label'].append(torch.argmax(prob, dim=-1))
        outputs['max_prob'].append(torch.max(prob, dim=-1).values)
        outputs['novelty_score'].append(-torch.sum(prob * torch.log(prob), dim=-1) / torch.log(torch.tensor(prob.shape[1], dtype=prob.dtype, device=prob.device)))
        # max_ent = torch.log(torch.tensor(prob.shape[1], dtype=prob.dtype, device=prob.device))
    
    for key in outputs.keys():
        outputs[key] = torch.stack(outputs[key], dim=1).cpu()
    
    torch.cuda.empty_cache()
    
    return outputs


def get_attention(model, inputs, device, batch_size=32):
    if isinstance(model, tuple):
        bert, probe = model
        bert.eval()
        probe.eval()
    else:   
        model.eval()
    
    num_seqs = len(inputs['input_ids'])
    attn_score_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_seqs, batch_size), desc='Calculating attention'):
            input_ids = inputs['input_ids'][i:min(i+batch_size, num_seqs)]
            attention_mask = inputs['attention_mask'][i:min(i+batch_size, num_seqs)]
            token_type_ids = inputs['token_type_ids'][i:min(i+batch_size, num_seqs)]
            
            if isinstance(model, tuple):
                outputs = bert(
                    input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    token_type_ids=token_type_ids.to(device),
                    output_hidden_states=False,
                    output_attentions=True,
                )
                
                layer = probe.layer_num - 1
                
                # batch_size * head * seq_len * seq_len
                attention = outputs['attentions'][layer].detach().cpu()
                # batch_size * seq_len
                attn_score = attention[:,:,0,1:-1].sum(1)
                
            else:
                outputs = model.forward(input_ids=input_ids.to(device), 
                                        attention_mask=attention_mask.to(device))
                
                layer = -1

                # batch_size * head * seq_len * seq_len
                attention = outputs['attentions'][layer].detach().cpu()
                # batch_size * seq_len
                attn_score = attention[:,:,0,:].sum(1)
            
            attn_score_list.append(attn_score)
            
    # num_seqs * seq_len
    attn_score = torch.vstack(attn_score_list)
    
    return attn_score


def split_attn_segs(dataset, tokenizer, seqs, seq_ipt, scores, num_segs):
    if dataset == 'PlantSeqs':
        avg_scores = torch.zeros((len(seqs), num_segs))
        for i in range(len(scores)):
            segment_length = sum(scores[i] != 0) // num_segs
            for j in range(num_segs):
                start = j * segment_length
                end = start + segment_length
                avg_scores[i, j] = torch.mean(scores[i, start:end])
    elif dataset == 'bordetella':
        assert seq_ipt['input_ids'].shape == scores.shape, f'{seq_ipt["input_ids"].shape} != {scores.shape}'
            # get special tokens
        special_tokens = tokenizer.special_tokens_map
            
            # get vocab with key as the id and value as the token
        token_to_id = tokenizer.get_vocab()
        id_to_token_length = {v: 0 if k in special_tokens.values() else len(k) for k, v in token_to_id.items()}
        id_to_token_length[0] = 1 # [UNK] token
            
        avg_scores = torch.zeros((len(seqs), num_segs))

        for i, seq_ids in enumerate(seq_ipt['input_ids']):
            seq_ids = seq_ids.detach().cpu().numpy()
            attn_scores = []
            for s, id in enumerate(seq_ids):
                attn_scores += [scores[i, s].item()] * id_to_token_length[id]
                
            assert len(attn_scores) == len(seqs[i]), f'{len(attn_scores)} != {len(seqs[i])}'
                
            segment_length = len(attn_scores) // num_segs
            for s in range(num_segs):
                start = s * segment_length
                end = start + segment_length
                avg_scores[i, s] = np.mean(attn_scores[start:end])
    else:
        raise ValueError('Please check the dataset name.')
    
    return avg_scores


def calculate_attention(dataset, model, tokenizer, device, seqs, seq_name_list=None, attention_file=None, num_segs=1):
    print(f"### Calculating attention .... ###")
    
    if dataset == 'PlantSeqs':
        seq_kmers = []
        for seq in seqs:
            seq_kmers.append(seq2kmers(seq))
        seq_ipt = tokenizer(seq_kmers, padding=True, truncation=True, max_length=512, return_tensors="pt",)
        
    elif dataset == 'bordetella':
        seq_ipt = tokenizer.batch_encode_plus(
            seqs, 
            max_length=5000, 
            return_tensors='pt', 
            padding='longest', 
            truncation=True
        )
        
    scores = get_attention(model, seq_ipt, device, batch_size=32)
    
    if num_segs > 1:
        avg_scores = split_attn_segs(dataset, tokenizer, seqs, seq_ipt, scores, num_segs)
        
        return avg_scores
    else:
        
        if dataset == 'PlantSeqs':
            attn_scores = scores.repeat(1, 3).reshape(scores.shape[0], 3, scores.shape[1]).permute(0, 2, 1).reshape(scores.shape[0], -1).to(torch.float64)
            
        elif dataset == 'bordetella':
            assert seq_ipt['input_ids'].shape == scores.shape, f'{seq_ipt["input_ids"].shape} != {scores.shape}'
            # get special tokens
            special_tokens = tokenizer.special_tokens_map
            
            # get vocab with key as the id and value as the token
            token_to_id = tokenizer.get_vocab()
            id_to_token_length = {v: 0 if k in special_tokens.values() else len(k) for k, v in token_to_id.items()}
            id_to_token_length[0] = 1 # [UNK] token

            max_length = max([len(seq) for seq in seqs])
            attn_scores = np.zeros((len(seqs), max_length))
            
            for i, seq_ids in enumerate(seq_ipt['input_ids']):
                seq_ids = seq_ids.detach().cpu().numpy()
                attn_score = []
                for s, id in enumerate(seq_ids):
                    attn_score += [scores[i, s].item()] * id_to_token_length[id]
                
                assert len(attn_score) == len(seqs[i]), f'{len(attn_score)} != {len(seqs[i])}'
                
                attn_scores[i, :len(attn_score)] = np.array(attn_score)
            
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        print(f"Size of attention score is {attn_scores.shape}")
        
        # save to file
        if attention_file is not None:
            # transfer scores to DataFrame
            scores_pd = pd.DataFrame(attn_scores, index=seq_name_list).reset_index()
            scores_pd.columns = ['seq_name'] + [f'attn_{i}' for i in range(attn_scores.shape[1])]
            scores_pd.insert(1, 'seq', seqs)
            
            scores_pd.to_csv(attention_file)

        return attn_scores
