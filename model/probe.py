import torch
from torch import nn

class LinearProbeMultiRank(nn.Module):
    def __init__(self, device, default_dtype, tax_ranks, labels_maps, model_name='bertphylo', probe_dim=600, temp=1.0):
        super().__init__()
        self.device = device
        self.default_dtype = default_dtype
        self.model_name = model_name
        
        if self.model_name == 'bertphylo':
            self.feature_dim = 768
            self.layer_num = 10
        elif self.model_name == 'dnabert':
            self.feature_dim = 768
            self.layer_num = 12
        elif self.model_name == 'bertax':
            self.feature_dim = 250
            self.layer_num = 12
        else:
            raise AssertionError(f'Unsupported model name: {self.model_name}')
        
        self.tax_ranks = tax_ranks
        self.labels_maps = labels_maps
            
        self.probe_dim = probe_dim
        self.temp = temp
        
        # create hierarchical linear probes
        self.probe = nn.ModuleList()
        for i, tax_rank in enumerate(self.tax_ranks):
            if i == 0:
                input_dim = self.feature_dim
            else:
                # add dim of id classes, except the ood class
                input_dim += len(self.labels_maps[self.tax_ranks[i - 1]]) - 1
                
            self.probe.append(
                nn.Sequential(
                    nn.Linear(input_dim, self.probe_dim),
                    nn.ReLU(),
                    nn.Linear(self.probe_dim, len(self.labels_maps[tax_rank]) - 1)
                )
            )
            
        self.probe.apply(self.init_weights)
                 
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0)
    
    def forward(self, sequence_output):
        logits_list = []
        input_feature = sequence_output
        for i, _ in enumerate(self.tax_ranks):                
            logits = self.probe[i](input_feature) * self.temp
            
            probs = torch.softmax(logits, dim=-1)
            input_feature = torch.cat([input_feature, probs], dim=-1)
            
            logits_list.append(logits)
        
        return logits_list
              
