"""
Created on Fri Jul 21 15:04:45 2023

@author: tangshuo
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Layer Normalization
在每一个样本的特征维度上独立地归一化
'''
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        '''
        缩放参数
        '''
        self.a_2 = nn.Parameter(torch.ones(features))
        '''
        偏移参数
        '''
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        '''
        定义两个线性变换层
        '''
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn

'''
bert: 预训练的BERT模型
opt: 模型的配置
mem_dim: GCN的输出维度
attention_heads: multi-head头数
bert_dim: Bert的维度
bert_drop：Sequence_output的dropout
pooled_drop: Pooled_output的dropout
gcn_drop: GCN output的dropout
layernorm: 归一化层
'''
class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()  
        
        self.bert = bert  
        self.opt = opt  
        self.layers = num_layers  # GCN的层数
        self.mem_dim = opt.bert_dim // 2  # GCN的输出维度，这里设置为BERT维度的一半
        self.attention_heads = opt.attention_heads  # 多头注意力的头数
        self.bert_dim = opt.bert_dim  # BERT的维度
        self.bert_drop = nn.Dropout(opt.bert_dropout)  # BERT后的dropout层
        self.pooled_drop = nn.Dropout(opt.bert_dropout)  # 对BERT的pool输出进行dropout
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)  # GCN layers的dropout层
        self.layernorm = LayerNorm(opt.bert_dim)  # 对BERT的输出进行LayerNorm
        '''
        initialize SynGCN layers
        '''
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))
        
        '''
        initialize SemGCN layers
        '''
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))
        
        '''
        multi-head attention
        '''
        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
        
        '''
        initialize biaffine
        '''
        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def forward(self, adj, inputs):
        def check_nan_inf(tensor, tensor_name=""):
            if torch.isnan(tensor).any():
                print(f"{tensor_name} has NaN values!")
            if torch.isinf(tensor).any():
                print(f"{tensor_name} has Inf values!")
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
        src_mask = src_mask.unsqueeze(-2)
        bert_outputs = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output = bert_outputs.last_hidden_state
        pooled_output = bert_outputs.pooler_output
        check_nan_inf(text_bert_indices, "text_bert_indices")
        check_nan_inf(bert_segments_ids, "bert_segments_ids")
        check_nan_inf(attention_mask, "attention_mask")
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)
                
        
        '''
        SemGCN obtains an attention matrix as an adjacency matrix via a self-attention mechanism.
         1. Split the attention tensor into a list of attention matrices
         2. Average the attention matrices over all heads
         3. Remove self-loops and add identity matrix (to preserve self-connections) in the adjacency matrix
         4. Apply the source mask to the adjacency matrix
        '''
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        adj_ag = None
        
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i].clone() # Create a new tensor that is not a view
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for j in range(adj_ag.size(0)):
            adj_ag[j] = adj_ag[j] - torch.diag(torch.diag(adj_ag[j])).to(device)
            adj_ag[j] = adj_ag[j] + torch.eye(adj_ag[j].size(0)).to(device)

        
        adj_ag = src_mask.transpose(1, 2) * adj_ag
        
        '''
        Calculate the degree of the dependency relation and attention adjacency matrix to be used in GCN computations
        '''
        denom_dep = adj.sum(2).unsqueeze(2) + 1
        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        
        
        '''
        Initialize the input for GCN
        outputs_ag: SemGCN的input
        outputs_dep: SynGCN的input
        '''
        outputs_ag = gcn_inputs
        outputs_dep = gcn_inputs
        
        
        '''
        Apply GCN layers
        gAxW_dep: Syntactic
        gAxW_ag: Semantic
        '''
        for l in range(self.layers):
            '''
            SynGCN computations: matrix multiplication, followed by division by degree, followed by ReLU activation
            '''
            Ax_dep = adj.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.leaky_relu(AxW_dep, negative_slope=0.05)
            '''
            SemGCN computations: similar to SynGCN but with a different adjacency matrix and weights
            '''
            Ax_ag = adj_ag.bmm(outputs_ag)
            AxW_ag = self.weight_list[l](Ax_ag)
            AxW_ag = AxW_ag / denom_ag
            gAxW_ag = F.leaky_relu(AxW_dep, negative_slope=0.05)
            '''
            Mutual Biaffine module: applies a softmaxed bilinear transformation on both types of GCN outputs
            '''
            A1 = F.softmax(torch.bmm(torch.matmul(gAxW_dep, self.affine1), torch.transpose(gAxW_ag, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(gAxW_ag, self.affine2), torch.transpose(gAxW_dep, 1, 2)), dim=-1)

            '''
            Updates GCN outputs with results from Biaffine module
            '''
            gAxW_dep, gAxW_ag = torch.bmm(A1, gAxW_ag), torch.bmm(A2, gAxW_dep)
            '''
            Apply dropout to the GCN outputs, unless it's the last layer
            '''
            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
            outputs_ag = self.gcn_drop(gAxW_ag) if l < self.layers - 1 else gAxW_ag
        
        return outputs_ag, outputs_dep, adj_ag, pooled_output 


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs

        h1, h2, adj_ag, pooled_output = self.gcn(adj_dep, inputs)
        
        '''
        Compute the sum of the aspect mask along dimension 1 and add an extra dimension at the end
        '''
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        
        '''
        Expand the aspect mask by repeating it across the last dimension (bert_dim // 2 times)
        '''
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2) 
        '''
        Multiply the aspect mask with the GCN outputs, sum over dimension 1, and divide by the sum of the aspect mask to get the average
        '''
        outputs1 = (h1*aspect_mask).sum(dim=1) / asp_wn
        outputs2 = (h2*aspect_mask).sum(dim=1) / asp_wn
        return outputs1, outputs2, adj_ag, adj_dep, pooled_output 
    
class DualGCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        
        '''
        Initialize the classifier layer as a fully connected (linear) layer
        '''
        self.classifier = nn.Linear(opt.bert_dim*2, opt.polarities_dim)
        
    def forward(self, inputs):

        outputs1, outputs2, adj_ag, adj_dep, pooled_output = self.gcn_model(inputs)
        '''
        Concatenate the outputs along the last dimension
        '''
        final_outputs = torch.cat((outputs1, outputs2, pooled_output), dim=-1)
        logits = self.classifier(final_outputs)

        '''
        expand the identity matrix to the size of adj_ag
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        identity = torch.eye(adj_ag.size(1)).to(device)
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))

        
        adj_ag_T = adj_ag.transpose(1, 2)
        ortho = adj_ag @ adj_ag_T

        '''
        Subtract the diagonal of the ortho matrix from itself and add the identity matrix
        '''
        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i])).to(device)
            ortho[i] += torch.eye(ortho[i].size(0)).to(device)


        penal = None

        
        if self.opt.losstype == 'doubleloss':
            penal1 = (torch.norm(ortho - identity) / adj_ag.size(0)).to(device)
            penal2 = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).to(device)
            penal = self.opt.alpha * penal1 + self.opt.beta * penal2
        elif self.opt.losstype == 'orthogonalloss':
            penal = (torch.norm(ortho - identity) / adj_ag.size(0)).to(device)
            penal = self.opt.alpha * penal
        elif self.opt.losstype == 'differentiatedloss':
            penal = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).to(device)
            penal = self.opt.beta * penal
        return logits, penal 
