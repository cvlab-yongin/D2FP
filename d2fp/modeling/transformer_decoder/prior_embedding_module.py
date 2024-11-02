import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class ModifiedSlotAttention(nn.Module):

    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 384):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        hidden_dim = max(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, slots, inputs) :
        B, _, D = inputs.shape
        inputs = self.norm_input(inputs)        
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim = 1) + self.eps
            attn = attn / attn.sum(dim = -1, keepdim = True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            slots = self.gru(updates.reshape(-1, D), slots_prev.reshape(-1, D))
            slots = slots.reshape(B, -1, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots
    
class PriorEmbeddingModule(nn.Module): 

    def __init__(self, num_queries, num_priors, num_features) : 
        super().__init__()
        self.attention_module_1 = ModifiedSlotAttention(num_priors, num_features, iters = 3)
        self.attention_module_2 = ModifiedSlotAttention(num_queries, num_features, iters = 3)
        self.attention_module_3 = ModifiedSlotAttention(num_queries, num_features, iters = 3)

    def forward(self, object_query, pos_embed, enhance_embed, prior) :
        object_query = rearrange(object_query, "n b c -> b n c")
        pos_embed = rearrange(pos_embed, "n b c -> b n c")
        enhance_embed = rearrange(enhance_embed, "n b c -> b n c")

        prior_enhanced = self.attention_module_1(prior, enhance_embed)
        output = self.attention_module_2(object_query, prior_enhanced)
        output = self.attention_module_3(output + pos_embed, prior_enhanced)

        return rearrange(output, "n b c -> b n c")
