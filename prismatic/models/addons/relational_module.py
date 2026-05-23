import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, context):
        # Cross-attention: query attends to context
        attn_out, _ = self.attn(query, context, context)
        query = self.norm(query + attn_out)

        # Feedforward
        ff_out = self.ff(query)
        query = self.norm2(query + ff_out)
        return query

class DualCrossAttentionLayer(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.visual_attn = CrossAttentionBlock(dim, n_heads)
        self.slot_attn = CrossAttentionBlock(dim, n_heads)

    def forward(self, relation_tokens, patch_tokens, slot_tokens):
        relation_tokens = self.visual_attn(relation_tokens, patch_tokens)
        relation_tokens = self.slot_attn(relation_tokens, slot_tokens)
        return relation_tokens

class RelationTokensGrounding(nn.Module):
    def __init__(self, dim, in_dim=None, num_relation_tokens=16, n_heads=4, num_layers=3):
        super().__init__()
        self.num_relation_tokens = num_relation_tokens
        self.in_dim = in_dim
        self.dim = dim

        # Projection: patch tokens → internal dim
        if self.in_dim != dim:
            self.patch_proj_in = nn.Linear(self.in_dim, dim)
            self.slot_proj_in = nn.Linear(self.in_dim, dim)
            self.relation_proj_out = nn.Linear(dim, self.in_dim)
        else:
            self.patch_proj_in = nn.Identity()
            self.slot_proj_in = nn.Identity()
            self.relation_proj_out = nn.Identity()

        # Learnable relation tokens (shared across batch)
        self.relation_tokens = nn.Parameter(torch.randn(1, num_relation_tokens, dim))

        # Cross-attention processing layers
        self.layers = nn.ModuleList([
            DualCrossAttentionLayer(dim, n_heads)
            for _ in range(num_layers)
        ])

    def forward(self, patch_tokens, slot_tokens):
        B = patch_tokens.size(0)

        # Project patch tokens in
        patch_tokens = self.patch_proj_in(patch_tokens)  # [B, N_patch, dim]
        slot_tokens = self.slot_proj_in(slot_tokens)

        # Expand learnable relation tokens
        relation_tokens = self.relation_tokens.expand(B, -1, -1)  # [B, N_rel, dim]

        # Process through stacked dual-attention
        for layer in self.layers:
            relation_tokens = layer(relation_tokens, patch_tokens, slot_tokens)

        # Project relation tokens out
        relation_tokens = self.relation_proj_out(relation_tokens)  # [B, N_rel, patch_dim]
        return relation_tokens  # [B, N_rel, D]
