import math
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Optional, List, Union, Tuple
from torchvision import transforms
from timm.models import create_model
from einops import rearrange, repeat
import copy
from einops import rearrange
from typing import Optional, List
from timm.models.layers import  trunc_normal_



class Cross_Attention(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, hidden_dim=768,
                 return_intermediate_dec=False, sa_first=True,):
        super().__init__()

        self.pre_norm = nn.LayerNorm(d_model) if normalize_before else None
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,
                                                sa_first = sa_first)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.enc_layers = num_encoder_layers

            
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, mask, query_embed):
        # # flatten NxCxHxW to HWxNxC
        # bs, c, h, w = src.shape
        # src = src.flatten(2).permute(2, 0, 1)
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1)

        # flatten NxTxC to TxNxC
        bs, t, c = src.shape
        src = src.permute(1, 0, 2)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask

        tgt = torch.zeros_like(query_embed)
        if self.pre_norm:
            memory = self.pre_norm(src)
        else:
            memory = src

        hs, attn_rollout, self_attn = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          query_pos=query_embed, num_frames=1, seq_len=t)
        return hs[0].transpose(0, 1), memory.permute(0, 1, 2), attn_rollout, self_attn



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                num_frames: Optional[int] = 4,
                seq_len: Optional[int] = 196):
        output = tgt

        intermediate = []
        Q, B = output.shape[:2]
        # attn_rollout = torch.ones(B,Q,memory.shape[0]).to(output.device)
        attns, self_attns= [],[]
        for layer_i,layer in enumerate(self.layers):
            output, attn, self_attn = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, counter=layer_i,
                           num_frames = num_frames, seq_len=seq_len)
            # attns.append(attn.view(B,Q,4,16,16))
            # self_attns.append(self_attn[28][0])
            # attn_rollout  = attn_rollout*attn
            # plot_attn_map(attn.view(B,Q,4,16,16)[27][0].detach().cpu(),name=str(layer_i) )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        # attn_rollout = attn_rollout.view(B,Q,4,16,16)
        # attns = torch.stack(attns).sum(0)
        # self_attns = torch.stack(self_attns)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), attns, self_attns
        return output.unsqueeze(0), attns, self_attns


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, sa_first=True):
        super().__init__()
        self.sa_first = sa_first

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                    counter: Optional[Tensor] = None,
                    num_frames: Optional[int] = 4,
                    seq_len: Optional[int] = 196):
        
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt2.transpose(0,1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    counter: Optional[Tensor] = None,
                    num_frames: Optional[int] = 4,
                    seq_len: Optional[int] = 196):
        if self.sa_first:
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2, self_attn = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)

            tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                        key=self.with_pos_embed(memory, pos),
                        value=memory, attn_mask=memory_mask,
                        key_padding_mask=memory_key_padding_mask)

        else:
            tgt2 = self.norm1(tgt)
            tgt2, attn = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)
            tgt = tgt + self.dropout1(tgt2)
            tgt2 = self.norm2(tgt)

            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2, self_attn = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
            
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, attn, self_attn

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                counter: Optional[Tensor] = None,
                num_frames: Optional[int] = 4,
                seq_len: Optional[int] = 196):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, 
                                    query_pos, counter, num_frames, seq_len)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, 
                                 query_pos, counter, num_frames, seq_len)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class ObjDecoder(nn.Module):
    def __init__(self, 
                 num_queries, 
                 feature_dim=768, 
                 aux_loss=False,
                 pred_traj=True,
                 num_frames=4,
                 patches_per_frame=256,
                 self_attn=False):
        super().__init__()

        self.num_queries = num_queries
        self.transformer = Cross_Attention(d_model=feature_dim, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True, hidden_dim=768)

        hidden_dim = self.transformer.d_model
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.n_decode = 1

    def forward(self, features):
        features = features
        B,T,D = features.shape
        mask = features.new_zeros(B,T).bool()

        hs,_,attn, self_attn = self.transformer(features, mask, self.query_embed.weight)
        return hs, attn, self_attn
