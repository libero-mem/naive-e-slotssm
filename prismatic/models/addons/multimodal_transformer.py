import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from typing import Optional
from torch import Tensor, nn
import copy
import math

def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N, layer_share=False):
    # import ipdb; ipdb.set_trace()
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res

class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.l_proj.weight)
        self.l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_v_proj.weight)
        self.values_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_l_proj.weight)
        self.values_l_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_l_proj.weight)
        self.out_l_proj.bias.data.fill_(0)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """
        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()
        bsz, tgt_len, _ = v.size()

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_v_states = value_v_states.view(*proj_shape)
        value_l_states = value_l_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, nimg, ntxt

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = torch.clamp(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        attn_weights_l = attn_weights_l.softmax(dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        attn_weights_v = attn_weights.softmax(dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2)
        attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l

import numpy as np
# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        gamma_v = init_values * np.ones((v_dim))
        gamma_l = init_values * np.ones((l_dim))
        self.gamma_v = nn.Parameter(torch.tensor(gamma_v))
        self.gamma_l = nn.Parameter(torch.tensor(gamma_l))

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v.to(torch.bfloat16), l.to(torch.bfloat16)

    # def forward(self, v:List[torch.Tensor], l, attention_mask_v=None, attention_mask_l=None)

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        pos: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
        # src_key_padding_mask: Optional[Tensor] = None,
    ):
        # # repeat attn mask
        # if src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
        #     # bs, num_q, num_k
        #     src_mask = src_mask.repeat(self.nhead, 1, 1)

        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]

        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class AfdMultimodalTransformer(nn.Module):
    def __init__(
        self,
        d_model=2176, 
        num_encoder_layers=3
    ):
        super().__init__()
        self.d_model = d_model*2
        self.num_encoder_layers = 3
        self.language_projector = MLP(input_dim=d_model, hidden_dim=self.d_model, output_dim=d_model, num_layers=1)
        self.multimodal_transformer = MultimodalTransformer(
            d_model=d_model, num_encoder_layers=num_encoder_layers
        )
        self.afd_predictor = MLP(input_dim=d_model, hidden_dim=self.d_model, output_dim=1, num_layers=2)

    def forward(
        self,
        visual_features,
        textual_features
    ):
        # print(visual_features.shape, textual_features.shape)
        textual_features = self.language_projector(textual_features)
        visual_features, textual_features = self.multimodal_transformer(visual_features, 
                                                                        textual_features)
        interactability = self.afd_predictor(visual_features).sigmoid()
        return interactability

class MultimodalTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=3,
        dim_feedforward=2048,
        normalize_before=False,
        # for deformable encoder
        num_feature_levels=1,
        text_dropout=0.1,
        image_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_queries = num_queries

        # choose encoder layer type
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead // 2,
            dim_feedforward=dim_feedforward // 2,
            dropout=image_dropout,
        )

        text_enhance_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead // 2,
            dim_feedforward=dim_feedforward // 2,
            dropout=text_dropout,
        )

        feature_fusion_layer = BiAttentionBlock(
            v_dim=d_model,
            l_dim=d_model,
            embed_dim=dim_feedforward // 2,
            num_heads=nhead // 2,
            dropout=fusion_dropout,
            drop_path=fusion_droppath,
        )

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries  # useful for single stage model only
        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None


        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def forward(self, slot_features, text_features):
        """
        Input:
            - srcs: List of multi features [bs, \sum{hxw}, c]

        """
        #########################################################
        # Begin Encoder
        #########################################################
        memory, memory_text = self.encoder(
            visual_features=slot_features,
            textual_features=text_features,
        )
        return memory, memory_text


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.text_layers = []
        self.fusion_layers = []
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if text_enhance_layer is not None:
                self.text_layers = _get_clones(
                    text_enhance_layer, num_layers, layer_share=enc_layer_share
                )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer

            if text_enhance_layer is not None:
                self.text_layers = []
                del text_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt
        # self.project_in_features  = nn.Linear(4096, 2176, bias=True)
        # self.project_out_features = nn.Linear(2176, 4096, bias=True)

    def forward(
        self,
        # for images
        visual_features: Tensor,
        # for texts
        textual_features: Tensor = None,
    ):
        """
        Input:
            - visual_features: [bs, sum(hi*wi), cc]
            - textual_features: bs, n_text, cc
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """
        output = visual_features # self.project_in_features(visual_features)

        if self.text_layers:
            # generate pos_text
            bs, n_text, text_dim = textual_features.shape
            pos_text = (
                torch.arange(n_text, device=textual_features.device)
                .float()
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(bs, 1, 1)
            )
            pos_text = get_sine_pos_embed(pos_text, num_pos_feats=visual_features.shape[-1], exchange_xy=False)

        # main process
        for layer_id, visual_layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                output, textual_features = self.fusion_layers[layer_id](
                    v=output,
                    l=textual_features,
                    # attention_mask_v=key_padding_mask,
                    # attention_mask_l=text_attention_mask,
                )

            if self.text_layers:
                textual_features = self.text_layers[layer_id](
                    src=textual_features.transpose(0, 1),
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                    # src_mask=~text_self_attention_masks,  # note we use ~ for mask here
                    # src_key_padding_mask=text_attention_mask,
                ).transpose(0, 1)

            # main process
            output = visual_layer(
                src=output,
                # pos=pos,
                # reference_points=reference_points,
                # spatial_shapes=spatial_shapes,
                # level_start_index=level_start_index,
                # key_padding_mask=key_padding_mask,
            )
        
        # output = self.project_out_features(output)

        return output, textual_features



class EgoExoTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=3,
        dim_feedforward=2048,
        normalize_before=False,
        # for deformable encoder
        num_feature_levels=1,
        text_dropout=0.1,
        image_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_queries = num_queries

        # choose encoder layer type
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead // 2,
            dim_feedforward=dim_feedforward // 2,
            dropout=image_dropout,
        )

        ego_enhance_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead // 2,
            dim_feedforward=dim_feedforward // 2,
            dropout=image_dropout,
        )

        feature_fusion_layer = BiAttentionBlock(
            v_dim=d_model,
            l_dim=d_model,
            embed_dim=dim_feedforward // 2,
            num_heads=nhead // 2,
            dropout=fusion_dropout,
            drop_path=fusion_droppath,
        )

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = EgoExoTransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            ego_enhance_layer=ego_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries  # useful for single stage model only
        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None


        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def forward(self, exo_features, ego_features):
        """
        Input:
            - srcs: List of multi features [bs, \sum{hxw}, c]

        """
        #########################################################
        # Begin Encoder
        #########################################################
        memory_exo, memory_ego = self.encoder(
            exo_features=exo_features,
            ego_features=ego_features,
        )
        return memory_exo, memory_ego


class EgoExoTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        ego_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.ego_layers = []
        self.fusion_layers = []
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if ego_enhance_layer is not None:
                self.ego_layers = _get_clones(
                    ego_enhance_layer, num_layers, layer_share=enc_layer_share
                )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer

            if ego_enhance_layer is not None:
                self.ego_layers = []
                del ego_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt
        # self.project_in_features  = nn.Linear(4096, 2176, bias=True)
        # self.project_out_features = nn.Linear(2176, 4096, bias=True)

    def forward(
        self,
        # for exo images
        exo_features: Tensor,
        # for ego images
        ego_features: Tensor = None,
    ):
        """
        Input:
            - exo_features: [bs, sum(hi*wi), cc]
            - ego_features: [bs, sum(hi*wi), cc]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """
        # exo_features = exo_features # self.project_in_features(visual_features)

        # main process
        for layer_id, exo_layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                exo_features, ego_features = self.fusion_layers[layer_id](
                    v=exo_features,
                    l=ego_features,
                    # attention_mask_v=key_padding_mask,
                    # attention_mask_l=text_attention_mask,
                )

            if self.ego_layers:
                ego_features = self.ego_layers[layer_id](
                    src=ego_features,
                    # pos=pos,
                    # reference_points=reference_points,
                    # spatial_shapes=spatial_shapes,
                    # level_start_index=level_start_index,
                    # key_padding_mask=key_padding_mask,
                )

            # main process
            exo_features = exo_layer(
                src=exo_features,
                # pos=pos,
                # reference_points=reference_points,
                # spatial_shapes=spatial_shapes,
                # level_start_index=level_start_index,
                # key_padding_mask=key_padding_mask,
            )
        
        # output = self.project_out_features(output)

        return exo_features, ego_features


class AfdTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=3,
        dim_feedforward=2048,
        normalize_before=False,
        # for deformable encoder
        num_feature_levels=1,
        text_dropout=0.1,
        image_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_queries = num_queries

        # choose encoder layer type
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead // 2,
            dim_feedforward=dim_feedforward // 2,
            dropout=image_dropout,
        )

        feature_fusion_layer = BiAttentionBlock(
            v_dim=d_model,
            l_dim=d_model,
            embed_dim=dim_feedforward // 2,
            num_heads=nhead // 2,
            dropout=fusion_dropout,
            drop_path=fusion_droppath,
        )

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = AfdTransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            ctx_enhance_layer=None,
            feature_fusion_layer=feature_fusion_layer,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.num_queries = num_queries  # useful for single stage model only
        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None


        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def forward(self, context_tokens, object_tokens):
        """
        Input:
            - srcs: List of multi features [bs, \sum{hxw}, c]

        """
        #########################################################
        # Begin Encoder
        #########################################################
        context_tokens = torch.concat([context_tokens, object_tokens], dim=1)
        memory_ctx, memory_afd = self.encoder(
            context_features=context_tokens,
            object_features=object_tokens,
        )
        return memory_afd


class AfdTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        ctx_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        """_summary_

        Args:
            encoder_layer (_type_): _description_
            num_layers (_type_): _description_
            norm (_type_, optional): _description_. Defaults to None.
            d_model (int, optional): _description_. Defaults to 256.
            num_queries (int, optional): _description_. Defaults to 300.
            enc_layer_share (bool, optional): _description_. Defaults to False.

        """
        super().__init__()
        # prepare layers
        self.layers = []
        self.ctx_layers = []
        self.fusion_layers = []
        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)

            if ctx_enhance_layer is not None:
                self.ctx_layers = _get_clones(
                    ctx_enhance_layer, num_layers, layer_share=enc_layer_share
                )
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(
                    feature_fusion_layer, num_layers, layer_share=enc_layer_share
                )
        else:
            self.layers = []
            del encoder_layer

            if ctx_enhance_layer is not None:
                self.ctx_layers = []
                del ctx_enhance_layer
            if feature_fusion_layer is not None:
                self.fusion_layers = []
                del feature_fusion_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt
        # self.project_in_features  = nn.Linear(4096, 2176, bias=True)
        # self.project_out_features = nn.Linear(2176, 4096, bias=True)

    def forward(
        self,
        # for exo images
        context_features: Tensor,
        # for ego images
        object_features: Tensor = None,
    ):
        """
        Input:
            - exo_features: [bs, sum(hi*wi), cc]
            - ego_features: [bs, sum(hi*wi), cc]
        Outpus:
            - output: [bs, sum(hi*wi), 256]
        """
        # exo_features = exo_features # self.project_in_features(visual_features)

        # main process
        for layer_id, obj_layer in enumerate(self.layers):
            # if output.isnan().any() or memory_text.isnan().any():
            #     if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
            #         import ipdb; ipdb.set_trace()
            if self.fusion_layers:
                object_features, context_features = self.fusion_layers[layer_id]( # to be changed
                    v=object_features,
                    l=context_features,
                    # attention_mask_v=key_padding_mask,
                    # attention_mask_l=text_attention_mask,
                )
                # context_features, object_features = self.fusion_layers[layer_id]( # current version
                #     v=context_features,
                #     l=object_features,
                #     # attention_mask_v=key_padding_mask,
                #     # attention_mask_l=text_attention_mask,
                # )

            if self.ctx_layers:
                context_features = self.ctx_layers[layer_id](
                    src=context_features,
                    # pos=pos,
                    # reference_points=reference_points,
                    # spatial_shapes=spatial_shapes,
                    # level_start_index=level_start_index,
                    # key_padding_mask=key_padding_mask,
                )

            # main process
            object_features = obj_layer(
                src=object_features,
                # pos=pos,
                # reference_points=reference_points,
                # spatial_shapes=spatial_shapes,
                # level_start_index=level_start_index,
                # key_padding_mask=key_padding_mask,
            )
        
        # output = self.project_out_features(output)

        return context_features, object_features




# class CrossAttentionBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         *,
#         context_dim=None,
#         dim_head=64,
#         heads=8,
#         parallel_ff=False,
#         ff_mult=4,
#         norm_context=False
#     ):
#         super().__init__()
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         inner_dim = heads * dim_head
#         context_dim = default(context_dim, dim)

#         # self.norm = LayerNorm(dim)
#         # self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
#         self.to_out = nn.Linear(inner_dim, dim, bias=False)

#         # whether to have parallel feedforward

#         ff_inner_dim = ff_mult * dim

#         self.ff = nn.Sequential(
#             nn.Linear(dim, ff_inner_dim * 2, bias=False),
#             SwiGLU(),
#             nn.Linear(ff_inner_dim, dim, bias=False)
#         ) if parallel_ff else None

#     def forward(self, x, context):
#         """
#         einstein notation
#         b - batch
#         h - heads
#         n, i, j - sequence length (base sequence length, source, target)
#         d - feature dimension
#         """

#         # pre-layernorm, for queries and context

#         x = x # self.norm(x)
#         context = context # self.context_norm(context)

#         # get queries

#         q = self.to_q(x)
#         q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

#         # scale

#         q = q * self.scale

#         # get key / values

#         k, v = self.to_kv(context).chunk(2, dim=-1)

#         # query / key similarity

#         sim = torch.einsum('b h i d, b j d -> b h i j', q, k)

#         # attention

#         sim = sim - sim.amax(dim=-1, keepdim=True)
#         attn = sim.softmax(dim=-1)

#         # aggregate

#         out = torch.einsum('b h i j, b j d -> b h i d', attn, v)

#         # merge and combine heads

#         out = rearrange(out, 'b h n d -> b n (h d)')
#         out = self.to_out(out)

#         # add parallel feedforward (for multimodal layers)

#         if exists(self.ff):
#             out = out + self.ff(x)

#         return out


# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, n_heads=8, dim_feedforward=2048, dropout=0.1):
#         """
#         Args:
#             d_model: Dimension of the model (embedding size).
#             n_heads: Number of attention heads.
#             dim_feedforward: Hidden layer size in the feedforward network.
#             dropout: Dropout probability.
#         """
#         super().__init__()
#         self.attention_layer = nn.MultiheadAttention(
#             embed_dim=d_model, 
#             num_heads=n_heads, 
#             dropout=dropout, 
#             batch_first=True
#         )
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.drop_out_1 = nn.Dropout(dropout)
#         self.drop_out_2 = nn.Dropout(dropout)

#     def forward(self, x, mask=None, key_padding_mask=None):
#         attn_out, _ = self.attention_layer(
#             query=x, key=x, value=x,
#             attn_mask=mask,
#             key_padding_mask=key_padding_mask
#         )
#         x = x + self.drop_out_1(attn_out)
#         x = x + self.linear2(self.drop_out_2(F.relu(self.linear1(x))))
#         return x

# class ParallelTransformerBlock(nn.Module):
#     def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
#         super().__init__()
#         # self.norm = LayerNorm(dim)

#         attn_inner_dim = dim_head * heads
#         ff_inner_dim = dim * ff_mult
#         self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

#         self.heads = heads
#         self.scale = dim_head**-0.5

#         self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
#         self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

#         self.ff_out = nn.Sequential(
#             SwiGLU(),
#             nn.Linear(ff_inner_dim, dim, bias=False)
#         )

#         # for caching causal mask and rotary embeddings

#         self.mask = None
#         self.pos_emb = None

#     def get_mask(self, n, device):
#         if self.mask is not None and self.mask.shape[-1] >= n:
#             return self.mask[:n, :n].to(device)

#         mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
#         self.mask = mask
#         return mask

#     def forward(self, x, attn_mask=None):
#         """
#         einstein notation
#         b - batch
#         h - heads
#         n, i, j - sequence length (base sequence length, source, target)
#         d - feature dimension
#         """

#         n, device, h = x.shape[1], x.device, self.heads

#         # pre layernorm

#         # x = self.norm(x)

#         # attention queries, keys, values, and feedforward inner

#         q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

#         # split heads
#         # they use multi-query single-key-value attention, yet another Noam Shazeer paper
#         # they found no performance loss past a certain scale, and more efficient decoding obviously
#         # https://arxiv.org/abs/1911.02150

#         q = rearrange(q, "b n (h d) -> b h n d", h=h)

#         # rotary embeddings

#         # positions = self.get_rotary_embedding(n, device)
#         # q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

#         # scale

#         q = q * self.scale

#         # similarity

#         sim = torch.einsum("b h i d, b j d -> b h i j", q, k)

#         # causal mask

#         causal_mask = self.get_mask(n, device)
#         sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

#         # extra attention mask - for masking out attention from text CLS token to padding

#         if exists(attn_mask):
#             attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
#             sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

#         # attention

#         sim = sim - sim.amax(dim=-1, keepdim=True).detach()
#         attn = sim.softmax(dim=-1)

#         # aggregate values

#         out = torch.einsum("b h i j, b j d -> b h i d", attn, v)

#         # merge heads

#         out = rearrange(out, "b h n d -> b n (h d)")
#         return self.attn_out(out) + self.ff_out(ff)

# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn

#     def forward(self, x, *args, **kwargs):
#         return self.fn(x, *args, **kwargs) + x

# class AfdTransformer(nn.Module):
#     def __init__(
#         self,
#         dim=2176,
#         dim_head=256,
#         n_heads=8,
#         levels=3,
#         ff_mult=4
#     ):
#         super().__init__()
#         self.attention_layers = nn.ModuleList([])
#         for ind in range(levels):
#             self.attention_layers.append(nn.ModuleList([
#                 Residual(CrossAttentionBlock(dim=dim, dim_head=dim_head, heads=n_heads, parallel_ff=True, ff_mult=ff_mult)),
#                 Residual(TransformerBlock(d_model=dim, n_heads=n_heads, dim_feedforward=dim, dropout=0.1))
#             ]))


#     def forward(self, context_tokens, object_tokens):
#         """
#         Input:
#             - srcs: List of multi features [bs, \sum{hxw}, c]

#         """
#         #########################################################
#         # Begin Encoder
#         #########################################################
#         context_tokens = torch.concat([context_tokens, object_tokens], dim=1)

#         # main process
#         for cross_attn, attn_ff in self.attention_layers:
#             object_tokens = cross_attn(object_tokens, context_tokens)
#             object_tokens = attn_ff(object_tokens)
#         # output = self.project_out_features(output)

#         return object_tokens
