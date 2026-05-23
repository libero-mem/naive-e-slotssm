import torch
import torch.nn as nn
import einops
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, f):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(f).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class InstanceNorm1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.InstanceNorm1d(dim, affine=True)
    def forward(self, x):
        ######input has shape: [b n c]
        x = einops.rearrange(x, 'b n c -> b c n')
        x = self.norm1(x)
        x = einops.rearrange(x, 'b c n -> b n c')
        return x

class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_token=5*16*28):
        super().__init__()
        self.encoder_pos = nn.Parameter(torch.randn(1, n_token, dim) * .02)
        self.norm1 = norm_layer(dim)
        # self.norm1 = InstanceNorm1d(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        # self.norm2 = InstanceNorm1d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        # self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        encoder_pos = self.encoder_pos.unsqueeze(2).repeat([1, 1, x.shape[1]//self.encoder_pos.shape[1], 1])
        encoder_pos = encoder_pos.view(1, x.shape[1], x.shape[2])
        x = x + encoder_pos
        inter, attn = self.attn(self.norm1(x))
        x = x + inter
        x = x + self.mlp(self.norm2(x))
        return x, attn

class CrossBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_token=5*16*28):
        super().__init__()
        self.encoder_pos = nn.Parameter(torch.randn(1, n_token, dim) * .02)
        self.norm1 = norm_layer(dim)
#         self.norm1 = InstanceNorm1d(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
#         self.norm2 = InstanceNorm1d(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, f):
        x = x + self.encoder_pos
        f = f + self.encoder_pos
        x = x + self.attn(self.norm1(x), self.norm1(f))
        x = x + self.mlp(self.norm2(x))
        return x

class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False, padding_mode='replicate'),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False, padding_mode='replicate'))

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class STBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, window=False, num_frames=3, end_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, 
                              window=window, num_frames=num_frames, end_size=end_size)
        self.time_norm1 = norm_layer(dim)
        self.time_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, 
                              window=window, num_frames=num_frames, end_size=end_size)
        self.time_fc = nn.Linear(dim, dim)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(0.1)

    def forward(self, x, S, T):
        # x B 1+ST C
        xt = x[:, 1:]
        xt = einops.rearrange(xt, 'b (t s) c -> (b s) t c', t=T, s=S)
        res_time = self.drop_path(self.time_attn(self.time_norm1(xt)))
        res_time = einops.rearrange(res_time, '(b s) t c -> b (t s) c', s=S)
        res_time = self.time_fc(res_time)
        xt = x[:, 1:] + res_time

        init_st_token = x[:, 0].unsqueeze(1)
        st_token = init_st_token.repeat(1, T, 1)
        st_token = einops.rearrange(st_token, 'b t c -> (b t) c').unsqueeze(1)
        xs = xt
        xs = einops.rearrange(xs, 'b (t s) c -> (b t) s c', t=T, s=S)
        xs = torch.cat([st_token, xs], dim=1)
        res_slot = self.drop_path(self.attn(self.norm1(xs)))

        st_token = res_slot[:, 0]
        st_token = einops.rearrange(st_token, '(b t) c -> b t c', t=T)
        st_token = torch.mean(st_token, dim=1, keepdim=True)
        res_slot = res_slot[:, 1:]
        res_slot = einops.rearrange(res_slot, '(b t) s c -> b (t s) c', t=T)
        x = xt

        x = torch.cat((init_st_token, x), 1) + torch.cat((st_token, res_slot), 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SlotBlock(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, window=False, num_frames=3, end_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, 
                              window=window, num_frames=num_frames, end_size=end_size)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(0.1)

    def forward(self, x):
        # x B T S C
        x = einops.rearrange(x, 'b t s c -> (b t) s c')
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(MEBlock, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.conv1 = nn.Conv2d(in_channels=self.channel, 
                               out_channels=self.channel//self.reduction, 
                               kernel_size=1, 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)
        
        self.conv2 = nn.Conv2d(in_channels=self.channel//self.reduction,
                               out_channels=self.channel//self.reduction,
                               kernel_size=3,
                               padding=1,
                               groups=self.channel//self.reduction,
                               bias=False)
        
        self.conv3 = nn.Conv2d(in_channels=self.channel//self.reduction,
                               out_channels=self.channel,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)
        
        self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
    
    def forward(self, x):
        n, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(n*t, c, h, w)
        xr = self.conv1(x)
        xr = self.bn1(xr)
        xrp = self.conv2(xr)
        xr = xr.view(n, t, c//self.reduction, h, w)[:, :t-1]
        xrp = xrp.view(n, t, c//self.reduction, h, w)[:, 1:]
        m = xrp - xr
        m = F.pad(m, self.pad, mode='constant', value=0)
        m = m.view(n*t, c//self.reduction, h, w)
        m = F.adaptive_avg_pool2d(m, (1, 1))
        m = self.conv3(m)
        m = self.bn3(m)
        m = torch.sigmoid(m) - 0.5
        out = m * x
        out = out.view(n, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return out
    


# The following snippet is taken from GroundingDINO
# https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/models/GroundingDINO/fuse_modules.py#L99

from timm.models.layers import DropPath

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
        # value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        # value_v_states = value_v_states.view(*proj_shape)
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

        # attn_weights_l = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        # if self.clamp_min_for_underflow:
        #     attn_weights_l = torch.clamp(
        #         attn_weights_l, min=-50000
        #     )  # Do not increase -50000, data type half has quite limited range
        # if self.clamp_max_for_overflow:
        #     attn_weights_l = torch.clamp(
        #         attn_weights_l, max=50000
        #     )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:
            attention_mask_v = (
                attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            # attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))

        # attn_weights_l = attn_weights_l.softmax(dim=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_l, float("-inf"))
        attn_weights_v = attn_weights.softmax(dim=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        # attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        # attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        # if attn_output_l.size() != (bsz * self.num_heads, src_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.size()}"
        #     )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim)

        # attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        # attn_output_l = attn_output_l.transpose(1, 2)
        # attn_output_l = attn_output_l.reshape(bsz, src_len, self.embed_dim)

        attn_output_v = self.out_v_proj(attn_output_v)
        # attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v # , attn_output_l


class CrossAttentionBlock(nn.Module):
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
        super(CrossAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = init_values # nn.Parameter(init_values * torch.ones((v_dim)), requires_grad=True)
        # self.gamma_l = nn.Parameter(init_values * torch.ones((l_dim)), requires_grad=True)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v = self.attn(
            v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l
        )

        # # v, l = v + delta_v, l + delta_l
        # v = v + self.drop_path(self.gamma_v * delta_v)
        # l = l + self.drop_path(self.gamma_l * delta_l)
        # return v, l

        # v = v + delta_v
        v = v + self.drop_path(self.gamma_v * delta_v)
        return v

    # def forward(self, v:List[torch.Tensor], l, attention_mask_v=None, attention_mask_l=None)