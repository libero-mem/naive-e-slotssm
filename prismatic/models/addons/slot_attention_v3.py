import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from typing import Callable, Optional, List, Union, Tuple
from torchvision import transforms
from timm.models import create_model
from einops import rearrange, repeat
from timm.models.layers import DropPath
from .multimodal_transformer import MultimodalTransformer

class LearnableQueries(nn.Module):
    def __init__(self, token_num, object_dim, init_const):
        super().__init__()
        self.tokens = nn.Parameter(
            torch.randn(1, token_num, object_dim) * init_const
        )
    
    def forward(self, x):
        object_tokens = self.tokens.repeat(x.shape[0], 1, 1)  # (b, 16, d)
        x = x + object_tokens
        return x

class CrossAttention(nn.Module):
    """
    CrossAttention module used in the Perceiver IO model.

    Args:
        query_dim (int): The dimension of the query input.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(self, query_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): The query input tensor.
            context (torch.Tensor): The context input tensor.
            mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            # fill in the masks with negative values
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        # dropout
        attn = self.dropout(attn)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)

class RandomConditioning(nn.Module):
    """Random conditioning with potentially learnt mean and stddev."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        learn_mean: bool = True,
        learn_std: bool = True,
        mean_init: Optional[Callable[[torch.Tensor], None]] = None,
        logsigma_init: Optional[Callable[[torch.Tensor], None]] = None
    ):
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim

        if learn_mean:
            self.slots_mu = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_mu", torch.zeros(1, 1, object_dim))

        if learn_std:
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_logsigma", torch.zeros(1, 1, object_dim))

        if mean_init is None:
            mean_init = nn.init.xavier_uniform_
        if logsigma_init is None:
            logsigma_init = nn.init.xavier_uniform_

        with torch.no_grad():
            mean_init(self.slots_mu)
            logsigma_init(self.slots_logsigma)

    def forward(self, batch_size: int):
        mu = self.slots_mu.expand(batch_size, self.n_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, self.n_slots, -1)
        return mu + sigma * torch.randn_like(mu)


class LearnableConditioning(nn.Module):
    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        init_const: float
    ):
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim
        self.init_const = init_const
        self.tokens = nn.Parameter(
            torch.randn(1, n_slots, object_dim) * init_const
        )
    
    def forward(self, batch_size: int):
        condition = self.tokens.expand(batch_size, -1, -1)
        return condition


def get_activation_fn(name: str, inplace: bool = True, leaky_relu_slope: Optional[float] = 0.1):
    if callable(name):
        return name

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "leaky_relu":
        if leaky_relu_slope is None:
            raise ValueError("Slope of leaky ReLU was not defined")
        return nn.LeakyReLU(leaky_relu_slope, inplace=inplace)
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function {name}")


class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return inputs + self.module(inputs)


def build_mlp(
    input_dim: int,
    output_dim: int,
    features: List[int],
    activation_fn: Optional[Union[str, Callable]] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_layer_norm: bool = False,
    residual: bool = False,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        if activation_fn is not None:
            layers.append(get_activation_fn(activation_fn))
        current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.xavier_uniform_(layers[-1].weight)
    nn.init.zeros_(layers[-1].bias)
    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    if residual:
        return Residual(nn.Sequential(*layers))
    return nn.Sequential(*layers)

def build_mlp_v2(
    input_dim: int,
    output_dim: int,
    features: List[int],
    activation_fn: Optional[Union[str, Callable]] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_layer_norm: bool = False,
    residual: bool = False,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        # nn.init.xavier_uniform_(layers[-1].weight)
        # nn.init.zeros_(layers[-1].bias)
        if activation_fn is not None:
            layers.append(get_activation_fn(activation_fn))
        current_dim = n_features

    feature_layers = nn.Sequential(*layers)
    if residual:
        feature_layers = Residual(feature_layers)

    all_layers = [feature_layers]
    all_layers.append(nn.Linear(current_dim, output_dim))
    # nn.init.xavier_uniform_(all_layers[-1].weight)
    # nn.init.zeros_(all_layers[-1].bias)
    if final_activation_fn is not None:
        all_layers.append(get_activation_fn(final_activation_fn))
    return nn.Sequential(*all_layers)

def build_two_layer_mlp(
    input_dim, output_dim, hidden_dim, initial_layer_norm: bool = False, residual: bool = False
):
    """Build a two layer MLP, with optional initial layer norm.

    Separate class as this type of construction is used very often for slot attention and
    transformers.
    """
    return build_mlp(
        input_dim, output_dim, [hidden_dim], initial_layer_norm=initial_layer_norm, residual=residual
    )


class SlotAttention(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp

    def step(self, slots, k, v, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            # Masked slots should not take part in the competition for features. By replacing their
            # dot-products with -inf, their attention values will become zero within the softmax.
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None):
        # Slot update.
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for _ in range(self.iters):
                slots, attn = self.step(slots, k, v, masks)
        return slots.to(torch.bfloat16), attn.to(torch.bfloat16)

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)

        return slots, attn


class SlotAttentionGrouping(nn.Module):
    """Implementation of SlotAttention for perceptual grouping.

    Args:
        feature_dim: Dimensionality of features to slot attention (after positional encoding).
        object_dim: Dimensionality of slots.
        kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
            `object_dim` is used.
        n_heads: Number of heads slot attention uses.
        iters: Number of slot attention iterations.
        eps: Epsilon in slot attention.
        ff_mlp: Optional module applied slot-wise after GRU update.
        positional_embedding: Optional module applied to the features before slot attention, adding
            positional encoding.
        use_projection_bias: Whether to use biases in key, value, query projections.
        use_implicit_differentiation: Whether to use implicit differentiation trick. If true,
            performs one more iteration of slot attention that is used for the gradient step after
            `iters` iterations of slot attention without gradients. Faster and more memory efficient
            than the standard version, but can not backpropagate gradients to the conditioning input.
        input_dim: Dimensionality of features before positional encoding is applied. Specifying this
            is optional but can be convenient to structure configurations.
    """

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        slot_mask_path: Optional[str] = None,
    ):
        super().__init__()

        self._object_dim = object_dim
        self.slot_attention = SlotAttention(
            dim=object_dim,
            feature_dim=object_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
        )

        self.positional_embedding = build_two_layer_mlp(input_dim=feature_dim,
                                                        output_dim=object_dim,
                                                        hidden_dim=feature_dim,
                                                        initial_layer_norm=True)

        if use_empty_slot_for_masked_slots:
            if slot_mask_path is None:
                raise ValueError("Need `slot_mask_path` for `use_empty_slot_for_masked_slots`")
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

    def forward(
        self,
        extracted_features,
        conditioning,
        slot_masks=None,
    ):
        features = self.positional_embedding(extracted_features)
        slots, attn = self.slot_attention(features, conditioning, slot_masks)
        if slot_masks is not None and self.empty_slot is not None:
            slots[slot_masks] = self.empty_slot.to(dtype=slots.dtype)

        return slots, attn

class PatchDecoderVideo(nn.Module):
    def __init__(
        self,
        object_dim: int,
        output_dim: int,
        num_patches_per_frame: int,
        num_frames: int,
        decoder_input_dim: Optional[int] = None,
        resize: int = 256
    ):
        nn.Module.__init__(self)
        self.output_dim = output_dim
        self.num_patches_per_frame = num_patches_per_frame
        self.num_frames = num_frames
        self.resize = resize

        if decoder_input_dim is not None:
            self.inp_transform = nn.Linear(object_dim, decoder_input_dim, bias=True)
            nn.init.xavier_uniform_(self.inp_transform.weight)
            nn.init.zeros_(self.inp_transform.bias)
        else:
            self.inp_transform = None
            decoder_input_dim = object_dim

        self.decoder = build_mlp_v2(input_dim=decoder_input_dim, output_dim=3+1, features=[512, 256, 128],
                                    activation_fn='relu', initial_layer_norm=True, residual=False)
        self.pos_embed = nn.Parameter(torch.randn(self.num_frames, self.num_patches_per_frame, decoder_input_dim) * 0.02)

    def forward(self, object_features: torch.Tensor, nh, nw):
        assert object_features.dim() >= 3   # Image or video data.  (b, s, d)

        initial_shape = object_features.shape[:-1]              # (b, s)
        object_features = object_features.flatten(0, -2)        # (b*s, d)
        
        if self.inp_transform is not None:
            object_features = self.inp_transform(object_features)

        # duplicate the slot representation into each patch, (b*s, t*n, d)
        object_features = object_features.unsqueeze(1).expand(-1, self.num_frames*nw*nh, -1)

        # Simple learned additive embedding as in ViT
        N = self.num_patches_per_frame
        dim = self.pos_embed.shape[-1]
        patch_pos_embed = nn.functional.interpolate(
            self.pos_embed.reshape(self.num_frames, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            size=(nh, nw),
            mode='bicubic',
            align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        object_features = object_features + patch_pos_embed
        # print('decoder before', object_features.shape)
        # print('decoder before', torch.min(object_features), torch.max(object_features))
        output = self.decoder(object_features)              # (b*s, t*n, d+1)
        output = output.unflatten(0, initial_shape)         # (b, s, t*n, d+1)

        # Split out alpha channel and normalize over slots.
        decoded_patches, alpha = output.split([self.output_dim, 1], dim=-1)     # (b, s, t*n, d), (b, s, t*n, 1)
        decoded_patches = decoded_patches
        alpha = alpha.softmax(dim=-3)       # (b, s, t*n, 1)
        # print('decoded_patches', decoded_patches.shape)
        # print('decoded_patches', torch.min(decoded_patches), torch.max(decoded_patches))
        # print('alpha', torch.min(alpha), torch.max(alpha))

        reconstruction = torch.sum(decoded_patches * alpha, dim=-3)     # (b, t*n, d)
        reconstruction = rearrange(reconstruction, "b (t nh nw) c -> (b t) c nh nw", t=self.num_frames, nh=nh, nw=nw)  # (b*t, s n)
        masks = alpha.squeeze(-1)           # (b, s, t*n)

        masks = rearrange(masks, "b s (t n) -> (b t) s n", t=self.num_frames)       # (b*t, s n)
        masks = masks.reshape(masks.shape[0], masks.shape[1], nh, nw)

        masks_as_image = masks

        # masks_as_image = resize_patches_to_image_non_square(
        #     masks,
        #     size=(nh, nw),
        #     resize_mode="bilinear"
        # )

        return reconstruction, masks, masks_as_image

def resize_patches_to_image_non_square(patches, size, resize_mode):
    H, W = size
    n_channels = patches.shape[-2]
    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, H, W),
        scale_factor=(16.0, 16.0),
        mode=resize_mode,
    )
    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])

# class GroupingVideoMAE(nn.Module):
#     """
#     Spatiotemporal Grouping for non-square multi-resolution input video.
#     """
#     def __init__(
#         self,
#         checkpoint_path,
#         object_dim=128,
#         n_slots=24,
#         feat_dim=768,
#         num_patches=256,
#         num_frames=4,
#         img_size=224
#     ):
#         super().__init__()
#         self.num_patches = num_patches
#         self.num_frames = num_frames
#         self.img_size = img_size

#         # conditioning
#         self.conditioning = RandomConditioning(object_dim=object_dim, n_slots=n_slots)

#         # feature extractor
#         self.model = create_model(
#             "videomae_vit_base_patch16_224",
#             pretrained=False,
#             num_classes=174,
#             all_frames=16 * 1,
#             drop_rate=0.0,
#             drop_path_rate=0.1,
#             attn_drop_rate=0.0,
#             drop_block_rate=None,
#             use_mean_pooling=False,
#             init_scale=0.001,
#         )

#         ckpt = torch.load(checkpoint_path,
#                           map_location='cpu')
#         new_ckpt = {}
#         for k, v in ckpt['model'].items():
#             if 'encoder.' in k:
#                 new_ckpt[k.replace('encoder.', '')] = v
#         self.model.load_state_dict(new_ckpt, strict=False)

#         self.model.requires_grad_(False)
#         self.model.eval()

#         # perceptual grouping
#         ff_mlp = build_two_layer_mlp(input_dim=object_dim, output_dim=object_dim, hidden_dim=object_dim*4,
#                                      initial_layer_norm=True, residual=True)
#         self.grouping = SlotAttentionGrouping(feature_dim=feat_dim,
#                                               object_dim=object_dim,
#                                               use_projection_bias=False,
#                                               ff_mlp=ff_mlp)

#         # object decoder
#         dec_mlp = build_mlp(input_dim=object_dim, output_dim=feat_dim+1, features=[1024, 1024, 1024],
#                             activation_fn=None)
#         self.decoder = PatchDecoderVideo(object_dim=object_dim,
#                                                  output_dim=feat_dim,
#                                                  num_patches_per_frame=num_patches,
#                                                  num_frames=num_frames,
#                                                  decoder=dec_mlp,
#                                                  decoder_input_dim=256,
#                                                  resize=img_size)

#     def forward(self, images):
#         _, _, H, W, _ = images.shape            # (b, t, h, w, c)
#         assert self.num_frames == images.shape[1]

#         conditioning = self.conditioning(images.shape[0])     # (b, s, d)

#         images = rearrange(images, 'b t h w c -> b c t h w')
#         self.model.eval()
#         features = self.model(images)  # (b, t*n, d)

#         slots, attn = self.grouping(features, conditioning)         # (b, s, d), (b, s, t*n)

#         reconstruction, masks, masks_as_image = self.decoder(slots, H, W) # H//16, W//16)
#         # (b, t*n, d),  (b, s, t*n),  (b*t, s, img_size, img_size)

#         masks_as_image = rearrange(masks_as_image, '(n t) s h w -> n t s h w', t=self.num_frames)

#         feat_recon_loss = F.mse_loss(reconstruction, features)

#         return feat_recon_loss, masks_as_image

#     def inference(self, images):
#         _, _, H, W, _ = images.shape
#         assert self.num_frames == images.shape[1]           # (b, t, h, w, c)
#         conditioning = self.conditioning(images.shape[0])     # (b, s, d)
#         images = rearrange(images, 'b t h w c -> b c t h w')
#         self.model.eval()
#         features = self.model(images)  # (b, t*n ,d)
#         slots, attn = self.grouping(features, conditioning)         # (b, s, d), (b, s, t*n)
#         reconstruction, masks, masks_as_image = self.decoder(slots, H//16, W//16)
#         # (b, t*n, d),  (b*t, s, n),  (b, t, s, img_size, img_size)
#         masks = rearrange(masks, '(n t) s l -> n t s l', t=self.num_frames)
#         masks_as_image = rearrange(masks_as_image, '(n t) s h w -> n t s h w', t=self.num_frames)

#         return masks, masks_as_image

def denormalize(tensor, mean=[0.484375, 0.455078125, 0.40625], std=[0.228515625, 0.2236328125, 0.224609375]):
    """
    Denormalizes a torch tensor using the given mean and std.
    
    Args:
        tensor (torch.Tensor): The normalized tensor with shape (C, H, W) or (N, C, H, W).
        mean (list): The mean values for each channel (list of floats).
        std (list): The standard deviation values for each channel (list of floats).
    
    Returns:
        torch.Tensor: The denormalized tensor with the same shape as input.
    """
    mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)  # Reshape to match tensor shape
    std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    
    return tensor * std + mean

INIT_CONST = 0.02
class ObjectCentricDynamics(nn.Module):
    def __init__(self, n_slots=24, feat_dim=2176, object_dim=2176, object_decode_dim=256, num_patches=256, num_frames=1, img_size=224, transformer_dim=4096,
                 conditioning='learnable'):
        super().__init__()
        # conditioning
        if conditioning == 'learnable':
            self.conditioning = LearnableConditioning(object_dim=object_dim, n_slots=n_slots, init_const=INIT_CONST)
        else:
            self.conditioning = RandomConditioning(object_dim=object_dim, n_slots=n_slots)
        # perceptual grouping
        ff_mlp = build_two_layer_mlp(input_dim=object_dim, output_dim=object_dim, hidden_dim=object_dim*4,
                                     initial_layer_norm=True, residual=True)
        self.grouping = SlotAttentionGrouping(feature_dim=feat_dim,
                                              object_dim=object_dim,
                                              use_projection_bias=False,
                                              ff_mlp=ff_mlp)

        # # object decoder
        # self.decoder = PatchDecoderVideo(object_dim=object_dim,
        #                                          output_dim=3,
        #                                          num_patches_per_frame=num_patches,
        #                                          num_frames=num_frames,
        #                                          decoder_input_dim=object_decode_dim,
        #                                          resize=img_size)

        token_num = n_slots
        self.token_num = token_num
        
        # self.tokens = LearnableQueries(
        #     token_num, 
        #     object_dim, 
        #     INIT_CONST
        # ) 
        # self.cross_attention = CrossAttention(
        #     object_dim,
        #     heads=16,
        #     dim_head=object_dim,
        #     dropout=0.0,
        # )
        #
        # self.transformer_encoder = MultimodalTransformer(
        #     d_model=transformer_dim, 
        #     nhead=8, 
        #     num_encoder_layers=3, 
        #     dim_feedforward=transformer_dim,
        # )
        # self.action_latent = build_mlp(input_dim=object_dim, output_dim=feat_dim, features=[1024])

    def forward(self, input_embeddings, patch_features, cur_pixel_values=None, nxt_pixel_values=None):
        b, s, d = patch_features.shape
        # Slot Encoding
        conditioning = self.conditioning(patch_features.shape[0])     # (b, s, d)
        # print('conditioning', conditioning.shape)
        slotted_features, attn = self.grouping(patch_features, conditioning)         # (b, s, d), (b, s, t*n)
        # print('slotted_features', slotted_features.shape)

        # if self.training:
        #     if s == 256:
        #         reconstruction, masks, masks_as_image = self.decoder(slotted_features, 224, 224) # the numbers correspond to size for expanding slots
        #         # (b, t*n, d),  (b, s, t*n),  (b*t, s, img_size, img_size)
        #     else:
        #         reconstruction, masks, masks_as_image = self.decoder(slotted_features, 16, 32)
        #         # (b, t*n, d),  (b, s, t*n),  (b*t, s, img_size, img_size)
        #     # print('reconstruction', reconstruction.shape)
        #     # masks_as_image = rearrange(masks_as_image, '(n t) s h w -> n t s h w', t=1)
        #     # print('reconstruction', torch.min(reconstruction), torch.max(reconstruction))
        #     # print('masks', torch.min(masks), torch.max(masks))
        #     cur_pixel_values = denormalize(cur_pixel_values[:,:3,:,:])
        #     # print('cur_pixel_values', torch.min(cur_pixel_values[:,:3,:,:]), torch.max(cur_pixel_values[:,:3,:,:]))
        #     cur_recon_loss = F.mse_loss(reconstruction, cur_pixel_values[:,:3,:,:])*10
        #     # print('recon_loss', cur_recon_loss)
        #     # 1/0
        # else:
        #     cur_recon_loss = 

        # ###########################################################################
        # # count at __embeddings
        # import os
        # import pickle
        # embedding_dir = '__embeddings_dino_slots_revised'
        # embeddings = os.listdir(embedding_dir)
        # data = {
        #     'reconstruction': reconstruction,
        #     'masks': masks,
        #     'cur_pixel_values': cur_pixel_values
        # }
        # # save at __embeddings
        # # Open a file and use dump() 
        # with open(os.path.join(embedding_dir, 'embed_' + str(len(embeddings)//2).zfill(5) + '.pkl'), 'wb') as file:
        #     # A new file will be created 
        #     pickle.dump(data, file) 
        # ###########################################################################

        # combine slots with learnable tokens, also cross attention them
        # patch_features = patch_features                             # (b, t*n, d)
        # Replicating tokens for each item in the batch and computing cross-attention
        # slotted_features = self.tokens(slotted_features)
        # slotted_features = self.cross_attention(slotted_features, patch_features)  # (b, 16, d)

        # aligned_slots, input_embeddings = self.transformer_encoder(slot_features=slotted_features, 
        #                                                            text_features=input_embeddings)

        # nxt_recon_loss = 0
        # if next_patch_features is not None:
        #     if s == 256:
        #         reconstruction, _, _ = self.decoder(aligned_slots, 16, 16)
        #         # (b, t*n, d),  (b, s, t*n),  (b*t, s, img_size, img_size)
        #     else:
        #         reconstruction, _, _ = self.decoder(aligned_slots, 16, 32)
        #     nxt_recon_loss = F.mse_loss(reconstruction, next_patch_features)

        #     aligned_slots = torch.concat([slotted_features, aligned_slots], dim=-1)
        #     aligned_slots = self.action_latent(slotted_features)
        # else:
        #     nxt_recon_loss = cur_recon_loss
            
        #     aligned_slots = torch.concat([slotted_features, aligned_slots], dim=-1)
        #     aligned_slots = self.action_latent(slotted_features)

        return slotted_features, input_embeddings










