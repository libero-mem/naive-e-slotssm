import functools
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from scipy.optimize import linear_sum_assignment


# Kernel Normalized Kernel
class WNConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False, norm_type="softmax", use_normed_logits=False):
        super(WNConv, self).__init__(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.norm_type = norm_type

    def forward(self, input):
        o_c, i_c, k1, k2 = self.weight.shape
        weight = torch.reshape(self.weight.data, (o_c, i_c, k1 * k2))
        weight = weight / torch.linalg.norm(weight, dim=-1, keepdim=True) # norm logits to 1.

        if 'linear' in self.norm_type:
            weight = weight / torch.sum(weight, dim=-1, keepdim=True)
        elif 'softmax' in self.norm_type:
            weight = F.softmax(weight, dim=-1)

        self.weight = nn.Parameter(torch.reshape(weight, (o_c, i_c, k1, k2)))

        # we don't recommend ver lower than 1.7
        if '1.7' in torch.__version__:
            return self._conv_forward(input, self.weight)
        else:
            return self._conv_forward(input, self.weight, self.bias)
        

class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_slots, encoder_dims, iters=3, hidden_dim=128, eps=1e-8, background=True):
        """Builds the Slot Attention module.
        Args:
            iters: Number of iterations.
            num_slots: Number of slots.
            encoder_dims: Dimensionality of slot feature vectors.
            hidden_dim: Hidden layer size of MLP.
            eps: Offset for attention coefficients before normalization.
        """
        super(SlotAttention, self).__init__()
        
        self.eps = eps
        self.iters = iters
        self.num_slots = num_slots
        self.scale = encoder_dims ** -0.5
        self.background = background
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.norm_input = nn.LayerNorm(encoder_dims)
        self.norm_slots = nn.LayerNorm(encoder_dims)
        self.norm_pre_ff = nn.LayerNorm(encoder_dims)
        self.slots_embedding = nn.Embedding(num_slots, encoder_dims)
        # Linear maps for the attention module.
        self.project_q = nn.Linear(encoder_dims, hidden_dim)
        self.project_k = nn.Linear(encoder_dims, hidden_dim)
        self.project_v = nn.Linear(encoder_dims, encoder_dims)
        # Slot update functions.
        self.gru = nn.GRUCell(encoder_dims, encoder_dims)
        self.mlp_1 = nn.Linear(encoder_dims, hidden_dim)
        self.mlp_2 = nn.Linear(hidden_dim, encoder_dims)

    def forward(self, inputs, bs=None, weight=None, init_slots=None, num_slots=None):
        # inputs has shape [batch_size, num_inputs, inputs_size].
        inputs = self.norm_input(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        # learnable slots initializations
        if init_slots == None:
            init_slots = self.slots_embedding(torch.arange(0, n_s).expand(b, n_s).to(self.device))
            if self.background:
                init_slots = torch.cat([init_slots, torch.zeros_like(init_slots[:, -1:, :])], dim=1)
            slots = init_slots
        else:
            slots = init_slots

        # Slot update.
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            # Multiple rounds of attention.
            for t in range(self.iters):
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention.
                q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
                dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
                attn = dots.softmax(dim=1) + self.eps
                attn = attn / attn.sum(dim=-1, keepdim=True)  # weighted mean.

                updates = torch.einsum('bjd,bij->bid', v, attn)

                # `updates` has shape: [batch_size, num_slots, slot_size].
                slots = self.gru(
                    updates.reshape(-1, d),
                    slots_prev.reshape(-1, d)
                )
                slots = slots.reshape(b, -1, d)
                slots = slots + self.mlp_2(F.relu(self.mlp_1(self.norm_pre_ff(slots))))
                # if t == self.iters-2:
                #     slots = slots.detach() - init_slots.detach() + init_slots
                
        return slots.to(torch.bfloat16), dots.to(torch.bfloat16)

def build_grid(resolution):
    ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        self.grid = self.grid.to(inputs.device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            grid = self.embedding(self.grid)
        return inputs + grid

    def get_pos_emb(self, pos):
        """ 
        pos (*, 2)
        """
        return self.embedding(torch.cat([pos, 1.0 - pos], dim=-1))


class Encoder(nn.Module):
    def __init__(self, resolution=[16,16], hid_dim=2176):
        super().__init__()

        self.resolution = resolution
        self.hid_dim = hid_dim
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)
        # self.layer_norm = nn.LayerNorm([resolution[0] * resolution[1], hid_dim])
        # self.mlp = nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        B, HW, D = x.shape
        assert(HW == self.resolution[0]*self.resolution[1])
        x = x.reshape(B, self.resolution[0], self.resolution[1], D) # [B, H, W ,D]
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        # x = self.layer_norm(x)
        # x = self.mlp(x)
        return x


class Decoder(nn.Module):

    def __init__(self, num_slots, init_dim, hid_dim, input_size, target_size):
        super().__init__()
        
        self.num_slots = num_slots
        self.init_dim = init_dim
        self.hid_dim = hid_dim
        self.target_size = target_size

        self.resolution = (target_size, target_size)
        self.dec_init_size = 1
        # dec_init_resolution = (self.dec_init_size, self.dec_init_size)
        # self.decoder_pos = SoftPositionEmbed(hid_dim, dec_init_resolution)
        upsample_step = int(np.log2(target_size // self.dec_init_size)) + int((target_size % self.dec_init_size) == 0)
        DEC_DEPTH = 7
        count_layer = 0 

        deconvs = nn.ModuleList()
        for _ in range(upsample_step):
            if count_layer == 0: 
                deconvs.extend([
                    nn.ConvTranspose2d(self.init_dim, self.hid_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                    nn.ReLU(),
                ])
            else: 
                deconvs.extend([
                    nn.ConvTranspose2d(self.hid_dim, self.hid_dim, 5, stride=(2, 2), padding=2, output_padding=1),
                    nn.ReLU(),
                ])
            
            count_layer += 1

        for _ in range(DEC_DEPTH - upsample_step - 1):
            
            if count_layer == 0: 
                deconvs.extend([nn.ConvTranspose2d(self.hid_dim, self.hid_dim, 5, stride=(1, 1), padding=2), nn.ReLU()])
            else: 
                deconvs.extend([nn.ConvTranspose2d(self.hid_dim, self.hid_dim, 5, stride=(1, 1), padding=2), nn.ReLU()])

            count_layer += 1

        deconvs.append(nn.ConvTranspose2d(self.hid_dim, self.init_dim+1, 3, stride=(1, 1), padding=1))
        count_layer += 1

        assert DEC_DEPTH == count_layer, "The number of layers of decoder differs from the configuration"
        self.deconvs = nn.Sequential(*deconvs)

    def forward(self, x):
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        # x: [batch_size, 16, 2176]
        x = x.reshape(-1, x.shape[-1]).unsqueeze(1).unsqueeze(2)       # x: [batch_size*16, 1, 1, 2176]
        # x = x.repeat((1, self.dec_init_size, self.dec_init_size, 1))   # x: [batch_size*16, 1, 1, 2176]
        # x = self.decoder_pos(x)
        x = x.permute(0, 3, 1, 2)                                      # x: [batch_size*16, 2176, 1, 1]
        x = self.deconvs(x)
        x = x[:, :, : self.resolution[0], : self.resolution[1]]
        x = x.permute(0, 2, 3, 1)
        return x


class PositionPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.WEAK_SUP.TYPE != ""

        slot_dim = cfg.MODEL.SLOT.DIM
        pos_num = 4 if cfg.WEAK_SUP.TYPE == 'bbox' else 2
        self.layer_norm = nn.LayerNorm(slot_dim)
        if cfg.POS_PRED.PP_SIZE == 'base':
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, slot_dim // 2), 
                nn.ReLU(), 
                nn.Linear(slot_dim // 2, slot_dim // 4), 
                nn.ReLU(), 
                nn.Linear(slot_dim // 4, pos_num)
            )
        elif cfg.POS_PRED.PP_SIZE == 'small':
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, slot_dim // 4), 
                nn.ReLU(), 
                nn.Linear(slot_dim // 4, pos_num)
            )
        elif cfg.POS_PRED.PP_SIZE == 'big':
            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, slot_dim), 
                nn.ReLU(), 
                nn.Linear(slot_dim, slot_dim // 4), 
                nn.ReLU(), 
                nn.Linear(slot_dim // 4, slot_dim // 8), 
                nn.ReLU(), 
                nn.Linear(slot_dim // 8, pos_num)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(self.layer_norm(x))) - 0.5


class PositionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        slot_dim = cfg.MODEL.SLOT.DIM
        pos_num = 4 if cfg.WEAK_SUP.TYPE == 'bbox' else 2
        self.encoder = nn.Sequential(
            nn.Linear(pos_num, slot_dim // 4), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 4, slot_dim // 2), 
            nn.ReLU(), 
            nn.Linear(slot_dim // 2, slot_dim)
        )
            
    def forward(self, x):
        return self.encoder(x)
        

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, num_slots=16, res=16, encoder_dims=2176, iters=3, hidden_dim=724, eps=1e-8, background=True):
        """Builds the Slot Attention-based auto-encoder."""
        super().__init__()
        self.obj_slots = num_slots
        self.background = background
        self.encoder_nn = Encoder(resolution=[res,res], hid_dim=encoder_dims)
        self.decoder_cnn = Decoder(num_slots=num_slots, init_dim=encoder_dims, hid_dim=hidden_dim, input_size=res, target_size=32)

        self.slot_attention = SlotAttention(num_slots=num_slots, encoder_dims=encoder_dims, 
                                            iters=iters, hidden_dim=hidden_dim, eps=eps, 
                                            background=background)
        self.criterion = nn.MSELoss()

    @property
    def num_slots(self):
        return self.obj_slots + int(self.background)

    def encode(self, patch_features):
        # `patch_features` has shape: [batch_size, feature_h*feature_w, feature_dim].
        
        # Convolutional encoder with position embedding.
        x = self.encoder_nn(patch_features)  # CNN Backbone.
        # `x` has shape: [B, height*width, hid_dim].

        # Slot Attention module.
        sa_outputs = self.slot_attention(x)
        return sa_outputs
    
    def decode(self, slotted_features, reference=None, recon_loss_only=True):
        # Interpolate the image tensor to the new size
        reference = F.interpolate(reference, size=self.decoder_cnn.resolution, mode='bilinear', align_corners=False)
        # print('slotted_features', slotted_features.shape)
        # print('reference', reference.shape)
        x = self.decoder_cnn(slotted_features)
        # `x` has shape: [B*K, height, width, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(reference.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]
                        ).split([x.shape[3]-1, 1], dim=-1)
        # `recons` has shape: [B, K, height, width, num_channels].
        # `masks` has shape: [B, K, height, width, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        # print('recons', recons.shape)
        # print('masks', masks.shape)

        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, num_channels, height, width].
        # print('recon_combined', recon_combined.shape)
        recon_loss = self.criterion(recon_combined, reference) 
        if recon_loss_only:
            return recon_loss, None
        return recon_loss, recon_combined

    def forward(self, image, pos=None, train=True):
        # `image` has shape: [batch_size, num_channels, height, width].
        
        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        # `x` has shape: [B, height*width, hid_dim].

        # Slot Attention module.
        sa_outputs = self.slot_attention(x, pos=pos, train=train)
        slots = sa_outputs["slots"]
        # `slots` has shape: [N, K, slot_dim].

        x = self.decoder_cnn(slots)
        # `x` has shape: [B*K, height, width, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]
                        ).split([3, 1], dim=-1)
        # `recons` has shape: [B, K, height, width, num_channels].
        # `masks` has shape: [B, K, height, width, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)

        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, num_channels, height, width].

        outputs = dict()
        outputs['recon_combined'] = recon_combined
        outputs['recons'] = recons
        outputs['masks'] = masks
        outputs['slots'] = slots
        outputs['attn'] = sa_outputs["attn"]
        outputs['pos_pred'] = sa_outputs["pos_pred"]
        outputs['pos_gt_aranged'] = sa_outputs['pos_gt_aranged']
        if not train: 
            outputs['attns'] = sa_outputs['attns']
            outputs['attns_origin'] = sa_outputs['attns_origin']
            # `attns`: list of (B, N_heads, N_in, K) x T

        return outputs