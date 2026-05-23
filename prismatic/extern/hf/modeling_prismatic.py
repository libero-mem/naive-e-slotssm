"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

Note =>> for the time being, not adding the custom HF "docstring" formatting.

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig, ObjectCentricVLAConfig
from .configuration_prismatic import SlotVLAConfig, SlotVLAV2Config, CustomOpenVLAConfig

# Get Logger
logger = logging.getLogger(__name__)

# === Dummy class ===
class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.zeros(1, 2, 1).to('cuda:0')
        self.past_key_values = torch.zeros(1, 2, 1).to('cuda:0')

    def __contains__(self, item):
        return hasattr(self, item)

# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


def resize_visual_features(features, upsample_rate=2, channel_reduction_rate=2):
    # assuming features : [N, P, D]
    N, P, D = features.shape
    features = F.avg_pool1d(features.transpose(1, 2), 
                            kernel_size=channel_reduction_rate, 
                            stride=channel_reduction_rate).transpose(1, 2)
    features = features.view(N, 16, 16, D//channel_reduction_rate)
    features = F.interpolate(
        features.permute(0, 3, 1, 2),
        scale_factor=upsample_rate,
        mode='bilinear',
        align_corners=False
    ).permute(0, 2, 3, 1)

    return features


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        return torch.cat([patches, patches_fused], dim=2)


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        print(self.language_model)
        # print(self.language_model); 1/0
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        token_weights = np.full(shape=self.language_model.config.vocab_size, fill_value=1.0)  # Uniform weight for all
        token_weights[31872] = 0.1  # Set weight for the frequent class, adjust index if necessary
        self.token_weights = torch.FloatTensor(token_weights)

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    def calculate_loss(self, logits, labels, weight=None):
        # tested to find that this loss (when without weight is the same as that in transformers)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss() # weight=weight
        shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])
        
        # import time
        # start = time.time()
        num_dimensions = len(pixel_values.shape)
        if num_dimensions == 4: # [bz, cc, h, w]
            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)
        else: # [bz, num_view, horizon, cc, h, w]
            # Visual Feature Extraction With Wrist Camera
            bz, num_view, horizon, cc, h, w = pixel_values.shape
            cur_pixel_values = pixel_values[:,:,0,:,:,:].reshape(-1, 6, 224, 224)
            patch_features = self.vision_backbone(cur_pixel_values)

            patch_features = patch_features.reshape(bz, -1, patch_features.shape[-1])


        # start = time.time()
        # Projection Logic =>> Update Attention Mask
        projected_patch_embeddings = self.projector(patch_features)
        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        # Get Input Embeddings (from Language Model Embeddings)
        input_embeddings = self.get_input_embeddings()(input_ids)

        # start = time.time()

        # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
            )

        # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                fill_value=IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

        # Dispatch to Language Model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.training:
            loss = self.calculate_loss(logits=language_model_output.logits, 
                                    labels=multimodal_labels, 
                                    weight=self.token_weights.to(multimodal_labels.device)) 
        else:
            loss = None

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats
        
        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of
        self.time_logs = []
    
    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around super().generate() that decodes predicted actions and de-normalizes them."""

        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        # if not torch.all(input_ids[:, -1] == 29871):
        #     input_ids = torch.cat(
        #         (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        #     )

        # Run VLA inference
        generated_ids = self.forward(input_ids=input_ids, **kwargs).logits[0:1].argmax(dim=2)

        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key)-2 :-2].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    def predict_visually(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around super().generate() that decodes predicted actions and de-normalizes them."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
        
        try:
            # Run VLA inference
            generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key), **kwargs)
            return generated_ids
        except:
            return None
        
    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None and len(norm_stats) != 1:
            raise ValueError(
                f"Your model was trained on more than one dataset. "
                f"Please pass a `unnorm_key` from the following options to choose the statistics used for "
                f"de-normalizing actions: {norm_stats.keys()}"
            )

        # If None, grab the (singular) dataset in `norm_stats` to use as `unnorm_key`
        unnorm_key = unnorm_key if unnorm_key is not None else next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {norm_stats.keys()}"
            )

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        # print('unnorm_key', unnorm_key); 1/0
        return self.norm_stats[unnorm_key]["action"]

######################################################################################
######################################################################################
######################################################################################
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

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

INIT_CONST = 0.02
class SlotAttention(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        n_slots: int,
        in_dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation
        self.feature_dim = feature_dim
        if kvq_dim is None:
            self.kvq_dim = feature_dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5

        # perceptual grouping
        self.init_proj = nn.Linear(in_dim, feature_dim, bias=use_projection_bias)
        # self.out_proj = nn.Linear(feature_dim, in_dim, bias=use_projection_bias)
        self.conditioning = LearnableConditioning(object_dim=feature_dim, n_slots=n_slots, init_const=INIT_CONST)

        self.to_q = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, feature_dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(feature_dim)
        self.ff_mlp = self.build_two_layer_mlp(input_dim=feature_dim, output_dim=feature_dim, hidden_dim=feature_dim*4,
                                            initial_layer_norm=True, residual=True)

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

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.feature_dim))

        slots = slots.reshape(bs, -1, self.feature_dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting #.mean(dim=2)

    def iterate(self, slots, k, v, masks=None):
        # Slot update.
        if self.training:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                for _ in range(self.iters):
                    slots, attn = self.step(slots, k, v, masks)
            slots, attn = slots.to(torch.bfloat16), attn.to(torch.bfloat16)
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                for _ in range(self.iters):
                    slots, attn = self.step(slots, k, v, masks)
            slots, attn = slots.to(torch.bfloat16), attn.to(torch.bfloat16)
        return slots, attn

    def forward(
        self, inputs: torch.Tensor, masks: Optional[torch.Tensor] = None,
        past_slots: torch.Tensor = None
    ):
        device = inputs.device
        b, n, d = inputs.shape
        if past_slots is None:
            slots = self.conditioning(b)
        else:
            slots = past_slots[:,0]
        inputs = self.init_proj(inputs)
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)

        # # ##########################################################################
        # # count at __embeddings
        # import os
        # import pickle
        # embedding_dir = '__embeddings_demo'
        # embeddings = os.listdir(embedding_dir)
        # # save at __embeddings
        # # Open a file and use dump() 
        # data = {
        #     'slots': slots,
        #     'attn': attn
        # }
        # # A new file will be created 
        # with open(os.path.join(embedding_dir, 'embed_' + str(len(embeddings)//2).zfill(5) + '.pkl'), 'wb') as file:
        #     pickle.dump(data, file)
        # # ##########################################################################

        upsampled_slots = None
        return slots, attn, upsampled_slots

    def get_activation_fn(self, name: str, inplace: bool = True, leaky_relu_slope: Optional[float] = 0.1):
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

    def build_mlp(
        self,
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
                layers.append(self.get_activation_fn(activation_fn))
            current_dim = n_features

        layers.append(nn.Linear(current_dim, output_dim))
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        if final_activation_fn is not None:
            layers.append(self.get_activation_fn(final_activation_fn))

        if residual:
            return Residual(nn.Sequential(*layers))
        return nn.Sequential(*layers)

    def build_two_layer_mlp(
        self, input_dim, output_dim, hidden_dim, initial_layer_norm: bool = False, residual: bool = False
    ):
        """Build a two layer MLP, with optional initial layer norm.

        Separate class as this type of construction is used very often for slot attention and
        transformers.
        """
        return self.build_mlp(
            input_dim, output_dim, [hidden_dim], initial_layer_norm=initial_layer_norm, residual=residual
        )


""" The MLP class for building linear nns for bbox predictions
"""
import torch.nn.functional as F
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

""" The mask prediction head for projecting each slot representations into a binary mask
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskPredictionHead(nn.Module):
    def __init__(self, slot_dim, hidden_dim=64, mask_size=(224, 224)):
        super().__init__()
        self.H, self.W = mask_size

        # Linear projection to conv-friendly shape
        self.proj = nn.Linear(slot_dim, hidden_dim)

        # Conv decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim + 2, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),  # Output: mask logits
        )

        # Positional encoding: 2 channels (x, y)
        self.positional_encoding = self._build_2d_pos_enc(self.H, self.W)

    def _build_2d_pos_enc(self, H, W):
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        pos = torch.stack([x, y], dim=0)  # Shape: [2, H, W]
        return pos  # Not learnable, but can be made learnable

    def forward(self, slot_features):
        """
        slot_features: [B, num_slots, slot_dim]
        Returns: binary_masks [B, num_slots, H, W]
        """
        B, num_slots, slot_dim = slot_features.shape

        # print('input', torch.min(slot_features), torch.max(slot_features))
        # Project and spatially broadcast
        slot_proj = self.proj(slot_features)  # [B, num_slots, hidden_dim]
        # print('projs', torch.min(slot_proj), torch.max(slot_proj))

        slot_proj = slot_proj.view(B * num_slots, -1, 1, 1)
        slot_proj = slot_proj.expand(-1, -1, self.H, self.W)  # [B*num_slots, hidden_dim, H, W]

        # Add positional encoding
        pos_enc = self.positional_encoding.to(slot_proj.device)  # [2, H, W]
        pos_enc = pos_enc.unsqueeze(0).expand(B * num_slots, -1, -1, -1)  # [B*num_slots, 2, H, W]
        x = torch.cat([slot_proj, pos_enc], dim=1)  # [B*num_slots, hidden_dim+2, H, W]

        logits = self.decoder(x)  # [B*num_slots, 1, H, W]
        masks = logits.view(B, num_slots, self.H, self.W)  # [B, num_slots, H, W]
        # print('masks', torch.min(masks), torch.max(masks))
        # print('sigmd', torch.min(masks.sigmoid()), torch.max(masks.sigmoid()))
        # print('')
        return masks

class OpenVLAForActionPrediction_SlotAtt(nn.Module):
    config_class: PretrainedConfig = ObjectCentricVLAConfig

    def __init__(self, config: ObjectCentricVLAConfig = None, model: OpenVLAForActionPrediction = None, number_of_slots = 16) -> None:
        if config is None and model is None:
            assert(False)
        if isinstance(config, ObjectCentricVLAConfig):
            super().__init__(config)
        elif model is not None:
            super().__init__()
            self.model = model

        self.object_token_num = number_of_slots

        # The object-centric tokenizer handles extraction of object-centric features for,
        # --- capturing object 2D coordinates and object captions
        # --- capturing object features 
        # dim 256: 0.7546
        self.object_centric_tokenizer = SlotAttention(n_slots=self.object_token_num, in_dim=2176, feature_dim=512) 
        self.object_centric_bbox_head = MLP(input_dim=512, hidden_dim=1024, output_dim=5, num_layers=2) # switch depth to 3 for good results
        self.object_centric_mask_head = MaskPredictionHead(slot_dim=512, hidden_dim=1024, mask_size=(64,64))

    def get_obj_slots_per_image(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        past_tokens:  Optional[torch.FloatTensor] = None
    ):
        """ This function receives pixel values then produces a set of object-centric slots 
                using the ObjectCentricTokenizer.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        num_dimensions = len(pixel_values.shape)
        if num_dimensions == 4: # [bz, cc, h, w]
            # Assuming Visual Feature Extraction on feature image of horizon 1, single view
            bz, horizon, num_view = pixel_values.shape[0], 1, 1
            patch_features = self.model.vision_backbone(pixel_values)

        else: # [bz, horizon, cc, h, w]
            1/0

        patch_features = patch_features.reshape(bz*horizon, -1, 2176)

        # Extract bboxes
        visual_tokens, attention, upsampled_visual_tokens = self.object_centric_tokenizer(patch_features, 
                                                                                          past_slots=past_tokens)
        if not self.training:
            return visual_tokens, None

        # print(patch_features.shape, visual_tokens.shape, torch.min(visual_tokens), torch.max(visual_tokens))
        bboxes = self.object_centric_bbox_head(visual_tokens).sigmoid()
        masks = self.object_centric_mask_head(visual_tokens)

        bboxes = torch.stack([bboxes], dim=1)
        masks = torch.stack([masks], dim=1)

        output = {}
        output['visual_tokens'] = visual_tokens.reshape([bz, horizon, self.object_token_num, -1])
        output['attentions'] = attention.reshape([bz, horizon, self.object_token_num, -1])
        output['bboxes'] = bboxes.reshape([bz, horizon, self.object_token_num, 5])
        output['masks'] = masks.reshape([bz, horizon, *masks.shape[2:]])

        return output

    def get_obj_slots(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        """ This function receives pixel values in a horizon to output the corresponding visual representations.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        output = {'visual_tokens': [], 'attentions': [], 'bboxes': [], 'masks': []}
        bz, horizon = pixel_values.shape[:2]
        past_tokens = None
        for h in range(horizon):
            suboutput = self.get_obj_slots_per_image(pixel_values[:,h], past_tokens)
            if 'visual_tokens' in suboutput:
                past_tokens = suboutput['visual_tokens'].detach()
            for key, value in suboutput.items():
                output[key].append(value)

        for key, value in output.items():
            output[key] = torch.cat(value, axis=1) # concatenate by horizon

        return output

from prismatic.models.addons.multimodal_transformer import AfdMultimodalTransformer
class OpenVLAForActionPrediction_LangSlotAtt(nn.Module):
    config_class: PretrainedConfig = ObjectCentricVLAConfig

    def __init__(self, config: ObjectCentricVLAConfig = None, model: OpenVLAForActionPrediction = None, number_of_slots = 16) -> None:
        if config is None and model is None:
            assert(False)
        if isinstance(config, ObjectCentricVLAConfig):
            super().__init__(config)
        elif model is not None:
            super().__init__()
            self.model = model

        self.object_token_num = number_of_slots

        # The object-centric tokenizer handles extraction of object-centric features for,
        # --- capturing object 2D coordinates and object captions
        # --- capturing object features 
        # dim 256: 0.7546
        self.object_centric_tokenizer = SlotAttention(n_slots=self.object_token_num, in_dim=2176, feature_dim=512) 
        self.object_centric_bbox_head = MLP(input_dim=512, hidden_dim=1024, output_dim=5, num_layers=2) # switch depth to 3 for good results
        self.object_centric_mask_head = MaskPredictionHead(slot_dim=512, hidden_dim=1024, mask_size=(64,64))

        # language tokenizer
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.object_centric_text_encoder = CLIPBasedTextEncoder(self.clip_model, self.clip_preprocess)
        # cross attention
        self.object_centric_afd_interact_head = AfdMultimodalTransformer(d_model=512, num_encoder_layers=3)


    def get_obj_slots_per_image(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        past_tokens:  Optional[torch.FloatTensor] = None
    ):
        """ This function receives pixel values then produces a set of object-centric slots 
                using the ObjectCentricTokenizer.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        num_dimensions = len(pixel_values.shape)
        if num_dimensions == 4: # [bz, cc, h, w]
            # Assuming Visual Feature Extraction on feature image of horizon 1, single view
            bz, horizon, num_view = pixel_values.shape[0], 1, 1
            patch_features = self.model.vision_backbone(pixel_values)

        else: # [bz, horizon, cc, h, w]
            1/0

        patch_features = patch_features.reshape(bz*horizon, -1, 2176)

        # Extract bboxes
        visual_tokens, attention, upsampled_visual_tokens = self.object_centric_tokenizer(patch_features, 
                                                                                          past_slots=past_tokens)
        if not self.training:
            return visual_tokens, None

        # print(patch_features.shape, visual_tokens.shape, torch.min(visual_tokens), torch.max(visual_tokens))
        bboxes = self.object_centric_bbox_head(visual_tokens).sigmoid()
        masks = self.object_centric_mask_head(visual_tokens)

        bboxes = torch.stack([bboxes], dim=1)
        masks = torch.stack([masks], dim=1)

        output = {}
        output['visual_tokens'] = visual_tokens.reshape([bz, horizon, self.object_token_num, -1])
        output['attentions'] = attention.reshape([bz, horizon, self.object_token_num, -1])
        output['bboxes'] = bboxes.reshape([bz, horizon, self.object_token_num, 5])
        output['masks'] = masks.reshape([bz, horizon, *masks.shape[2:]])

        return output

    def get_obj_slots(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        texts: Optional[str] = None,
    ):
        """ This function receives pixel values in a horizon to output the corresponding visual representations.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        output = {'visual_tokens': [], 'attentions': [], 'bboxes': [], 'masks': [], 'interact': []}
        
        device = pixel_values.device
        bz, horizon = pixel_values.shape[:2]
        past_tokens = None
        general_texts, texts, texts_attn = self.object_centric_text_encoder(texts, temporal_length=1, device=device)
        for h in range(horizon):
            suboutput = self.get_obj_slots_per_image(pixel_values[:,h], past_tokens)
            interactability = self.object_centric_afd_interact_head(suboutput['visual_tokens'].squeeze(1), 
                                                                    texts.squeeze(1)) #, attention=texts_attn)
            suboutput['interact'] = torch.unsqueeze(interactability, dim=1)
            if 'visual_tokens' in suboutput:
                past_tokens = suboutput['visual_tokens'].detach()
            for key, value in suboutput.items():
                output[key].append(value)

        for key, value in output.items():
            output[key] = torch.cat(value, axis=1) # concatenate by horizon

        return output


""" From here begins the modeling code for SlotSSM into OpenVLA (but without action modeling yet)
"""
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0., inverted=False, bias=True,
                 norm_over_input=True, epsilon=1e-5, d_model_hidden=None):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.inverted = inverted
        self.norm_over_input = norm_over_input

        self.epsilon = epsilon

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        if d_model_hidden is None:
            d_model_hidden = d_model

        self.proj_q = nn.Linear(d_model, d_model_hidden, bias=bias)
        self.proj_k = nn.Linear(d_model, d_model_hidden, bias=bias)
        self.proj_v = nn.Linear(d_model, d_model_hidden, bias=bias)
        self.proj_o = nn.Linear(d_model_hidden, d_model, bias=bias)

    def forward(self, q, k, v, attn_mask=None, attn_bias=None, output_attentions=False):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model·
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape

        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)

        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_bias is not None:
            attn = attn + attn_bias

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))

        if self.inverted:
            attn = F.softmax(attn.flatten(start_dim=1, end_dim=2), dim=1).reshape(B, self.num_heads, T, S)
            attn_vis = attn.detach()

            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask, float('0.'))

            # attn /= attn.sum(dim=-1, keepdim=True) + self.epsilon
            if self.norm_over_input:
                attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
        else:
            # print('attn b4', attn.shape, torch.min(attn), torch.max(attn))
            attn = F.softmax(attn, dim=-1)
            # print('attn af', attn.shape, torch.min(attn), torch.max(attn))
            attn_vis = attn.detach()

        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)

        outputs = (output, attn_vis) if output_attentions else (output,)
        return outputs

""" CLIP text encoder for open vocab chunks
"""
TASK_DECOMP = {
    'pick the bowl from the plate and place it back 7 times': ['pick', 'bowl', 'from plate', 'place', 'it', '7 times'], 
    'pick the bowl from the plate and place it back 5 times':  ['pick', 'bowl', 'from plate', 'place', 'it', '5 times'], 
    'pick the bowl from the plate and place it back 3 times':  ['pick', 'bowl', 'from plate', 'place', 'it', '3 times'], 
    'pick the bowl from the plate and place it back 1 time':   ['pick', 'bowl', 'from plate', 'place', 'it', '1 time'],
    'swap the 3 bowls from left to right using the intermediary plate': ['swap', '3 bowls', 'from left to right', 'using', 'intermediary plate'],
    'swap the 2 bowls using the intermediary plate': ['swap', '2 bowls', 'using', 'intermediary plate'],

    'robot open the middle drawer of the cabinet': ['robot', 'open', 'middle drawer', 'of', 'cabinet'],
    'robot open the top drawer and put the bowl inside': ['robot', 'open', 'top drawer', 'and', 'put', 'bowl', 'inside'],
    'robot push the plate to the front of the stove': ['robot', 'push', 'plate', 'to', 'front of', 'stove'],
    'robot put the bowl on the plate': ['robot', 'put', 'bowl', 'on', 'plate'],
    'robot put the bowl on the stove': ['robot', 'put', 'bowl', 'on', 'stove'],
    'robot put the bowl on top of the cabinet': ['robot', 'put', 'bowl', 'on top of', 'cabinet'],
    'robot put the cream cheese in the bowl': ['robot', 'put', 'cream cheese', 'in', 'bowl'],
    'robot put the wine bottle on the rack': ['robot', 'put', 'wine bottle', 'on', 'rack'],
    'robot put the wine bottle on top of the cabinet': ['robot', 'put', 'wine bottle', 'on top of', 'cabinet'],
    'robot turn on the stove': ['robot', 'turn on', 'stove'],

    "robot turn on the stove and put the moka pot on it": ['robot', 'stove', 'moka pot'],
    "robot put the black bowl in the bottom drawer of the cabinet and close it": ['robot', 'bowl', 'cabinet'],
    "robot put the yellow and white mug in the microwave and close it": ['robot', 'yellow and white mug', 'microwave'],
    "robot put both moka pots on the stove": ['robot', 'moka pot', 'stove'],
    "robot put both the alphabet soup and the cream cheese box in the basket": ['robot', 'alphabet soup', 'cream cheese'],
    "robot put both the alphabet soup and the tomato sauce in the basket": ['robot', 'alphabet soup', 'tomato'],
    "robot put both the cream cheese box and the butter in the basket": ['robot', 'cream cheese', 'butter', 'basket'],
    "robot put the white mug on the left plate and put the yellow and white mug on the right plate": ['robot', 'white mug', 'plate', 'yellow and white mug'],
    "robot put the white mug on the plate and put the chocolate pudding to the right of the plate": ['robot', 'white mug', 'plate', 'chocolate pudding'],
    "robot pick up the book and place it in the back compartment of the caddy": ['robot', 'book', 'caddy'],

    "robot pick up the bowl and place it back on the plate": ['robot', 'pick up', 'bowl', 'place back on', 'plate'],
    "robot lift the bottle and put it down on the plate": ['robot', 'lift', 'bottle', 'put down on', 'plate'],
    "robot lift the bowl and place it back on the plate 3 times": ['robot', 'lift', 'bowl', 'place back on', 'plate', 'three times'],
    "robot pick up the bottle and put it down the plate 3 times": ['robot', 'pick up', 'bottle', 'put down', 'plate', 'three times'],
    "robot lift the bowl and place it back on the plate 5 times": ['robot', 'lift', 'bowl', 'place back on', 'plate', 'five times'],
    "robot pick up the bowl and place it on the plate 7 times": ['robot', 'pick up', 'bowl', 'place on', 'plate', 'seven times'],
    "robot swap the 2 bowls on their plates using the empty plate": ['robot', 'swap', '2 bowls', 'on their plates', 'using', 'empty plate'],
    "robot rotate the 3 bowls on their plates from left to right using the empty plate": ['robot', 'rotate', '3 bowls', 'on their plates', 'from left to right', 'using', 'empty plate'],
    "robot put the cream cheese in the nearest basket and place that basket in the center": ['robot', 'put', 'cream cheese', 'in', 'nearest basket', 'place', 'that basket', 'in the center'],
    "robot put the cream cheese in the nearest basket and place the empty basket in the center": ['robot', 'put', 'cream cheese', 'in', 'nearest basket', 'place', 'empty basket', 'in the center'],

    "robot pick up the black bowl between the plate and the ramekin and place it on the plate": ['robot', 'bowl', 'plate', 'ramekin'],
    "robot pick up the black bowl from table center and place it on the plate": ['robot', 'bowl', 'plate'],
    "robot pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": ['robot', 'bowl', 'cabinet', 'plate'],
    "robot pick up the black bowl next to the cookie box and place it on the plate": ['robot', 'bowl', 'cookies', 'plate'],
    "robot pick up the black bowl next to the plate and place it on the plate": ['robot', 'bowl', 'plate'],
    "robot pick up the black bowl next to the ramekin and place it on the plate": ['robot', 'bowl', 'ramekin', 'plate'],
    "robot pick up the black bowl on the cookie box and place it on the plate": ['robot', 'bowl', 'cookies', 'plate'],
    "robot pick up the black bowl on the ramekin and place it on the plate": ['robot', 'bowl', 'ramekin', 'plate'],
    "robot pick up the black bowl on the stove and place it on the plate": ['robot', 'bowl', 'stove', 'plate'],
    "robot pick up the black bowl on the wooden cabinet and place it on the plate": ['robot', 'bowl', 'cabinet', 'plate'],

    "robot pick up the butter and place it in the basket": ['robot', 'pick up', 'butter', 'place', 'in', 'basket'],
    "robot pick up the bbq sauce and place it in the basket": ['robot', 'pick up', 'bbq sauce', 'place', 'in', 'basket'],
    "robot pick up the cream cheese and place it in the basket": ['robot', 'pick up', 'cream cheese', 'place', 'in', 'basket'],
    "robot pick up the salad dressing and place it in the basket": ['robot', 'pick up', 'salad dressing', 'place', 'in', 'basket'],
    "robot pick up the orange juice and place it in the basket": ['robot', 'pick up', 'orange juice', 'place', 'in', 'basket'],
    "robot pick up the alphabet soup and place it in the basket": ['robot', 'pick up', 'alphabet soup', 'place', 'in', 'basket'],
    "robot pick up the tomato sauce and place it in the basket": ['robot', 'pick up', 'tomato sauce', 'place', 'in', 'basket'],
    "robot pick up the ketchup and place it in the basket": ['robot', 'pick up', 'ketchup', 'place', 'in', 'basket'],
    "robot pick up the chocolate pudding and place it in the basket": ['robot', 'pick up', 'chocolate pudding', 'place', 'in', 'basket'],
    "robot pick up the milk and place it in the basket": ['robot', 'pick up', 'milk', 'place', 'in', 'basket'],

    "pick up the black bowl between the plate and the ramekin and place it on the plate": ['robot', 'bowl', 'plate', 'ramekin'],
    "pick up the black bowl from table center and place it on the plate": ['robot', 'bowl', 'plate'],
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": ['robot', 'bowl', 'cabinet', 'plate'],
    "pick up the black bowl next to the cookie box and place it on the plate": ['robot', 'bowl', 'cookies', 'plate'],
    "pick up the black bowl next to the plate and place it on the plate": ['robot', 'bowl', 'plate'],
    "pick up the black bowl next to the ramekin and place it on the plate": ['robot', 'bowl', 'ramekin', 'plate'],
    "pick up the black bowl on the cookie box and place it on the plate": ['robot', 'bowl', 'cookies', 'plate'],
    "pick up the black bowl on the ramekin and place it on the plate": ['robot', 'bowl', 'ramekin', 'plate'],
    "pick up the black bowl on the stove and place it on the plate": ['robot', 'bowl', 'stove', 'plate'],
    "pick up the black bowl on the wooden cabinet and place it on the plate": ['robot', 'bowl', 'cabinet', 'plate'],

    "robot pick up mayo and place in the blue bowl": ['robot', 'pick up', 'mayo', 'place in', 'blue bowl'],
    "robot place mayo and mustard in the blue bowl": ['robot', 'place', 'mayo', 'and', 'mustard', 'in', 'blue bowl'],
    "robot pick up mayo and move it to the right of the blue bowl": ['robot', 'pick up', 'mayo', 'move it', 'to right of', 'blue bowl'],
}
import clip
class CLIPBasedTextEncoder:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def get_text_decomposition(self, text):
        return TASK_DECOMP[text]

    def encode(self, texts, device):
        all_masks = []
        all_feats = []
        all_feats_gen = []

        for text in texts:
            decomposed = self.get_text_decomposition(text)
            with torch.no_grad():
                tokenized = clip.tokenize(decomposed).to(device)
                text_features = self.model.encode_text(tokenized)
                masks = torch.zeros(text_features.shape[0], dtype=torch.bool, device=device)
                all_masks.append(masks)
                all_feats.append(text_features)
                all_feats_gen.append(
                    self.model.encode_text(
                        clip.tokenize([text]).to(device)
                    )
                )

        # Pad features and masks to the same length
        max_chunks = max(len(seq) for seq in all_feats)
        feat_dim = all_feats[0].shape[-1]

        padded_features = []
        padded_masks = []

        for feats, masks in zip(all_feats, all_masks):
            token_len = feats.shape[0]
            pad_len = max_chunks - token_len
            pad_feats = torch.zeros((pad_len, feat_dim), device=device)
            pad_masks = torch.ones(pad_len, dtype=torch.bool, device=device) # ones for -infinity masking

            feats = torch.cat([feats, pad_feats])
            masks = torch.cat([masks, pad_masks])

            padded_features.append(feats)
            padded_masks.append(masks)

        features_gen_tensor = torch.stack(all_feats_gen) # [B, feat_dim]
        features_tensor = torch.stack(padded_features)   # [B, max_chunks, feat_dim]
        masks_tensor = torch.stack(padded_masks)         # [B, max_chunks]

        return features_gen_tensor, features_tensor, masks_tensor

    def encode_simple(self, texts, device):
        all_feats = []

        for text in texts:
            decomposed = [text]
            with torch.no_grad():
                tokenized = clip.tokenize(decomposed).to(device)
                text_features = self.model.encode_text(tokenized)
                all_feats.append(text_features)

        features_tensor = torch.stack(all_feats)  # [B, max_chunks, feat_dim]
        return features_tensor

    def __call__(self, texts, temporal_length, device):
        features_gen_tensor, features_tensor, masks_tensor = self.encode(texts, device)
        # Expand across time
        features_aligned = features_tensor.unsqueeze(1).expand(-1, temporal_length, -1, -1)  # [B, T, C, feat_dim]
        masks_aligned = masks_tensor.unsqueeze(1).expand(-1, temporal_length, -1)  # [B, T, C]
        return features_gen_tensor, features_aligned, masks_aligned

""" Here starts modeling for SSM
"""
from einops import rearrange
from mamba_ssm import Mamba, Mamba2
from flash_attn.modules.mha import MHA as FlashMHA

class EmbodiedSlotSSMBlock(nn.Module):
    def __init__(self, d_model, space_attn_num_heads, visual_d_model=None, textual_d_model=None, use_ffn=False,
                 encoder_attn_num_heads=None, use_cross_attn=True, use_inverted_attention=False,
                 mamba_version='mamba2', layer_idx=None, attn_impl="flash_attention_2",
                 mamba_d_state=128, mamba_d_conv=4, mamba_expand=2, mamba_headdim=64,
                 lookback_only=False):
        super().__init__()
        assert mamba_version in ['mamba1', 'mamba2'], "Mamba version must be mamba1 or mamba2"
        assert attn_impl in ["flash_attention_2", "eager"], \
            "Attention implementation must be flash_attention_2 or eager"
        self.attn_impl = attn_impl
        self.lookback_only = lookback_only
        if use_cross_attn:
            self.visual_cross_attn_input_norm = nn.LayerNorm(d_model)
            self.visual_cross_attn_ref_norm = nn.LayerNorm(d_model)
            self.visual_proj = nn.Linear(visual_d_model, d_model) if visual_d_model is not None else None
            if not lookback_only:
                self.textual_cross_attn_input_norm = nn.LayerNorm(d_model)
                self.textual_cross_attn_ref_norm = nn.LayerNorm(d_model)
                self.textual_proj = nn.Linear(textual_d_model, d_model) if textual_d_model is not None else None
            if use_inverted_attention:
                if attn_impl == "flash_attention_2":
                    mprint("Inverted attention with flash attention 2 is not supported, using eager instead")
                self.visual_cross_attn = MultiHeadAttention(
                    d_model=d_model,
                    num_heads=encoder_attn_num_heads,
                    inverted=True
                )

                if not lookback_only:
                    self.textual_cross_attn = MultiHeadAttention(
                        d_model=d_model,
                        num_heads=encoder_attn_num_heads,
                        inverted=True
                    )
            else:
                if attn_impl == "flash_attention_2":
                    self.visual_cross_attn = FlashMHA(
                        embed_dim=d_model,
                        num_heads=encoder_attn_num_heads,
                        cross_attn=True
                    )
                    if not lookback_only:
                        self.textual_cross_attn = MultiHeadAttention(
                            d_model=d_model,
                            num_heads=encoder_attn_num_heads,
                            inverted=False,
                            # output_attentions=True
                        )

                else:
                    self.visual_cross_attn = MultiHeadAttention(
                        d_model=d_model,
                        num_heads=encoder_attn_num_heads,
                        inverted=False,
                        # output_attentions=True
                    )
                    if not lookback_only:
                        self.textual_cross_attn = MultiHeadAttention(
                            d_model=d_model,
                            num_heads=encoder_attn_num_heads,
                            inverted=False,
                            # output_attentions=True
                        )

        self.time_mixer_norm = nn.LayerNorm(d_model)
        if mamba_version == 'mamba2':
            # need to check if d_model * expand / headdim = multiple of 8
            assert (d_model * mamba_expand / mamba_headdim) % 8 == 0, "d_model * expand must be a multiple of headdim"
            self.time_mixer = Mamba2(
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                headdim=mamba_headdim,
                layer_idx=layer_idx
            )
        else:
            self.time_mixer = Mamba(
                d_model=d_model,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                layer_idx=layer_idx
            )

        self.space_attn_norm = nn.LayerNorm(d_model)
        if attn_impl == "flash_attention_2":
            self.space_attn = FlashMHA(
                embed_dim=d_model,
                num_heads=space_attn_num_heads
            )
        else:
            self.space_attn = MultiHeadAttention(
                d_model=d_model,
                num_heads=space_attn_num_heads
            )

        if use_ffn:
            self.ffn_norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            )

    def single_modality_cross_attn(self, input, ref, ref_attn=None,
                                   input_proj_fn=None,
                                   cross_attn_input_norm_fn=None,
                                   cross_attn_ref_norm_fn=None,
                                   cross_attn_fn=None,
                                   output_attentions=False, debug=False):
        """
        input: B, T, N, D (slots)
        ref: B, T, L, D (reference input for cross attention)
        ref_attn: B, T, L (mask attention)
        input_proj_fn: projection function for input
        cross_attn_input_norm: normalization function for input
        cross_attn_ref_norm: normalization function for reference
        cross_attn_fn: cross attention function for reference
        """
        B, T, N, D = input.shape
        input_reshape = rearrange(input, 'b t n d -> (b t) n d')
        x = cross_attn_input_norm_fn(input_reshape)
        if debug:
            print('te_x', x.shape, torch.min(x), torch.max(x))

        ref_reshape = rearrange(ref, 'b t n d -> (b t) n d')
        if input_proj_fn is not None:
            ref_proj = input_proj_fn(ref_reshape)
        else:
            ref_proj = ref_reshape
        ref_proj = cross_attn_ref_norm_fn(ref_proj)
        if debug:
            print('te_ref', ref_proj.shape, torch.min(ref_proj), torch.max(ref_proj))

        if ref_attn is not None:
            ref_attn = rearrange(ref_attn, 'b t n -> (b t) n', b=B, t=T).unsqueeze(1).unsqueeze(1)
        
        if isinstance(cross_attn_fn, FlashMHA):
            output_attn = cross_attn_fn(x=x, x_kv=ref_proj)
            input = input + rearrange(output_attn, '(b t) n d -> b t n d', b=B, t=T)
        else:
            cross_attn_out = cross_attn_fn(
                x, ref_proj, ref_proj, attn_mask=ref_attn, output_attentions=output_attentions
            )
            if isinstance(cross_attn_out, tuple) and len(cross_attn_out) > 1:
                output_attn = cross_attn_out[1]
            else:
                output_attn = cross_attn_out[0]
            if debug:
                print('te_att', output_attn.shape, torch.min(output_attn), torch.max(output_attn))
            input = input + rearrange(output_attn, '(b t) n d -> b t n d', b=B, t=T)
        
        if not output_attentions:
            return input, None    
        return input, output_attn


    def forward(self, input, visual_ref, textual_ref=None, textual_attn=None, cache_params=None, output_attentions=False):
        """
        input: B, T, N, D (slots)
        visual_ref: B, T, VL, D (reference input for cross attention)
        textual_ref: B, TL, D (reference input for cross attention)
        """
        B, T, N, D = input.shape
        # Cross attention is enabled and ref is provided
        output_attn = None
        if not self.lookback_only:
            input, tex_output_attn = self.single_modality_cross_attn(input, ref=textual_ref, ref_attn=textual_attn,
                                                    input_proj_fn=self.textual_proj,
                                                    cross_attn_input_norm_fn=self.textual_cross_attn_input_norm,
                                                    cross_attn_ref_norm_fn=self.textual_cross_attn_ref_norm,
                                                    cross_attn_fn=self.textual_cross_attn,
                                                    output_attentions=output_attentions, debug=False)
            # print('te', input.shape, torch.min(input), torch.max(input))
        input, vis_output_attn = self.single_modality_cross_attn(input, ref=visual_ref,
                                                input_proj_fn=self.visual_proj,
                                                cross_attn_input_norm_fn=self.visual_cross_attn_input_norm,
                                                cross_attn_ref_norm_fn=self.visual_cross_attn_ref_norm,
                                                cross_attn_fn=self.visual_cross_attn,
                                                output_attentions=output_attentions)
        # if not self.lookback_only:
        #     print('vi', input.shape, torch.min(input), torch.max(input))
        output_attn = vis_output_attn

        # Time mixing (Mamba)
        input_reshape = rearrange(input, 'b t n d -> (b n) t d')
        x = self.time_mixer_norm(input_reshape)
        time_mixed = self.time_mixer(x, inference_params=cache_params)
        input = input + rearrange(time_mixed, '(b n) t d -> b t n d', b=B, n=N)
        # if not self.lookback_only:
        #     print('tm', input.shape, torch.min(input), torch.max(input))

        # Space attention
        input_reshape = rearrange(input, 'b t n d -> (b t) n d')
        x = self.space_attn_norm(input_reshape)
        if self.attn_impl == "flash_attention_2":
            space_attn_out = self.space_attn(x)
        else:
            space_attn_out = self.space_attn(x, x, x)[0]
        input = input + rearrange(space_attn_out, '(b t) n d -> b t n d', b=B, t=T)
        # if not self.lookback_only:
        #     print('ss', input.shape, torch.min(input), torch.max(input))

        if hasattr(self, 'ffn_norm'):
            input_norm = self.ffn_norm(input)
            input = input + self.ffn(input_norm)
        # if not self.lookback_only:
        #     print('ff', input.shape, torch.min(input), torch.max(input))

        return input if not output_attentions else (input, output_attn)

class EmbodiedSlotSSM(nn.Module):
    def __init__(
        self,
        num_slots: int = 32,
        num_blocks: int = 4,
        d_model: int = 256,
        d_input: int = 2176,
        d_output: int = None,
        visual_d_model: int = None,
        textual_d_model: int = None,
        use_cross_attn: bool = True,
        space_attn_num_heads: int = None,
        use_inverted_attention: bool = False,
        encoder_attn_num_heads: int = None,
        use_ffn: bool = False,
        mamba_version: str = "mamba2",
        attn_impl: str = "flash_attention_2",
        mamba_d_state: int = 128,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 64,
        lookback_only: bool = False,
        **kwargs
    ):
        super().__init__()

        if d_input is not None:
            self.in_proj = nn.Linear(d_input, d_model)
        else:
            self.in_proj = None
        self.blocks = nn.ModuleList([
            EmbodiedSlotSSMBlock(
                d_model=d_model,
                visual_d_model=visual_d_model,
                textual_d_model=textual_d_model,
                space_attn_num_heads=space_attn_num_heads,
                use_inverted_attention=use_inverted_attention,
                encoder_attn_num_heads=encoder_attn_num_heads,
                use_ffn=use_ffn,
                use_cross_attn=use_cross_attn,
                mamba_version=mamba_version,
                attn_impl=attn_impl,
                mamba_d_state=mamba_d_state,
                mamba_d_conv=mamba_d_conv,
                mamba_expand=mamba_expand,
                mamba_headdim=mamba_headdim,
                layer_idx=idx,
                lookback_only=lookback_only
            )
            for idx in range(num_blocks)
        ])
        self.init_slots = nn.Parameter(torch.randn(1, 1, num_slots, d_model))

        if d_output is not None:
            self.out_proj = nn.Linear(d_model, d_output)
        else:
            self.out_proj = None


    def forward(self, slots, visual_ref=None, textual_ref=None, textual_att=None, cache_params=None, output_attentions=False):
        """
        slots: B, T, N, D
        visual_ref: B, T, VL, D
        textual_ref: B, TL, D
        textual_att: B, TL
        """
        B, T, N, D = slots.shape
        if self.in_proj is not None:
            slots = self.in_proj(slots)
        slot_lists = []
        attentions = []
        for block in self.blocks:
            slots = block(slots, visual_ref, textual_ref, textual_att, cache_params=cache_params, output_attentions=output_attentions)
            if isinstance(slots, tuple):
                slots, attn = slots
                attentions.append(attn)
            slot_lists.append(slots)          

        if cache_params is not None:
            cache_params.seqlen_offset += T

        if self.out_proj is not None:
            slots = self.out_proj(slots)
        # print('out slots', torch.min(slots), torch.max(slots))
        return slots, attentions, slot_lists

from prismatic.models.addons.relational_module import RelationTokensGrounding

#####################################################################################################
#####################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLP(nn.Module):
    """
    Residual MLP with N blocks ⇒ each block:
        LayerNorm → Linear(d→h) → GELU → Dropout → Linear(h→d) + skip
    After the last block: LayerNorm → Linear → logits
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list | tuple,          # e.g. (256, 128) for two blocks
        n_classes: int = 1,                 # >1 for soft-max
        dropout: float = 0.1,
    ):
        super().__init__()
        d = in_dim
        blocks = []

        for h in hidden_dims:               # build each residual block
            blocks += [
                nn.LayerNorm(d),
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h, d),
            ]
        self.blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, n_classes),
        )

        # optional: Kaiming init for all Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=F.gelu(torch.tensor(0.0)).item())
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # each residual block is five layers; step over them in chunks of 5
        for i in range(0, len(self.blocks), 5):
            blk = self.blocks[i : i + 5]
            x = x + blk[-1](blk[-2](blk[-3](blk[-4](blk[-5](x)))))  # skip-conn
        return self.head(x)

#####################################################################################################
#####################################################################################################
#####################################################################################################


def visual_pool(features: torch.Tensor, reduction: int) -> torch.Tensor:
    """
    Pools the visual features by reducing the sequence length v by 'reduction' times.

    Args:
        features (torch.Tensor): Input tensor of shape (batch_size, v, d)
        reduction (int): Reduction factor (e.g., 4, 16)

    Returns:
        torch.Tensor: Pooled tensor of shape (batch_size, v // reduction, d)
    """
    bz, v, d = features.shape
    if v % reduction != 0:
        raise ValueError(f"v ({v}) must be divisible by reduction ({reduction}).")

    # Reshape: group v into reduction groups
    features = features.view(bz, v // reduction, reduction, d)

    # Average over the reduction dimension
    pooled = features.mean(dim=2)

    return pooled

class SlotFusion(nn.Module):
    def __init__(self, input_dims, embed_dims, output_dim=None, normalize=True):
        """
        Args:
            input_dims (list of int): List of input dimensions [d1, d2, d3, ...].
            embed_dims (list of int): List of embed dimensions [d1, d2, d3, ...] before fusion.
            normalize (bool): Whether to apply LayerNorm to each input before fusion.
        """
        super().__init__()
        self.normalize = normalize
        self.projectors = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i, d in enumerate(input_dims):
            embed_dim = embed_dims[i]
            # self.projectors.append(nn.Linear(d, embed_dim))
            self.projectors.append(nn.Linear(d, embed_dim))
            if normalize:
                self.norms.append(nn.LayerNorm(d))
            else:
                self.norms.append(nn.Identity())

        self.out_projection = None
        if output_dim is not None:
            self.out_projection = nn.Linear(sum(embed_dims), output_dim)

    def forward(self, inputs):
        """
        Args:
            inputs (list of tensors): Each tensor of shape (batch_size, v, d_i)

        Returns:
            Tensor: Fused tensor of shape (batch_size, v, total_dim)
        """
        fused = []
        for x, proj, norm in zip(inputs, self.projectors, self.norms):
            x = norm(x)
            x = proj(x)
            fused.append(x)

        # Concatenate along the feature dimension
        out = torch.cat(fused, dim=-1)
        if self.out_projection is not None:
            out = self.out_projection(out)
        return out

from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead

ACTION_DIM = 7
NUM_ACTIONS_CHUNK = 5


class EmbodiedObject_LangSlot(nn.Module):
    config_class: PretrainedConfig = ObjectCentricVLAConfig

    def __init__(self, config: ObjectCentricVLAConfig = None, base_model: OpenVLAForActionPrediction = None,
                 use_slotgoals=False, number_of_slots=16) -> None:
        if config is None and base_model is None:
            assert(False)
        if isinstance(config, ObjectCentricVLAConfig):
            super().__init__(config)
        elif base_model is not None:
            super().__init__()
            self.base_model = base_model

        self.object_token_num = number_of_slots # 24
        self.use_slotgoals = use_slotgoals
        # The object-centric tokenizer handles extraction of object-centric features for,
        # --- capturing object 2D coordinates and object captions
        # --- capturing object features 
        # dim 256: 0.7546
        self.object_centric_tokenizer = SlotAttention(n_slots=self.object_token_num, in_dim=2176, feature_dim=512) 
        self.object_centric_bbox_head = MLP(input_dim=512, hidden_dim=1024, output_dim=5, num_layers=2) # switch depth to 3 for good results
        self.object_centric_mask_head = MaskPredictionHead(slot_dim=512, hidden_dim=1024, mask_size=(64,64))
        self.object_centric_slot_projector = PrismaticProjector(use_fused_vision_backbone=False, vision_dim=512, llm_dim=4096)

        # language tokenizer
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.object_centric_text_encoder = CLIPBasedTextEncoder(self.clip_model, self.clip_preprocess)

        # cross attention
        self.object_centric_afd_interact_head = AfdMultimodalTransformer(d_model=512, num_encoder_layers=3)

        self.object_centric_clip_text_projector = PrismaticProjector(use_fused_vision_backbone=False, vision_dim=512, llm_dim=4096)
        self.object_centric_action_head = L1RegressionActionHead(
            input_dim=4096,
            hidden_dim=4096,
            action_dim=7,
        )

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self.base_model._check_unnorm_key(self.norm_stats, unnorm_key)
        # print('unnorm_key', unnorm_key); 1/0
        return self.norm_stats[unnorm_key]["action"]

    def get_obj_slots_per_image(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        past_tokens:  Optional[torch.FloatTensor] = None
    ):
        """ This function receives pixel values then produces a set of object-centric slots 
                using the ObjectCentricTokenizer.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        num_dimensions = len(pixel_values.shape)
        if num_dimensions == 4: # [bz, cc, h, w]
            # Assuming Visual Feature Extraction on feature image of horizon 1, single view
            bz, horizon, num_view = pixel_values.shape[0], 1, 1
            patch_features = self.base_model.vision_backbone(pixel_values)

        else: # [bz, horizon, cc, h, w]
            1/0

        patch_features = patch_features.reshape(bz*horizon, -1, 2176)

        # Extract bboxes
        visual_tokens, attention, upsampled_visual_tokens = self.object_centric_tokenizer(patch_features, 
                                                                                          past_slots=past_tokens)
        if not self.training:
            return visual_tokens, None

        # print(patch_features.shape, visual_tokens.shape, torch.min(visual_tokens), torch.max(visual_tokens))
        bboxes = self.object_centric_bbox_head(visual_tokens).sigmoid()
        masks = self.object_centric_mask_head(visual_tokens)

        bboxes = torch.stack([bboxes], dim=1)
        masks = torch.stack([masks], dim=1)

        output = {}
        output['patch_features'] = patch_features.reshape([bz, horizon, -1, 2176])
        output['visual_tokens'] = visual_tokens.reshape([bz, horizon, self.object_token_num, -1])
        output['attentions'] = attention.reshape([bz, horizon, self.object_token_num, -1])
        output['bboxes'] = bboxes.reshape([bz, horizon, self.object_token_num, 5])
        output['masks'] = masks.reshape([bz, horizon, *masks.shape[2:]])

        return output

    def get_obj_slots(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        texts: Optional[str] = None,
        past_tokens: Optional[torch.FloatTensor] = None,
    ):
        """ This function receives pixel values in a horizon to output the corresponding visual representations.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        output = {'patch_features': [], 'visual_tokens': [], 'interactable_features': [], 'attentions': [], 'bboxes': [], 'masks': []}

        device = pixel_values.device
        bz, horizon = pixel_values.shape[:2]
        past_tokens = past_tokens
        general_texts, texts, texts_attn = self.object_centric_text_encoder(texts, temporal_length=1, device=device)
        for h in range(horizon):
            suboutput = self.get_obj_slots_per_image(pixel_values[:,h], past_tokens)
            interactability = self.object_centric_afd_interact_head(suboutput['visual_tokens'].squeeze(1), 
                                                                    texts.squeeze(1)) #, attention=texts_attn)
            suboutput['interactable_features'] = torch.unsqueeze(interactability, dim=1)
            if 'visual_tokens' in suboutput:
                past_tokens = suboutput['visual_tokens'].detach()
            for key, value in suboutput.items():
                if key in output:
                    output[key].append(value)

        for key, value in output.items():
            output[key] = torch.cat(value, axis=1) # concatenate by horizon

        slots = output['visual_tokens'] # [B, T, N, D]
        patch_features = output['patch_features'] # [B, T, VL, VD]
        # texts: [B, T, TL, TD]
        output['general_texts'] = general_texts
        output['texts'] = texts
        output['texts_attn'] = texts_attn
        return output

    def select_top_k_slots(
        self,
        slot_features,
        interactable_features,
        k=4
    ):
        """
        Selects the top-k interactable slots from slot_features based on interactable_features.
        
        Args:
            slot_features: Tensor of shape [bz, T, slot_num, dim]
            interactable_features: Binary Tensor of shape [bz, T, slot_num] indicating interactability
            k: Number of top slots to select

        Returns:
            selected_slots: Tensor of shape [bz, k, dim] containing the top-k interactable slots.
        """
        bz, T, slot_num, dim = slot_features.shape

        # Ensure interactable_features is float for sorting
        interactable_features = interactable_features.float()

        # Get top-k indices along slot_num axis (dim=2), keeping time dimension
        top_k_indices = torch.topk(
            interactable_features,
            k=min(k, slot_num),
            dim=2,   # <-- slot_num axis
            largest=True
        ).indices  # [bz, T, k]

        # Gather slot features corresponding to top-k indices
        selected_slots = torch.gather(
            slot_features,
            2,  # gather along slot_num axis
            top_k_indices.expand(-1, -1, -1, dim)  # [bz, T, k, dim]
        )

        return selected_slots, top_k_indices



    def decode_continuous_actions(
        self,
        patch_features: Optional[torch.FloatTensor] = None,
        slotted_features: Optional[torch.FloatTensor] = None,
        clip_embeddings:  Optional[torch.FloatTensor] = None,
        clip_attention_mask: Optional[torch.Tensor] = None,
        llama_input_ids: Optional[torch.LongTensor] = None,
        llama_attention_mask: Optional[torch.Tensor] = None,
        llama_labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        
        # Projection Logic =>> Update Attention Mask
        bz, t, n, d = slotted_features.shape
        if 'interactable_features' in kwargs:
            if 'top_k' in kwargs:
                top_k = kwargs['top_k']
            else:
                top_k = 4
            slotted_features, _ = self.select_top_k_slots(
                slotted_features, kwargs['interactable_features'], k=top_k)

        patch_features = rearrange(patch_features, 'b t v d -> (b t) v d')
        slotted_features = rearrange(slotted_features, 'b t n d -> (b t) n d')
        clip_embeddings = rearrange(clip_embeddings, 'b t n d -> (b t) n d')
        clip_attention_mask = rearrange(clip_attention_mask, 'b t n -> (b t) n')

        # Get Input Embeddings (from Language Model Embeddings)
        llama_input_embeddings = self.base_model.get_input_embeddings()(llama_input_ids)
        llama_input_embeddings = llama_input_embeddings.unsqueeze(1).repeat(1, t, 1, 1).flatten(0,1)
        llama_attention_mask = llama_attention_mask.unsqueeze(1).repeat(1, t, 1).flatten(0,1)
        llama_labels = llama_labels
        projected_clip_embeddings = self.object_centric_clip_text_projector(clip_embeddings)
        projected_clip_embeddings = projected_clip_embeddings.unsqueeze(1).repeat(1, t, 1, 1).flatten(0,1)
        clip_attention_mask = clip_attention_mask.unsqueeze(1).repeat(1, t, 1).flatten(0,1)

        # Get Slot encodings
        # projected_patch_embeddings = self.base_model.projector(patch_features)
        slotted_features = self.object_centric_slot_projector(slotted_features)
        projected_patch_embeddings = self.base_model.projector(patch_features)

        projected_patch_attention_mask = None
        if llama_attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], slotted_features.shape[1]),
                fill_value=True,
                dtype=llama_attention_mask.dtype,
                device=llama_attention_mask.device,
            )

        # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [llama_input_embeddings[:, :1, :], slotted_features, projected_clip_embeddings, llama_input_embeddings[:, 1:, :]], dim=1
        )
        multimodal_attention_mask = None
        if llama_attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [llama_attention_mask[:, :1], projected_patch_attention_mask, clip_attention_mask, llama_attention_mask[:, 1:]], dim=1
            )

        # Dispatch to Language Model
        language_model_output = self.base_model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[:,-1-(NUM_ACTIONS_CHUNK*ACTION_DIM):-1,:]  # (B, act_chunk_len, D)

        # L1 regression prediction
        continuous_actions_pred = self.object_centric_action_head.predict_action(actions_hidden_states)
        return continuous_actions_pred

class EmbodiedObject_LangSlotTemporal(nn.Module):
    config_class: PretrainedConfig = ObjectCentricVLAConfig

    def __init__(self, config: ObjectCentricVLAConfig = None, base_model: OpenVLAForActionPrediction = None,
                 use_slotgoals=False, number_of_slots=16) -> None:
        if config is None and base_model is None:
            assert(False)
        if isinstance(config, ObjectCentricVLAConfig):
            super().__init__(config)
        elif base_model is not None:
            super().__init__()
            self.base_model = base_model

        self.object_token_num = number_of_slots # 24
        self.use_slotgoals = use_slotgoals
        # The object-centric tokenizer handles extraction of object-centric features for,
        # --- capturing object 2D coordinates and object captions
        # --- capturing object features 
        # dim 256: 0.7546
        self.object_centric_tokenizer = SlotAttention(n_slots=self.object_token_num, in_dim=2176, feature_dim=512) 
        self.object_centric_bbox_head = MLP(input_dim=512, hidden_dim=1024, output_dim=5, num_layers=2) # switch depth to 3 for good results
        self.object_centric_mask_head = MaskPredictionHead(slot_dim=512, hidden_dim=1024, mask_size=(64,64))
        self.object_centric_slot_projector = PrismaticProjector(use_fused_vision_backbone=False, vision_dim=512, llm_dim=4096)

        # language tokenizer
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.object_centric_text_encoder = CLIPBasedTextEncoder(self.clip_model, self.clip_preprocess)

        # cross attention
        self.object_centric_afd_interact_head = AfdMultimodalTransformer(d_model=512, num_encoder_layers=3)

        self.object_centric_clip_text_projector = PrismaticProjector(use_fused_vision_backbone=False, vision_dim=512, llm_dim=4096)
        self.object_centric_action_head = L1RegressionActionHead(
            input_dim=4096,
            hidden_dim=4096,
            action_dim=7,
        )

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self.base_model._check_unnorm_key(self.norm_stats, unnorm_key)
        # print('unnorm_key', unnorm_key); 1/0
        return self.norm_stats[unnorm_key]["action"]

    def get_obj_slots_per_image(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        past_tokens:  Optional[torch.FloatTensor] = None
    ):
        """ This function receives pixel values then produces a set of object-centric slots 
                using the ObjectCentricTokenizer.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        num_dimensions = len(pixel_values.shape)
        if num_dimensions == 4: # [bz, cc, h, w]
            # Assuming Visual Feature Extraction on feature image of horizon 1, single view
            bz, horizon, num_view = pixel_values.shape[0], 1, 1
            patch_features = self.base_model.vision_backbone(pixel_values)

        else: # [bz, horizon, cc, h, w]
            1/0

        patch_features = patch_features.reshape(bz*horizon, -1, 2176)

        # Extract bboxes
        visual_tokens, attention, upsampled_visual_tokens = self.object_centric_tokenizer(patch_features, 
                                                                                          past_slots=past_tokens)
        if not self.training:
            return visual_tokens, None

        # print(patch_features.shape, visual_tokens.shape, torch.min(visual_tokens), torch.max(visual_tokens))
        bboxes = self.object_centric_bbox_head(visual_tokens).sigmoid()
        masks = self.object_centric_mask_head(visual_tokens)

        bboxes = torch.stack([bboxes], dim=1)
        masks = torch.stack([masks], dim=1)

        output = {}
        output['patch_features'] = patch_features.reshape([bz, horizon, -1, 2176])
        output['visual_tokens'] = visual_tokens.reshape([bz, horizon, self.object_token_num, -1])
        output['attentions'] = attention.reshape([bz, horizon, self.object_token_num, -1])
        output['bboxes'] = bboxes.reshape([bz, horizon, self.object_token_num, 5])
        output['masks'] = masks.reshape([bz, horizon, *masks.shape[2:]])

        return output

    def get_obj_slots(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        texts: Optional[str] = None,
        past_tokens: Optional[torch.FloatTensor] = None,
    ):
        """ This function receives pixel values in a horizon to output the corresponding visual representations.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        output = {'patch_features': [], 'visual_tokens': [], 'interactable_features': [], 'attentions': [], 'bboxes': [], 'masks': []}

        device = pixel_values.device
        bz, horizon = pixel_values.shape[:2]
        past_tokens = past_tokens
        general_texts, texts, texts_attn = self.object_centric_text_encoder(texts, temporal_length=1, device=device)
        for h in range(horizon):
            suboutput = self.get_obj_slots_per_image(pixel_values[:,h], past_tokens)
            interactability = self.object_centric_afd_interact_head(suboutput['visual_tokens'].squeeze(1), 
                                                                    texts.squeeze(1)) #, attention=texts_attn)
            suboutput['interactable_features'] = torch.unsqueeze(interactability, dim=1)
            if 'visual_tokens' in suboutput:
                past_tokens = suboutput['visual_tokens'].detach()
            for key, value in suboutput.items():
                if key in output:
                    output[key].append(value)

        for key, value in output.items():
            output[key] = torch.cat(value, axis=1) # concatenate by horizon

        slots = output['visual_tokens'] # [B, T, N, D]
        patch_features = output['patch_features'] # [B, T, VL, VD]
        # texts: [B, T, TL, TD]
        output['general_texts'] = general_texts
        output['texts'] = texts
        output['texts_attn'] = texts_attn
        return output

    def select_top_k_slots(
        self,
        slot_features,
        interactable_features,
        k=4
    ):
        """
        Selects the top-k interactable slots from slot_features based on interactable_features.
        
        Args:
            slot_features: Tensor of shape [bz, T, slot_num, dim]
            interactable_features: Binary Tensor of shape [bz, T, slot_num] indicating interactability
            k: Number of top slots to select

        Returns:
            selected_slots: Tensor of shape [bz, k, dim] containing the top-k interactable slots.
        """
        bz, T, slot_num, dim = slot_features.shape

        # Ensure interactable_features is float for sorting
        interactable_features = interactable_features.float()

        # Get top-k indices along slot_num axis (dim=2), keeping time dimension
        top_k_indices = torch.topk(
            interactable_features,
            k=min(k, slot_num),
            dim=2,   # <-- slot_num axis
            largest=True
        ).indices  # [bz, T, k]

        # Gather slot features corresponding to top-k indices
        selected_slots = torch.gather(
            slot_features,
            2,  # gather along slot_num axis
            top_k_indices.expand(-1, -1, -1, dim)  # [bz, T, k, dim]
        )

        return selected_slots, top_k_indices



    def decode_continuous_actions(
        self,
        patch_features: Optional[torch.FloatTensor] = None,
        slotted_features: Optional[torch.FloatTensor] = None,
        clip_embeddings:  Optional[torch.FloatTensor] = None,
        clip_attention_mask: Optional[torch.Tensor] = None,
        llama_input_ids: Optional[torch.LongTensor] = None,
        llama_attention_mask: Optional[torch.Tensor] = None,
        llama_labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        
        # Projection Logic =>> Update Attention Mask
        bz, t, n, d = slotted_features.shape
        if 'interactable_features' in kwargs:
            if 'top_k' in kwargs:
                top_k = kwargs['top_k']
            else:
                top_k = 4
            slotted_features, _ = self.select_top_k_slots(
                slotted_features, kwargs['interactable_features'], k=top_k)

        patch_features = rearrange(patch_features, 'b t v d -> (b t) v d')
        slotted_features = rearrange(slotted_features, 'b t n d -> (b t) n d')
        clip_embeddings = rearrange(clip_embeddings, 'b t n d -> (b t) n d')
        clip_attention_mask = rearrange(clip_attention_mask, 'b t n -> (b t) n')

        # Get Input Embeddings (from Language Model Embeddings)
        llama_input_embeddings = self.base_model.get_input_embeddings()(llama_input_ids)
        # llama_input_embeddings = llama_input_embeddings.unsqueeze(1).repeat(1, t, 1, 1).flatten(0,1)
        # llama_attention_mask = llama_attention_mask.unsqueeze(1).repeat(1, t, 1).flatten(0,1)
        llama_labels = llama_labels
        projected_clip_embeddings = self.object_centric_clip_text_projector(clip_embeddings)
        # projected_clip_embeddings = projected_clip_embeddings.unsqueeze(1).repeat(1, t, 1, 1).flatten(0,1)
        # clip_attention_mask = clip_attention_mask.unsqueeze(1).repeat(1, t, 1).flatten(0,1)

        # Get Slot encodings
        # projected_patch_embeddings = self.base_model.projector(patch_features)
        slotted_features = self.object_centric_slot_projector(slotted_features)
        projected_patch_embeddings = self.base_model.projector(patch_features)

        slotted_features = rearrange(slotted_features, '(b t) n d -> b (t n) d', b=bz, t=t)
        
        projected_patch_attention_mask = None
        if llama_attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (bz, slotted_features.shape[1]),
                fill_value=True,
                dtype=llama_attention_mask.dtype,
                device=llama_attention_mask.device,
            )

        # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [llama_input_embeddings[:, :1, :], slotted_features, projected_clip_embeddings, llama_input_embeddings[:, 1:, :]], dim=1
        )
        multimodal_attention_mask = None
        if llama_attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [llama_attention_mask[:, :1], projected_patch_attention_mask, clip_attention_mask, llama_attention_mask[:, 1:]], dim=1
            )

        # Dispatch to Language Model
        language_model_output = self.base_model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[:,-1-(NUM_ACTIONS_CHUNK*ACTION_DIM):-1,:]  # (B, act_chunk_len, D)

        # L1 regression prediction
        continuous_actions_pred = self.object_centric_action_head.predict_action(actions_hidden_states)
        return continuous_actions_pred


class EmbodiedRelationSlot(nn.Module):
    config_class: PretrainedConfig = ObjectCentricVLAConfig

    def __init__(self, config: ObjectCentricVLAConfig = None, base_model: OpenVLAForActionPrediction = None,
                 use_slotgoals=False, number_of_slots=16) -> None:
        if config is None and base_model is None:
            assert(False)
        if isinstance(config, ObjectCentricVLAConfig):
            super().__init__(config)
        elif base_model is not None:
            super().__init__()
            self.base_model = base_model

        self.object_token_num = number_of_slots # 24
        self.use_slotgoals = use_slotgoals
        # The object-centric tokenizer handles extraction of object-centric features for,
        # --- capturing object 2D coordinates and object captions
        # --- capturing object features 
        # dim 256: 0.7546
        self.object_centric_tokenizer = SlotAttention(n_slots=self.object_token_num, in_dim=2176, feature_dim=512) 
        self.object_centric_bbox_head = MLP(input_dim=512, hidden_dim=1024, output_dim=5, num_layers=2) # switch depth to 3 for good results
        self.object_centric_mask_head = MaskPredictionHead(slot_dim=512, hidden_dim=1024, mask_size=(64,64))
        self.object_centric_slot_projector = PrismaticProjector(use_fused_vision_backbone=False, vision_dim=512, llm_dim=4096)

        # language tokenizer
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.object_centric_text_encoder = CLIPBasedTextEncoder(self.clip_model, self.clip_preprocess)
        self.object_centric_relation_encoder = RelationTokensGrounding(dim=512, in_dim=4096, num_relation_tokens=self.object_token_num, n_heads=4, num_layers=3)

        self.object_centric_clip_text_projector = PrismaticProjector(use_fused_vision_backbone=False, vision_dim=512, llm_dim=4096)
        self.object_centric_action_head = L1RegressionActionHead(
            input_dim=4096,
            hidden_dim=4096,
            action_dim=7,
        )

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self.base_model._check_unnorm_key(self.norm_stats, unnorm_key)
        # print('unnorm_key', unnorm_key); 1/0
        return self.norm_stats[unnorm_key]["action"]

    def get_obj_slots_per_image(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        past_tokens:  Optional[torch.FloatTensor] = None
    ):
        """ This function receives pixel values then produces a set of object-centric slots 
                using the ObjectCentricTokenizer.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        num_dimensions = len(pixel_values.shape)
        if num_dimensions == 4: # [bz, cc, h, w]
            # Assuming Visual Feature Extraction on feature image of horizon 1, single view
            bz, horizon, num_view = pixel_values.shape[0], 1, 1
            patch_features = self.base_model.vision_backbone(pixel_values)

        else: # [bz, horizon, cc, h, w]
            1/0

        patch_features = patch_features.reshape(bz*horizon, -1, 2176)

        # Extract bboxes
        visual_tokens, attention, upsampled_visual_tokens = self.object_centric_tokenizer(patch_features, 
                                                                                          past_slots=past_tokens)
        if not self.training:
            return visual_tokens, None

        # print(patch_features.shape, visual_tokens.shape, torch.min(visual_tokens), torch.max(visual_tokens))
        bboxes = self.object_centric_bbox_head(visual_tokens).sigmoid()
        masks = self.object_centric_mask_head(visual_tokens)

        bboxes = torch.stack([bboxes], dim=1)
        masks = torch.stack([masks], dim=1)

        output = {}
        output['patch_features'] = patch_features.reshape([bz, horizon, -1, 2176])
        output['visual_tokens'] = visual_tokens.reshape([bz, horizon, self.object_token_num, -1])
        output['attentions'] = attention.reshape([bz, horizon, self.object_token_num, -1])
        output['bboxes'] = bboxes.reshape([bz, horizon, self.object_token_num, 5])
        output['masks'] = masks.reshape([bz, horizon, *masks.shape[2:]])

        return output

    def get_obj_slots(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        texts: Optional[str] = None,
        past_tokens: Optional[torch.FloatTensor] = None,
    ):
        """ This function receives pixel values in a horizon to output the corresponding visual representations.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        output = {'patch_features': [], 'visual_tokens': []}

        device = pixel_values.device
        bz, horizon = pixel_values.shape[:2]
        past_tokens = past_tokens
        general_texts, texts, texts_attn = self.object_centric_text_encoder(texts, temporal_length=1, device=device)
        for h in range(horizon):
            suboutput = self.get_obj_slots_per_image(pixel_values[:,h], past_tokens)
            if 'visual_tokens' in suboutput:
                past_tokens = suboutput['visual_tokens'].detach()
            for key, value in suboutput.items():
                if key in output:
                    output[key].append(value)

        for key, value in output.items():
            output[key] = torch.cat(value, axis=1) # concatenate by horizon

        slots = output['visual_tokens'] # [B, T, N, D]
        patch_features = output['patch_features'] # [B, T, VL, VD]
        # texts: [B, T, TL, TD]
        output['general_texts'] = general_texts
        output['texts'] = texts
        output['texts_attn'] = texts_attn
        return output

    def select_top_k_slots(
        self,
        slot_features,
        interactable_features,
        k=4
    ):
        """
        Selects the top-k interactable slots from slot_features based on interactable_features.
        
        Args:
            slot_features: Tensor of shape [bz, T, slot_num, dim]
            interactable_features: Binary Tensor of shape [bz, T, slot_num] indicating interactability
            k: Number of top slots to select

        Returns:
            selected_slots: Tensor of shape [bz, k, dim] containing the top-k interactable slots.
        """
        bz, T, slot_num, dim = slot_features.shape

        # Ensure interactable_features is float for sorting
        interactable_features = interactable_features.float()

        # Get top-k indices along slot_num axis (dim=2), keeping time dimension
        top_k_indices = torch.topk(
            interactable_features,
            k=min(k, slot_num),
            dim=2,   # <-- slot_num axis
            largest=True
        ).indices  # [bz, T, k]

        # Gather slot features corresponding to top-k indices
        selected_slots = torch.gather(
            slot_features,
            2,  # gather along slot_num axis
            top_k_indices.expand(-1, -1, -1, dim)  # [bz, T, k, dim]
        )

        return selected_slots, top_k_indices



    def decode_continuous_actions(
        self,
        patch_features: Optional[torch.FloatTensor] = None,
        slotted_features: Optional[torch.FloatTensor] = None,
        clip_embeddings:  Optional[torch.FloatTensor] = None,
        clip_attention_mask: Optional[torch.Tensor] = None,
        llama_input_ids: Optional[torch.LongTensor] = None,
        llama_attention_mask: Optional[torch.Tensor] = None,
        llama_labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        
        # Projection Logic =>> Update Attention Mask
        bz, t, n, d = slotted_features.shape

        patch_features = rearrange(patch_features, 'b t v d -> (b t) v d')
        slotted_features = rearrange(slotted_features, 'b t n d -> (b t) n d')
        clip_embeddings = rearrange(clip_embeddings, 'b t n d -> (b t) n d')
        clip_attention_mask = rearrange(clip_attention_mask, 'b t n -> (b t) n')

        # Get Input Embeddings (from Language Model Embeddings)
        llama_input_embeddings = self.base_model.get_input_embeddings()(llama_input_ids)
        llama_input_embeddings = llama_input_embeddings.unsqueeze(1).repeat(1, t, 1, 1).flatten(0,1)
        llama_attention_mask = llama_attention_mask.unsqueeze(1).repeat(1, t, 1).flatten(0,1)
        llama_labels = llama_labels
        projected_clip_embeddings = self.object_centric_clip_text_projector(clip_embeddings)
        projected_clip_embeddings = projected_clip_embeddings.unsqueeze(1).repeat(1, t, 1, 1).flatten(0,1)
        clip_attention_mask = clip_attention_mask.unsqueeze(1).repeat(1, t, 1).flatten(0,1)

        # Get Slot encodings
        # projected_patch_embeddings = self.base_model.projector(patch_features)
        slotted_features = self.object_centric_slot_projector(slotted_features)
        projected_patch_embeddings = self.base_model.projector(patch_features)
        projected_patch_embeddings = self.object_centric_relation_encoder(projected_patch_embeddings, slotted_features)

        projected_patch_attention_mask = None
        if llama_attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1] + slotted_features.shape[1]),
                fill_value=True,
                dtype=llama_attention_mask.dtype,
                device=llama_attention_mask.device,
            )

        # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [llama_input_embeddings[:, :1, :], projected_patch_embeddings, slotted_features, projected_clip_embeddings, llama_input_embeddings[:, 1:, :]], dim=1
        )
        multimodal_attention_mask = None
        if llama_attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [llama_attention_mask[:, :1], projected_patch_attention_mask, clip_attention_mask, llama_attention_mask[:, 1:]], dim=1
            )

        # Dispatch to Language Model
        language_model_output = self.base_model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[:,-1-(NUM_ACTIONS_CHUNK*ACTION_DIM):-1,:]  # (B, act_chunk_len, D)

        # L1 regression prediction
        continuous_actions_pred = self.object_centric_action_head.predict_action(actions_hidden_states)
        return continuous_actions_pred


class EmbodiedRelation_LangSlot(nn.Module):
    config_class: PretrainedConfig = ObjectCentricVLAConfig

    def __init__(self, config: ObjectCentricVLAConfig = None, base_model: OpenVLAForActionPrediction = None,
                 use_slotgoals=False, number_of_slots=16) -> None:
        if config is None and base_model is None:
            assert(False)
        if isinstance(config, ObjectCentricVLAConfig):
            super().__init__(config)
        elif base_model is not None:
            super().__init__()
            self.base_model = base_model

        self.object_token_num = number_of_slots # 24
        self.use_slotgoals = use_slotgoals
        # The object-centric tokenizer handles extraction of object-centric features for,
        # --- capturing object 2D coordinates and object captions
        # --- capturing object features 
        # dim 256: 0.7546
        self.object_centric_tokenizer = SlotAttention(n_slots=self.object_token_num, in_dim=2176, feature_dim=512) 
        self.object_centric_bbox_head = MLP(input_dim=512, hidden_dim=1024, output_dim=5, num_layers=2) # switch depth to 3 for good results
        self.object_centric_mask_head = MaskPredictionHead(slot_dim=512, hidden_dim=1024, mask_size=(64,64))
        self.object_centric_slot_projector = PrismaticProjector(use_fused_vision_backbone=False, vision_dim=512, llm_dim=4096)

        # language tokenizer
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.object_centric_text_encoder = CLIPBasedTextEncoder(self.clip_model, self.clip_preprocess)
        self.object_centric_relation_encoder = RelationTokensGrounding(dim=512, in_dim=4096, num_relation_tokens=self.object_token_num, n_heads=4, num_layers=3)
        # cross attention
        self.object_centric_afd_interact_head = AfdMultimodalTransformer(d_model=512, num_encoder_layers=3)

        self.object_centric_clip_text_projector = PrismaticProjector(use_fused_vision_backbone=False, vision_dim=512, llm_dim=4096)
        self.object_centric_action_head = L1RegressionActionHead(
            input_dim=4096,
            hidden_dim=4096,
            action_dim=7,
        )

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self.base_model._check_unnorm_key(self.norm_stats, unnorm_key)
        # print('unnorm_key', unnorm_key); 1/0
        return self.norm_stats[unnorm_key]["action"]

    def get_obj_slots_per_image(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        past_tokens:  Optional[torch.FloatTensor] = None
    ):
        """ This function receives pixel values then produces a set of object-centric slots 
                using the ObjectCentricTokenizer.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        num_dimensions = len(pixel_values.shape)
        if num_dimensions == 4: # [bz, cc, h, w]
            # Assuming Visual Feature Extraction on feature image of horizon 1, single view
            bz, horizon, num_view = pixel_values.shape[0], 1, 1
            patch_features = self.base_model.vision_backbone(pixel_values)

        else: # [bz, horizon, cc, h, w]
            1/0

        patch_features = patch_features.reshape(bz*horizon, -1, 2176)

        # Extract bboxes
        visual_tokens, attention, upsampled_visual_tokens = self.object_centric_tokenizer(patch_features, 
                                                                                          past_slots=past_tokens)
        if not self.training:
            output = {}
            output['patch_features'] = patch_features.reshape([bz, horizon, -1, 2176])
            output['visual_tokens'] = visual_tokens.reshape([bz, horizon, self.object_token_num, -1])
            return output
        
        # print(patch_features.shape, visual_tokens.shape, torch.min(visual_tokens), torch.max(visual_tokens))
        bboxes = self.object_centric_bbox_head(visual_tokens).sigmoid()
        masks = self.object_centric_mask_head(visual_tokens)

        bboxes = torch.stack([bboxes], dim=1)
        masks = torch.stack([masks], dim=1)

        output = {}
        output['patch_features'] = patch_features.reshape([bz, horizon, -1, 2176])
        output['visual_tokens'] = visual_tokens.reshape([bz, horizon, self.object_token_num, -1])
        output['attentions'] = attention.reshape([bz, horizon, self.object_token_num, -1])
        output['bboxes'] = bboxes.reshape([bz, horizon, self.object_token_num, 5])
        output['masks'] = masks.reshape([bz, horizon, *masks.shape[2:]])
        
        return output

    def get_obj_slots(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        texts: Optional[Union[str, torch.FloatTensor]] = None,
        past_tokens: Optional[torch.FloatTensor] = None,
    ):
        """ This function receives pixel values in a horizon to output the corresponding visual representations.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        output = {'patch_features': [], 'visual_tokens': [], 'interactable_features': []}

        device = pixel_values.device
        bz, horizon = pixel_values.shape[:2]
        past_tokens = past_tokens
        if torch.is_tensor(texts):
            general_texts = None
            texts = texts
            texts_attn = None
        else:
            general_texts, texts, texts_attn = self.object_centric_text_encoder(texts, temporal_length=1, device=device)
        
        for h in range(horizon):
            suboutput = self.get_obj_slots_per_image(pixel_values[:,h], past_tokens)
            interactability = self.object_centric_afd_interact_head(suboutput['visual_tokens'].squeeze(1), 
                                                                    texts.squeeze(1)) #, attention=texts_attn)
            suboutput['interactable_features'] = torch.unsqueeze(interactability, dim=1)
            if 'visual_tokens' in suboutput:
                past_tokens = suboutput['visual_tokens'].detach()
            for key, value in suboutput.items():
                if key in output:
                    output[key].append(value)

        for key, value in output.items():
            output[key] = torch.cat(value, axis=1) # concatenate by horizon

        slots = output['visual_tokens'] # [B, T, N, D]
        patch_features = output['patch_features'] # [B, T, VL, VD]
        # texts: [B, T, TL, TD]
        output['general_texts'] = general_texts
        output['texts'] = texts
        output['texts_attn'] = texts_attn
        return output

    def select_top_k_slots(
        self,
        slot_features,
        interactable_features,
        k=4
    ):
        """
        Selects the top-k interactable slots from slot_features based on interactable_features.
        
        Args:
            slot_features: Tensor of shape [bz, T, slot_num, dim]
            interactable_features: Binary Tensor of shape [bz, T, slot_num] indicating interactability
            k: Number of top slots to select

        Returns:
            selected_slots: Tensor of shape [bz, k, dim] containing the top-k interactable slots.
        """
        bz, T, slot_num, dim = slot_features.shape

        # Ensure interactable_features is float for sorting
        interactable_features = interactable_features.float()

        # Get top-k indices along slot_num axis (dim=2), keeping time dimension
        top_k_indices = torch.topk(
            interactable_features,
            k=min(k, slot_num),
            dim=2,   # <-- slot_num axis
            largest=True
        ).indices  # [bz, T, k]

        # Gather slot features corresponding to top-k indices
        selected_slots = torch.gather(
            slot_features,
            2,  # gather along slot_num axis
            top_k_indices.expand(-1, -1, -1, dim)  # [bz, T, k, dim]
        )

        return selected_slots, top_k_indices



    def decode_continuous_actions(
        self,
        patch_features: Optional[torch.FloatTensor] = None,
        slotted_features: Optional[torch.FloatTensor] = None,
        clip_embeddings:  Optional[torch.FloatTensor] = None,
        clip_attention_mask: Optional[torch.Tensor] = None,
        llama_input_ids: Optional[torch.LongTensor] = None,
        llama_attention_mask: Optional[torch.Tensor] = None,
        llama_labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        
        # Projection Logic =>> Update Attention Mask
        bz, t, n, d = slotted_features.shape
        if 'interactable_features' in kwargs:
            if 'top_k' in kwargs:
                top_k = kwargs['top_k']
            else:
                top_k = 4
            slotted_features, _ = self.select_top_k_slots(
                slotted_features, kwargs['interactable_features'], k=top_k)

        patch_features = rearrange(patch_features, 'b t v d -> (b t) v d')
        slotted_features = rearrange(slotted_features, 'b t n d -> (b t) n d')
        clip_embeddings = rearrange(clip_embeddings, 'b t n d -> (b t) n d')
        clip_attention_mask = rearrange(clip_attention_mask, 'b t n -> (b t) n')

        # Get Input Embeddings (from Language Model Embeddings)
        llama_input_embeddings = self.base_model.get_input_embeddings()(llama_input_ids)
        llama_input_embeddings = llama_input_embeddings.unsqueeze(1).repeat(1, t, 1, 1).flatten(0,1)
        llama_attention_mask = llama_attention_mask.unsqueeze(1).repeat(1, t, 1).flatten(0,1)
        llama_labels = llama_labels
        projected_clip_embeddings = self.object_centric_clip_text_projector(clip_embeddings)
        projected_clip_embeddings = projected_clip_embeddings.unsqueeze(1).repeat(1, t, 1, 1).flatten(0,1)
        clip_attention_mask = clip_attention_mask.unsqueeze(1).repeat(1, t, 1).flatten(0,1)

        # Get Slot encodings
        # projected_patch_embeddings = self.base_model.projector(patch_features)
        slotted_features = self.object_centric_slot_projector(slotted_features)
        projected_patch_embeddings = self.base_model.projector(patch_features)
        projected_patch_embeddings = self.object_centric_relation_encoder(projected_patch_embeddings, slotted_features)

        projected_patch_attention_mask = None
        if llama_attention_mask is not None:
            # projected_patch_attention_mask = torch.full(
            #     (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1] + slotted_features.shape[1]),
            #     fill_value=True,
            #     dtype=llama_attention_mask.dtype,
            #     device=llama_attention_mask.device,
            # )
            projected_patch_attention_mask = torch.ones(
                (projected_patch_embeddings.shape[0],
                projected_patch_embeddings.shape[1] + slotted_features.shape[1]),
                dtype=llama_attention_mask.dtype,
                device=llama_attention_mask.device,
            )

        # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [llama_input_embeddings[:, :1, :], projected_patch_embeddings, slotted_features, projected_clip_embeddings, llama_input_embeddings[:, 1:, :]], dim=1
        )
        multimodal_attention_mask = None
        if llama_attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [llama_attention_mask[:, :1], projected_patch_attention_mask, clip_attention_mask, llama_attention_mask[:, 1:]], dim=1
            )

        # Dispatch to Language Model
        language_model_output = self.base_model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[:,-1-(NUM_ACTIONS_CHUNK*ACTION_DIM):-1,:]  # (B, act_chunk_len, D)

        # L1 regression prediction
        continuous_actions_pred = self.object_centric_action_head.predict_action(actions_hidden_states)
        return continuous_actions_pred


class EmbodiedRelation_LangSlotTemporal(nn.Module):
    config_class: PretrainedConfig = ObjectCentricVLAConfig

    def __init__(self, config: ObjectCentricVLAConfig = None, base_model: OpenVLAForActionPrediction = None,
                 use_slotgoals=False, number_of_slots=16) -> None:
        if config is None and base_model is None:
            assert(False)
        if isinstance(config, ObjectCentricVLAConfig):
            super().__init__(config)
        elif base_model is not None:
            super().__init__()
            self.base_model = base_model

        self.object_token_num = number_of_slots # 24
        self.use_slotgoals = use_slotgoals
        # The object-centric tokenizer handles extraction of object-centric features for,
        # --- capturing object 2D coordinates and object captions
        # --- capturing object features 
        # dim 256: 0.7546
        self.object_centric_tokenizer = SlotAttention(n_slots=self.object_token_num, in_dim=2176, feature_dim=512) 
        self.object_centric_bbox_head = MLP(input_dim=512, hidden_dim=1024, output_dim=5, num_layers=2) # switch depth to 3 for good results
        self.object_centric_mask_head = MaskPredictionHead(slot_dim=512, hidden_dim=1024, mask_size=(64,64))
        self.object_centric_slot_projector = PrismaticProjector(use_fused_vision_backbone=False, vision_dim=512, llm_dim=4096)

        # language tokenizer
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.object_centric_text_encoder = CLIPBasedTextEncoder(self.clip_model, self.clip_preprocess)
        self.object_centric_relation_encoder = RelationTokensGrounding(dim=512, in_dim=4096, num_relation_tokens=self.object_token_num, n_heads=4, num_layers=3)
        # cross attention
        self.object_centric_afd_interact_head = AfdMultimodalTransformer(d_model=512, num_encoder_layers=3)

        self.object_centric_clip_text_projector = PrismaticProjector(use_fused_vision_backbone=False, vision_dim=512, llm_dim=4096)
        self.object_centric_action_head = L1RegressionActionHead(
            input_dim=4096,
            hidden_dim=4096,
            action_dim=7,
        )

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self.base_model._check_unnorm_key(self.norm_stats, unnorm_key)
        # print('unnorm_key', unnorm_key); 1/0
        return self.norm_stats[unnorm_key]["action"]

    def get_obj_slots_per_image(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        past_tokens:  Optional[torch.FloatTensor] = None
    ):
        """ This function receives pixel values then produces a set of object-centric slots 
                using the ObjectCentricTokenizer.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        num_dimensions = len(pixel_values.shape)
        if num_dimensions == 4: # [bz, cc, h, w]
            # Assuming Visual Feature Extraction on feature image of horizon 1, single view
            bz, horizon, num_view = pixel_values.shape[0], 1, 1
            patch_features = self.base_model.vision_backbone(pixel_values)

        else: # [bz, horizon, cc, h, w]
            1/0

        patch_features = patch_features.reshape(bz*horizon, -1, 2176)

        # Extract bboxes
        visual_tokens, attention, upsampled_visual_tokens = self.object_centric_tokenizer(patch_features, 
                                                                                          past_slots=past_tokens)
        if not self.training:
            return visual_tokens, None

        # print(patch_features.shape, visual_tokens.shape, torch.min(visual_tokens), torch.max(visual_tokens))
        bboxes = self.object_centric_bbox_head(visual_tokens).sigmoid()
        masks = self.object_centric_mask_head(visual_tokens)

        bboxes = torch.stack([bboxes], dim=1)
        masks = torch.stack([masks], dim=1)

        output = {}
        output['patch_features'] = patch_features.reshape([bz, horizon, -1, 2176])
        output['visual_tokens'] = visual_tokens.reshape([bz, horizon, self.object_token_num, -1])
        output['attentions'] = attention.reshape([bz, horizon, self.object_token_num, -1])
        output['bboxes'] = bboxes.reshape([bz, horizon, self.object_token_num, 5])
        output['masks'] = masks.reshape([bz, horizon, *masks.shape[2:]])

        return output

    def get_obj_slots(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        texts: Optional[Union[str, torch.FloatTensor]] = None,
        past_tokens: Optional[torch.FloatTensor] = None,
    ):
        """ This function receives pixel values in a horizon to output the corresponding visual representations.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        output = {'patch_features': [], 'visual_tokens': [], 'interactable_features': []}

        device = pixel_values.device
        bz, horizon = pixel_values.shape[:2]
        past_tokens = past_tokens
        if torch.is_tensor(texts):
            general_texts = None
            texts = texts
            texts_attn = None
        else:
            general_texts, texts, texts_attn = self.object_centric_text_encoder(texts, temporal_length=1, device=device)
    
        for h in range(horizon):
            suboutput = self.get_obj_slots_per_image(pixel_values[:,h], past_tokens)
            interactability = self.object_centric_afd_interact_head(suboutput['visual_tokens'].squeeze(1), 
                                                                    texts.squeeze(1)) #, attention=texts_attn)
            suboutput['interactable_features'] = torch.unsqueeze(interactability, dim=1)
            if 'visual_tokens' in suboutput:
                past_tokens = suboutput['visual_tokens'].detach()
            for key, value in suboutput.items():
                if key in output:
                    output[key].append(value)

        for key, value in output.items():
            output[key] = torch.cat(value, axis=1) # concatenate by horizon

        slots = output['visual_tokens'] # [B, T, N, D]
        patch_features = output['patch_features'] # [B, T, VL, VD]
        # texts: [B, T, TL, TD]
        output['general_texts'] = general_texts
        output['texts'] = texts
        output['texts_attn'] = texts_attn
        return output

    def select_top_k_slots(
        self,
        slot_features,
        interactable_features,
        k=4
    ):
        """
        Selects the top-k interactable slots from slot_features based on interactable_features.
        
        Args:
            slot_features: Tensor of shape [bz, T, slot_num, dim]
            interactable_features: Binary Tensor of shape [bz, T, slot_num] indicating interactability
            k: Number of top slots to select

        Returns:
            selected_slots: Tensor of shape [bz, k, dim] containing the top-k interactable slots.
        """
        bz, T, slot_num, dim = slot_features.shape

        # Ensure interactable_features is float for sorting
        interactable_features = interactable_features.float()

        # Get top-k indices along slot_num axis (dim=2), keeping time dimension
        top_k_indices = torch.topk(
            interactable_features,
            k=min(k, slot_num),
            dim=2,   # <-- slot_num axis
            largest=True
        ).indices  # [bz, T, k]

        # Gather slot features corresponding to top-k indices
        selected_slots = torch.gather(
            slot_features,
            2,  # gather along slot_num axis
            top_k_indices.expand(-1, -1, -1, dim)  # [bz, T, k, dim]
        )

        return selected_slots, top_k_indices



    def decode_continuous_actions(
        self,
        patch_features: Optional[torch.FloatTensor] = None,
        slotted_features: Optional[torch.FloatTensor] = None,
        clip_embeddings:  Optional[torch.FloatTensor] = None,
        clip_attention_mask: Optional[torch.Tensor] = None,
        llama_input_ids: Optional[torch.LongTensor] = None,
        llama_attention_mask: Optional[torch.Tensor] = None,
        llama_labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        
        # Projection Logic =>> Update Attention Mask
        bz, t, n, d = slotted_features.shape
        if 'interactable_features' in kwargs:
            if 'top_k' in kwargs:
                top_k = kwargs['top_k']
            else:
                top_k = 4
            slotted_features, _ = self.select_top_k_slots(
                slotted_features, kwargs['interactable_features'], k=top_k)

        patch_features = rearrange(patch_features, 'b t v d -> (b t) v d')
        slotted_features = rearrange(slotted_features, 'b t n d -> (b t) n d')
        clip_embeddings = rearrange(clip_embeddings, 'b t n d -> (b t) n d')
        clip_attention_mask = rearrange(clip_attention_mask, 'b t n -> (b t) n')

        # Get Input Embeddings (from Language Model Embeddings)
        llama_input_embeddings = self.base_model.get_input_embeddings()(llama_input_ids)
        # llama_input_embeddings = llama_input_embeddings.unsqueeze(1).repeat(1, t, 1, 1).flatten(0,1)
        # llama_attention_mask = llama_attention_mask.unsqueeze(1).repeat(1, t, 1).flatten(0,1)
        llama_labels = llama_labels
        projected_clip_embeddings = self.object_centric_clip_text_projector(clip_embeddings)
        # projected_clip_embeddings = projected_clip_embeddings.unsqueeze(1).repeat(1, t, 1, 1).flatten(0,1)
        # clip_attention_mask = clip_attention_mask.unsqueeze(1).repeat(1, t, 1).flatten(0,1)

        # Get Slot encodings
        # projected_patch_embeddings = self.base_model.projector(patch_features)
        slotted_features = self.object_centric_slot_projector(slotted_features)
        projected_patch_embeddings = self.base_model.projector(patch_features)
        projected_patch_embeddings = self.object_centric_relation_encoder(projected_patch_embeddings, slotted_features)

        slotted_features = rearrange(slotted_features, '(b t) n d -> b (t n) d', b=bz, t=t)
        projected_patch_embeddings = rearrange(projected_patch_embeddings, '(b t) n d -> b (t n) d', b=bz, t=t)

        projected_patch_attention_mask = None
        if llama_attention_mask is not None:
            # projected_patch_attention_mask = torch.full(
            #     (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1] + slotted_features.shape[1]),
            #     fill_value=True,
            #     dtype=llama_attention_mask.dtype,
            #     device=llama_attention_mask.device,
            # )
            projected_patch_attention_mask = torch.ones(
                (projected_patch_embeddings.shape[0],
                projected_patch_embeddings.shape[1] + slotted_features.shape[1]),
                dtype=llama_attention_mask.dtype,
                device=llama_attention_mask.device,
            )

        # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
        multimodal_embeddings = torch.cat(
            [llama_input_embeddings[:, :1, :], projected_patch_embeddings, slotted_features, projected_clip_embeddings, llama_input_embeddings[:, 1:, :]], dim=1
        )
        multimodal_attention_mask = None
        if llama_attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [llama_attention_mask[:, :1], projected_patch_attention_mask, clip_attention_mask, llama_attention_mask[:, 1:]], dim=1
            )

        # Dispatch to Language Model
        language_model_output = self.base_model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[-1]  # (B, seq_len, D)
        actions_hidden_states = last_hidden_states[:,-1-(NUM_ACTIONS_CHUNK*ACTION_DIM):-1,:]  # (B, act_chunk_len, D)

        # L1 regression prediction
        continuous_actions_pred = self.object_centric_action_head.predict_action(actions_hidden_states)
        return continuous_actions_pred


#####################################################################################################
#####################################################################################################

class OpenVLAForActionPrediction_SlotSSM(nn.Module):
    config_class: PretrainedConfig = ObjectCentricVLAConfig

    def __init__(self, config: ObjectCentricVLAConfig = None, base_model: OpenVLAForActionPrediction = None, 
                 number_of_slots = 16, backward_step = 24, forward_step = 0) -> None:
        if config is None and base_model is None:
            assert(False)
        if isinstance(config, ObjectCentricVLAConfig):
            super().__init__(config)
        elif base_model is not None:
            super().__init__()
            self.base_model = base_model

        self.object_token_num = number_of_slots

        # The object-centric tokenizer handles extraction of object-centric features for,
        # --- capturing object 2D coordinates and object captions
        # --- capturing object features 
        # dim 256: 0.7546
        self.object_centric_tokenizer = SlotAttention(n_slots=self.object_token_num, in_dim=2176, feature_dim=512) 
        self.object_centric_bbox_head = MLP(input_dim=512, hidden_dim=1024, output_dim=5, num_layers=2) # switch depth to 3 for good results
        self.object_centric_mask_head = MaskPredictionHead(slot_dim=512, hidden_dim=1024, mask_size=(64,64))
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.object_centric_text_encoder = CLIPBasedTextEncoder(self.clip_model, self.clip_preprocess)
        d_model = 512
        # self.object_centric_pre_ssm_fusion = SlotFusion(input_dims=[512,512], embed_dims=[512, 512], output_dim=512)
        self.object_centric_ssm = EmbodiedSlotSSM(
            num_slots=self.object_token_num, num_blocks=3, d_model=d_model,
            d_input=None, visual_d_model=2176, textual_d_model=None,
            space_attn_num_heads=d_model // 64, use_inverted_attention=False, 
            encoder_attn_num_heads=d_model // 64, lookback_only=True
        )
        # predicting the 16 steps backward and 8 steps head
        self.backward_steps = backward_step # 16
        self.forward_steps = forward_step # 0
        self.object_centric_latent_decoder = nn.Linear(1024, 512*(self.backward_steps + self.forward_steps)) 

    def get_obj_slots_per_image(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        past_tokens:  Optional[torch.FloatTensor] = None
    ):
        """ This function receives pixel values then produces a set of object-centric slots 
                using the ObjectCentricTokenizer.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        num_dimensions = len(pixel_values.shape)
        if num_dimensions == 4: # [bz, cc, h, w]
            # Assuming Visual Feature Extraction on feature image of horizon 1, single view
            bz, horizon, num_view = pixel_values.shape[0], 1, 1
            patch_features = self.base_model.vision_backbone(pixel_values)

        else: # [bz, horizon, cc, h, w]
            1/0

        patch_features = patch_features.reshape(bz*horizon, -1, 2176)

        # Extract bboxes
        visual_tokens, attention, upsampled_visual_tokens = self.object_centric_tokenizer(patch_features, 
                                                                                          past_slots=past_tokens)
        if not self.training:
            return visual_tokens, None

        # print(patch_features.shape, visual_tokens.shape, torch.min(visual_tokens), torch.max(visual_tokens))
        bboxes = self.object_centric_bbox_head(visual_tokens).sigmoid()
        masks = self.object_centric_mask_head(visual_tokens)

        bboxes = torch.stack([bboxes], dim=1)
        masks = torch.stack([masks], dim=1)

        output = {}
        output['patch_features'] = patch_features.reshape([bz, horizon, -1, 2176])
        output['visual_tokens'] = visual_tokens.reshape([bz, horizon, self.object_token_num, -1])
        output['attentions'] = attention.reshape([bz, horizon, self.object_token_num, -1])
        output['bboxes'] = bboxes.reshape([bz, horizon, self.object_token_num, 5])
        output['masks'] = masks.reshape([bz, horizon, *masks.shape[2:]])

        return output

    def get_obj_slots(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        texts: Optional[list] = None,
    ):
        """ This function receives pixel values in a horizon to output the corresponding visual representations.

            It operates only on each image,
                if in training mode, outputs the slots and the object-centric predictions.
                else, outputs only the slots.
        """
        device = pixel_values.device
        output = {'patch_features': [], 'visual_tokens': [], 'attentions': [], 'bboxes': [], 'masks': []}
        bz, horizon = pixel_values.shape[:2]
        past_tokens=None
        for h in range(horizon):
            suboutput = self.get_obj_slots_per_image(pixel_values[:,h], past_tokens)
            if 'visual_tokens' in suboutput:
                past_tokens = suboutput['visual_tokens'].detach()
            for key, value in suboutput.items():
                output[key].append(value)

        for key, value in output.items():
            output[key] = torch.cat(value, axis=1) # concatenate by horizon

        slots = output['visual_tokens'] # [B, T, N, D]
        patch_features = output['patch_features'] # [B, T, VL, VD]
        # texts: [B, T, TL, TD]
        general_texts, texts, texts_attn = self.object_centric_text_encoder(texts, temporal_length=horizon, device=device)
        output['general_texts'] = general_texts
        output['texts'] = texts
        output['texts_attn'] = texts_attn

        return output

    def get_slot_dynamics(
        self,
        slots,
        subgoals,
        patch_features, 
        texts, 
        texts_attn,
        output,
        redundant_steps=0
    ):

        # slots: [B, T, N, D]
        if redundant_steps > 0:
            slots = slots[:,:-redundant_steps]
            subgoals = subgoals[:,:-redundant_steps]
            patch_features = patch_features[:,:-redundant_steps]
            texts       = texts[:,:-redundant_steps]
            texts_attn  = texts_attn[:,:-redundant_steps]

        # slots = self.object_centric_pre_ssm_fusion([slots, subgoals])
        latent_slots, attention, slot_lists = self.object_centric_ssm(slots, patch_features, texts, texts_attn, output_attentions=False)
        slots = self.object_centric_latent_decoder(torch.cat([latent_slots, slots], dim=-1))
        B, T, N, D_x_horizon_steps = slots.shape
        D = D_x_horizon_steps // (self.backward_steps + self.forward_steps)
        slots_reshaped = slots.reshape(B*T, N, self.backward_steps + self.forward_steps, D)
        nxt_bboxes = self.object_centric_bbox_head(slots_reshaped.detach()[:,:,-1,:]).sigmoid() # selecting only the last step for evaluation
        nxt_bboxes = nxt_bboxes.reshape(B, T, N, 5)
        nxt_masks = self.object_centric_mask_head(slots_reshaped.detach()[:,:,-1,:]) # selecting only the last step for evaluation
        nxt_masks = nxt_masks.reshape([B, T, N, *nxt_masks.shape[2:]])
        # print(slots.shape, torch.min(slots), torch.max(slots))

        output['latent_slots'] = latent_slots
        output['nxt_tokens'] = slots.reshape(B, T, N, self.backward_steps + self.forward_steps, D)
        output['nxt_bboxes'] = nxt_bboxes
        output['nxt_masks'] = nxt_masks

        return output

