# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import II

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    EXTRACTOR_MODE_CHOICES,
    MASKING_DISTRIBUTION_CHOICES,
    LAYER_TYPE_CHOICES,
    ConvFeatureExtractionModel,
    TransformerEncoder,
)
from fairseq.modules import GradMultiply, LayerNorm
from fairseq.tasks.hubert_pretraining import (
    HubertPretrainingConfig,
    HubertPretrainingTask,
)
from fairseq import checkpoint_utils
import os
import asteroid
import json
import random
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass
class HubertConfig(FairseqDataclass):
    label_rate: float = II("task.label_rate")

    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group "
            "norm with d groups in the first conv block, whereas layer_norm "
            "has layer norms in every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )

    # dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a tarnsformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many "
            "dimensions. set to encoder_embed_dim is <= 0"
        },
    )
    untie_final_proj: bool = field(
        default=False,
        metadata={"help": "use separate projection for each target"},
    )
    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    conv_feature_layers: str = field(
        default="[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        metadata={
            "help": "string describing convolutional feature extraction "
            "layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65,
        metadata={"help": "probability of replacing a token with mask"},
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={"help": "legacy (to be removed)"},
    )

    # loss computation
    skip_masked: bool = field(
        default=False,
        metadata={"help": "skip computing losses over masked frames"},
    )
    skip_nomask: bool = field(
        default=False,
        metadata={"help": "skip computing losses over unmasked frames"},
    )

    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # FP16 optimization
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )

    # Conformer
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})
    init: str = field(
        default="",
        metadata={
            "help": "model initialized from path"
        },
        )
    source_enh: bool = field(
        default=False,
        metadata={"help": "apply enhancement models to the source or not"},
    )
    enh_models: Optional[str] = field(
        default=None,
        metadata={"help": "JSON file for enhancement models and their paths"},
    )
    debug_mode: int = field(
        default=1,
        metadata={"help": "Debug Mode"},
    )
    fusion_layer: Optional[str] = field(
        default=None, # 'mean', 'concat', 'add'
        metadata={"help": "Fusion layer for the enhanced features"},
    )



@register_model("hubert", dataclass=HubertConfig)
class HubertModel(BaseFairseqModel):
    def __init__(
        self,
        cfg: HubertConfig,
        task_cfg: HubertPretrainingConfig,
        dictionaries: List[Dictionary],
    ) -> None:
        super().__init__()
        logger.info(f"HubertModel Config: {cfg}")

        feature_enc_layers = eval(cfg.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = cfg.label_rate * feature_ds_rate / task_cfg.sample_rate

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.untie_final_proj = cfg.untie_final_proj
        if self.untie_final_proj:
            self.final_proj = nn.Linear(
                cfg.encoder_embed_dim, final_dim * len(dictionaries)
            )
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        # modules below are not needed during fine-tuning
        if any([d is None for d in dictionaries]):
            logger.info("cannot find dictionary. assume will be used for fine-tuning")
        else:
            self.num_classes = [len(d) for d in dictionaries]
            self.label_embs_concat = nn.Parameter(
                torch.FloatTensor(sum(self.num_classes), final_dim)
            )
            nn.init.uniform_(self.label_embs_concat)

        # apply enhancement models to the source or not.
        self.source_enh = cfg.source_enh
        self.enh_name2meta_info = {}
        if self.source_enh:
            with open(cfg.enh_models, "r") as f:
                obj = json.load(f)
            self.enh_name2meta_info.update(obj)
        
        # import pdb; pdb.set_trace()
        self.enh_name2model = {}

        for name, path in self.enh_name2meta_info.items():
            if name in ['ConvTasNet', 'DCCRNet', 'DCUNet', 'DPRNNTasNet', 'DPTNet']:
                model_cls = getattr(asteroid, name)
                model = model_cls.from_pretrained(path)
                for param in model.parameters():
                    param.requires_grad = False
                self.enh_name2model[name] = model.cuda()

        self.debug_mode = cfg.debug_mode
        logger.info(f"Debug Mode: {cfg.debug_mode}")

        self.fusion_type = cfg.fusion_layer
        if self.fusion_type == '': self.fusion_type = None
        if self.fusion_type is not None:
            self.encoder_embed_dim = cfg.encoder_embed_dim
            fusion_layers_num = 1 + cfg.encoder_layers
            self.fusion_layers = self.init_fusion_layers(fusion_layers_num, self.fusion_type)
    
    def init_fusion_layers(self, layers_num, type: str):
        if type == 'mean':
            return [lambda clean, noisy: (clean + noisy) / 2] * layers_num
        if type == 'add':
            return [lambda clean, noisy: clean + noisy] * layers_num
        elif type == 'concat':
            concat_fusion_layers = nn.ModuleList([nn.Linear(2 * self.embed, self.embed).cuda()] + 
                                                 [nn.Linear(2 * self.encoder_embed_dim, self.encoder_embed_dim) for _ in range(layers_num - 1)])
            self.concat_fusion_layers = concat_fusion_layers
            # first layer: B x C X T -> B x 2C x T -> B x T x 2C -> B x T x C -> B x C x T
            # other layers: T x B x C -> T x B x 2C -> T x B x C
            return [lambda clean, noisy: self.concat_fusion_layers[0](torch.cat([clean, noisy], dim=1).transpose(1,2)).transpose(1,2)] + \
                [lambda clean, noisy, i=i: self.concat_fusion_layers[i](torch.cat([clean, noisy], dim=-1)) for i in range(1, layers_num)]
        elif type == 'attention':
            self.attention_fusion_layers_enh = nn.ModuleList([nn.Sigmoid(nn.Linear(self.embed, 1).cuda()) + 
                                                              nn.Sigmoid(nn.Linear(self.encoder_embed_dim, 1).cuda()) for _ in range(layers_num - 1)])
            self.attention_fusion_layers_noisy = nn.ModuleList([nn.Sigmoid(nn.Linear(self.embed, 1).cuda()) + 
                                                              nn.Sigmoid(nn.Linear(self.encoder_embed_dim, 1).cuda()) for _ in range(layers_num - 1)])
            return [lambda clean, noisy: clean * self.attention_fusion_layers_enh[0](clean.transpose(1,2)).transpose(1,2) + \
                     noisy * self.attention_fusion_layers_noisy[0](noisy.transpose(1,2)).transpose(1,2)] + \
                     [lambda clean, noisy, i=i: clean * self.attention_fusion_layers_enh[i](clean) + \
                      noisy * self.attention_fusion_layers_noisy[i](noisy) for i in range(1, layers_num) ]
        elif type is None:
            return [lambda clean, noisy: clean] * layers_num
        else: raise NotImplementedError

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertConfig, task: HubertPretrainingTask):
        """Build a new model instance."""

        model = HubertModel(cfg, task.cfg, task.dictionaries)
        if cfg.init is not None:

            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.init, {})
            model_dict = model.state_dict()
            state_dict = {k:v for k,v in model_dict.items() if k in state["model"].keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict, strict=True)
            # model.load_state_dict(state["model"], strict=True)
            logger.info(f"model initialized from {cfg.init}")
        return model

    def apply_mask(self, x, padding_mask, target_list, x_noisy=None):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
            if x_noisy is not None: x_noisy[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0
            if x_noisy is not None: x_noisy[mask_channel_indices] = 0

        return x, x_noisy, mask_indices

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.logit_temp
        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list, feat_tsz

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def enh_partial(self, model, source_aug, source=None, max_enh_samples = 16000 * 6):
        # import pdb; pdb.set_trace()
        assert source_aug.dim() == 2
        max_duration = source_aug.size(1)

        if max_duration <= max_enh_samples:
            return model(source_aug).squeeze(1)
        else:
            # debug1: disable mixures of enhanced and noisy speech
            if self.debug_mode==1:
                return source_aug
            
            # debug2: replace enhanced segment with original speech
            elif self.debug_mode==2:
                # breakpoint()
                start = np.random.randint(0, max_duration - max_enh_samples)
                source_aug[:, start:start + max_enh_samples] = source[:, start:start + max_enh_samples]
                return source_aug

            # debug3: normalize volumn of enhanced segment with (max volumn of the corresponding original segment)
            elif self.debug_mode==3:
                start = np.random.randint(0, max_duration - max_enh_samples)
                original_segment = source[:, start:start + max_enh_samples]
                enhanced_segment = model(source_aug[:, start:start + max_enh_samples]).squeeze(1)
                original_max = torch.max(original_segment, axis=1, keepdim=True)[0]
                enhanced_max = torch.max(enhanced_segment, axis=1, keepdim=True)[0]
                norm = torch.maximum(original_max, enhanced_max)
                norm[norm < 1e-5] = 1
                enhanced_segment /= norm
                source_aug[:, start:start+max_enh_samples] = enhanced_segment
                return source_aug

    def norm_source_aug(self, origin_source, enhanced_source):
        original_max = torch.max(origin_source, axis=1, keepdim=True)[0]
        enhanced_max = torch.max(enhanced_source, axis=1, keepdim=True)[0]
        norm = torch.maximum(original_max, enhanced_max)
        norm[norm < 1e-5] = 1
        enhanced_source /= norm
        return enhanced_source


    # def fusion(self, clean:torch.Tensor, noisy:torch.Tensor, type=None):
    #     assert noisy.shape == clean.shape
    #     if type is None: type = self.fusion_type
    #     if type == 'mean':
    #         return (noisy + clean) / 2
    #     if type == 'add':
    #         return noisy + clean
    #     elif type == 'concat':
    #         return torch.cat([noisy, clean], dim=1)
    #     elif type is None or type == '':
    #         return clean
    #     else: raise NotImplementedError


    def forward_source_only(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:

        
        """output layer is 1-based"""
        features = self.forward_features(source)
        if target_list is not None:
            features, target_list, feat_tsz = self.forward_targets(features, target_list)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        # unmasked_features = features.clone() # What use?

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        # unmasked_features = self.dropout_features(unmasked_features)

        if mask:
            x, _, mask_indices = self.apply_mask(features, padding_mask, target_list)
        else:
            x = features
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x, None, fusion_layers=None,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def forward(
        self,
        source: torch.Tensor,
        source_aug: torch.Tensor = None,
        already_enhanced: bool = False,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:

        breakpoint()
        make_fuse = not (source_aug is None or self.fusion_type is None)
        if not make_fuse:
            if source_aug is not None:
                # already enhanced
                # source is the enhanced source and source_aug is the original noisy source
                source = self.norm_source_aug(source_aug, source)
            return self.forward_source_only(source, target_list, padding_mask, mask, features_only, output_layer)
        
        if source_aug is not None:
            source_noisy = source_aug.clone().detach()
            if self.source_enh and not already_enhanced:
                name = random.choice(list(self.enh_name2model.keys()))
                enh_model = self.enh_name2model[name]
                # logger.info(f"{name} is applied to the source.")
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        source_aug = self.enh_partial(enh_model, source_aug, source)
                source = source_aug
            else: # already enhanced
                # source is the enhanced source
                source = self.norm_source_aug(source_aug, source)
        else: 
            source_noisy = None

        # breakpoint()
        # sf.write("source0.wav",source[0].float().cpu(),16000)

        
        """output layer is 1-based"""
        features = self.forward_features(source)
        features_noisy = self.forward_features(source_noisy).detach()
        if target_list is not None:
            features, target_list, feat_tsz = self.forward_targets(features, target_list)
            features_noisy = features_noisy[..., :feat_tsz]

        features = self.fusion_layers[0](features, features_noisy)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        # unmasked_features = features.clone() # What use?

        features_noisy = features_noisy.transpose(1, 2)
        features_noisy = self.layer_norm(features_noisy)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
            features_noisy = self.post_extract_proj(features_noisy).detach()

        features = self.dropout_input(features)
        # unmasked_features = self.dropout_features(unmasked_features)

        features_noisy = self.dropout_input(features_noisy)

        if mask:
            x, x_noisy, mask_indices = self.apply_mask(features, padding_mask, target_list, features_noisy)
        else:
            x = features
            x_noisy = features_noisy
            mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        x, _ = self.encoder(
            x, x_noisy, fusion_layers=self.fusion_layers[1:],
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        def compute_pred(proj_x, target, label_embs):
            # compute logits for the i-th label set
            y = torch.index_select(label_embs, 0, target.long())
            negs = label_embs.unsqueeze(1).expand(-1, proj_x.size(0), -1)
            if self.target_glu:
                y = self.target_glu(y)
                negs = self.target_glu(negs)
            # proj_x: (S, D)
            # y: (S, D)
            # negs: (Neg, S, D)
            return self.compute_nce(proj_x, y, negs)

        label_embs_list = self.label_embs_concat.split(self.num_classes, 0)

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            if self.untie_final_proj:
                proj_x_m_list = proj_x_m.chunk(len(target_list), dim=-1)
            else:
                proj_x_m_list = [proj_x_m for _ in range(len(target_list))]
            logit_m_list = [
                compute_pred(proj_x_m, t[masked_indices], label_embs_list[i])
                for i, (proj_x_m, t) in enumerate(zip(proj_x_m_list, target_list))
            ]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            if self.untie_final_proj:
                proj_x_u_list = proj_x_u.chunk(len(target_list), dim=-1)
            else:
                proj_x_u_list = [proj_x_u for _ in range(len(target_list))]

            logit_u_list = [
                compute_pred(proj_x_u, t[nomask_indices], label_embs_list[i])
                for i, (proj_x_u, t) in enumerate(zip(proj_x_u_list, target_list))
            ]
        else:
            logit_u_list = [None for _ in target_list]

        result = {
            "logit_m_list": logit_m_list,
            "logit_u_list": logit_u_list,
            "padding_mask": padding_mask,
            "features_pen": features_pen,
        }
        return result

    def extract_features(
        self,
        source: torch.Tensor,
        source_aug: torch.Tensor = None,
        already_enhanced: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source, source_aug, already_enhanced=already_enhanced,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.target_glu = None
        self.final_proj = None
