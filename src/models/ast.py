"""Modern model: Audio Spectrogram Transformer (AST, Gong et al., Interspeech 2021).

* pretrained=True : ImageNet + AudioSet pretrained checkpoint
  (MIT/ast-finetuned-audioset-10-10-0.4593, HuggingFace) fine-tuned on ESC-50.
* pretrained=False: the SAME architecture randomly initialised -> trained from scratch.
  This pair is the pretrained-vs-scratch comparison required by the project.

Own code here:
  * positional-embedding resizing (1024 -> 512 frames) so 5-s clips don't waste
    half of the sequence on padding (bilinear interpolation over the patch grid,
    as in the original AST code for different input lengths);
  * SpecAugment on the fbank input (shared SpecAugment module).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SpecAugment

DEFAULT_CKPT = "MIT/ast-finetuned-audioset-10-10-0.4593"
POS_KEY = "audio_spectrogram_transformer.embeddings.position_embeddings"


def _grid(cfg):
    f = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
    t = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
    return f, t


def resize_pos_embed(pos: torch.Tensor, old_cfg, new_cfg, n_special: int = 2) -> torch.Tensor:
    """pos: (1, n_special + F*T_old, D) -> (1, n_special + F*T_new, D)."""
    f_old, t_old = _grid(old_cfg)
    f_new, t_new = _grid(new_cfg)
    if (f_old, t_old) == (f_new, t_new):
        return pos
    special, grid = pos[:, :n_special], pos[:, n_special:]
    D = grid.size(-1)
    grid = grid.reshape(1, f_old, t_old, D).permute(0, 3, 1, 2)          # (1, D, F, T)
    grid = F.interpolate(grid, size=(f_new, t_new), mode="bilinear", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, f_new * t_new, D)
    return torch.cat([special, grid], dim=1)


class AST(nn.Module):
    def __init__(self, n_classes: int = 50, pretrained: bool = True, checkpoint: str = DEFAULT_CKPT,
                 max_length: int = 512, hidden_dropout: float = 0.0, attention_dropout: float = 0.0,
                 spec_augment: dict | None = None, freeze_layers: int = 0, arch_config: dict | None = None):
        super().__init__()
        from transformers import ASTConfig, ASTForAudioClassification

        if pretrained:
            src = ASTForAudioClassification.from_pretrained(checkpoint)
            cfg = ASTConfig.from_dict(src.config.to_dict())
        elif arch_config:                          # rebuilding a trained model offline
            src, cfg = None, ASTConfig.from_dict(arch_config)
        else:
            src, cfg = None, ASTConfig()          # defaults == AST-base architecture
        old_cfg = ASTConfig.from_dict(cfg.to_dict())
        cfg.max_length = max_length
        cfg.num_labels = n_classes
        cfg.id2label = {i: str(i) for i in range(n_classes)}
        cfg.label2id = {str(i): i for i in range(n_classes)}
        cfg.hidden_dropout_prob = hidden_dropout
        cfg.attention_probs_dropout_prob = attention_dropout
        self.net = ASTForAudioClassification(cfg)
        self.arch_config = cfg.to_dict()           # saved in the checkpoint by train.py

        if src is not None:
            sd = {k: v for k, v in src.state_dict().items() if not k.startswith("classifier.dense")}
            sd[POS_KEY] = resize_pos_embed(sd[POS_KEY], old_cfg, cfg)
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            assert all(k.startswith("classifier.dense") for k in missing), missing
            del src

        if freeze_layers > 0:                      # optional: freeze patch embed + first N blocks
            enc = self.net.audio_spectrogram_transformer
            for p in enc.embeddings.parameters():
                p.requires_grad = False
            for blk in enc.encoder.layer[:freeze_layers]:
                for p in blk.parameters():
                    p.requires_grad = False

        sa = dict(spec_augment or {})
        self.spec_augment = SpecAugment(**{k: v for k, v in sa.items() if k != "enabled"}) \
            if sa and sa.get("enabled", True) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B, T, mel)
        if self.training and not isinstance(self.spec_augment, nn.Identity):
            x = self.spec_augment(x.transpose(1, 2).unsqueeze(1)).squeeze(1).transpose(1, 2)
        return self.net(input_values=x).logits
