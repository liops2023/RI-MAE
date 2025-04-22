import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import math
import copy
import torch.distributed as dist

from timm.models.layers import DropPath, trunc_normal_
from torchvision import transforms

# 예시: your custom imports
from datasets import data_transforms
from .partial_fc import PartialFC_V2
# ProxyAnchorLoss를 직접 import할 수도 있음
from utils.losses import ProxyAnchorLoss

# 다른 유틸
from utils.logger import print_log

##############################
# Example MLP, Attention, etc
##############################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


##############################
# Example RI-MAE Base
##############################

class RI_MAE_Base(nn.Module):
    """
    Rotation Invariant MAE의 base encoder, dual-branch에 사용될 수 있음
    (간소화된 예시)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth_encoder
        self.num_heads = config.transformer_config.num_heads
        # etc

        # 아래 간단히 encoder만 구성
        self.blocks = nn.ModuleList([
            Block(dim=self.embed_dim, num_heads=self.num_heads) for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

        # 예시 Learnable token
        self.mask_token = nn.Parameter(torch.zeros(1,1,self.embed_dim))

        # etc init
        trunc_normal_(self.mask_token, std=.02)

    def forward(self, x):
        """
        x shape = (B, G, D) = (batch, num_groups, embed_dim)
        """
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class EMA(nn.Module):
    """
    Exponential Moving Average teacher
    """
    def __init__(self, model, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.num_updates = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, student_model):
        self.num_updates += 1
        alpha = min(1 - 1/self.num_updates, self.decay)
        for ema_p, s_p in zip(self.model.parameters(), student_model.parameters()):
            ema_p.data.mul_(alpha).add_(s_p.data, alpha=1-alpha)

    def forward(self, x):
        return self.model(x)


class RI_MAE(nn.Module):
    """
    Dual-branch: student - teacher
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.student = RI_MAE_Base(config)
        self.teacher = EMA(self.student, decay=config.ema_decay)

    def forward(self, x, noaug=False):
        # student forward
        s_out = self.student(x)
        # teacher forward (no grad)
        with torch.no_grad():
            t_out = self.teacher(x)
        return s_out, t_out

    def ema_step(self):
        self.teacher.update(self.student)


##############################
# Fine-tune class
##############################

class RITransformer_Finetune(nn.Module):
    """
    분류, 세그멘테이션, Retrieval 모두 가능한 예시
    retrieval 시에는 partial-fc + proxy anchor를 사용 가능
    """
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.tasks = config.model.get('tasks', ['classification'])

        # --- Backbone (pretrained RI_MAE_Base) ---
        self.backbone = RI_MAE_Base(config)

        # 임베딩 차원
        trans_dim = config.transformer_config.trans_dim
        # backbone_output_dim 가정: pooling 등으로 2*trans_dim? or trans_dim?
        # 여기서는 그냥 trans_dim 사용
        self.backbone_output_dim = trans_dim

        # --- Task-Specific Heads ---
        self.heads = nn.ModuleDict()
        self.loss_funcs = {}
        self.loss_weights = config.model.get('loss_weights', {})

        # Classification
        if 'classification' in self.tasks:
            cls_dim = config.model.cls_dim
            self.heads['classification'] = nn.Sequential(
                nn.Linear(self.backbone_output_dim, cls_dim)
            )
            self.loss_funcs['classification'] = nn.CrossEntropyLoss()
            print_log(f'[Finetune] Classification head -> {cls_dim}', logger='RITransformer_Finetune')

        # Segmentation
        if 'segmentation' in self.tasks:
            seg_num_classes = config.model.get('seg_num_classes', 50)
            self.heads['segmentation'] = nn.Sequential(
                nn.Conv1d(self.backbone_output_dim, seg_num_classes, kernel_size=1)
            )
            self.loss_funcs['segmentation'] = nn.CrossEntropyLoss()
            print_log(f'[Finetune] Segmentation head -> {seg_num_classes}', logger='RITransformer_Finetune')

        # Retrieval
        if 'retrieval' in self.tasks:
            ret_config = config.model.get('retrieval_config', {})
            ret_embed_size = ret_config.get('embedding_size', 128)
            ret_num_classes = ret_config.get('num_classes', 1000)
            use_pfc = ret_config.get('use_partial_fc', False)
            margin = ret_config.get('loss_margin', 0.1)
            alpha = ret_config.get('loss_alpha', 32)

            # Projection head
            self.heads['retrieval_proj'] = nn.Linear(self.backbone_output_dim, ret_embed_size)

            if use_pfc:
                # Partial FC
                self.heads['retrieval_pfc'] = PartialFC_V2(
                    margin_softmax=None,  # 여기서는 None, => 별도 로스 함수에서 계산
                    embedding_size=ret_embed_size,
                    num_classes=ret_num_classes,
                    sample_rate=ret_config.get('partial_fc_sample_rate', 1.0),
                    fp16=False,
                )
                # Proxy Anchor
                self.loss_funcs['retrieval'] = ProxyAnchorLoss(
                    nb_classes=ret_num_classes, sz_embed=ret_embed_size, margin=margin, alpha=alpha
                )
                print_log('[Finetune] Retrieval with PartialFC + ProxyAnchorLoss', logger='RITransformer_Finetune')
            else:
                # 그냥 ProxyAnchorLoss
                proxy_anchor_loss_fn = ProxyAnchorLoss(
                    nb_classes=ret_num_classes, sz_embed=ret_embed_size, margin=margin, alpha=alpha
                )
                self.heads['retrieval_loss'] = proxy_anchor_loss_fn
                self.loss_funcs['retrieval'] = proxy_anchor_loss_fn
                print_log('[Finetune] Retrieval with normal ProxyAnchorLoss (no partial FC)', logger='RITransformer_Finetune')

        for t in self.tasks:
            if t not in self.loss_weights:
                self.loss_weights[t] = 1.0

    def load_model_from_ckpt(self, ckpt_path):
        if not ckpt_path or (not os.path.exists(ckpt_path)):
            print_log(f'[Finetune] No valid ckpt path: {ckpt_path}', logger='RITransformer_Finetune', level='warning')
            return
        print_log(f'[Finetune] Loading backbone from {ckpt_path}', logger='RITransformer_Finetune')
        # 예시: torch.load -> self.backbone.load_state_dict(...)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt
        # filter backbone keys
        backbone_keys = {}
        for k,v in sd.items():
            if k.startswith('backbone.'):
                backbone_keys[k.replace('backbone.', '')] = v
        msg = self.backbone.load_state_dict(backbone_keys, strict=False)
        print_log(f'[Finetune] load msg: {msg}', logger='RITransformer_Finetune')

    def forward(self, x, targets=None, return_features=False):
        """
        x: (B, N, 3) or pre-processed
        """
        # 1) Backbone forward
        #   (여기서는 x 모양이 곧 (B,G,dim)이거나, 전처리 등 수행했다고 가정)
        feats = self.backbone(x)  # (B, G, D)

        # 간단히 풀링해서 (B, D) 만들어주기 (classification, retrieval 등)
        # segmentation이라면 per-group/point feature가 필요
        B, G, D = feats.shape
        global_feat = feats.mean(dim=1)  # (B, D)

        if return_features and (targets is None):
            return global_feat

        outputs = {}
        if 'classification' in self.tasks:
            logits_cls = self.heads['classification'](global_feat)  # (B, cls_dim)
            outputs['classification'] = logits_cls

        if 'segmentation' in self.tasks:
            # (B, D, G) -> input of conv1d => (B, D, G)
            feats_trans = feats.permute(0,2,1)  # (B, D, G)
            seg_logits = self.heads['segmentation'](feats_trans)  # (B, seg_cls, G)
            outputs['segmentation'] = seg_logits

        if 'retrieval' in self.tasks:
            embed = self.heads['retrieval_proj'](global_feat)  # (B, ret_embed)
            outputs['retrieval'] = embed

        if targets is None:
            # inference
            return outputs

        # training -> compute losses
        total_loss = 0.0
        losses_dict = {}
        metrics_dict = {}

        # classification
        if 'classification' in self.tasks:
            cls_labels = targets.get('classification', None)
            if cls_labels is not None:
                cls_logits = outputs['classification']
                loss_cls = self.loss_funcs['classification'](cls_logits, cls_labels)
                total_loss += self.loss_weights['classification'] * loss_cls
                losses_dict['classification'] = loss_cls.detach()
                acc = (cls_logits.argmax(dim=-1) == cls_labels).float().mean()
                metrics_dict['classification_acc'] = acc.detach()
        # segmentation
        if 'segmentation' in self.tasks:
            seg_labels = targets.get('segmentation', None)
            if seg_labels is not None:
                seg_logits = outputs['segmentation']  # (B, seg_cls, G)
                # seg_labels: (B, G)
                if seg_labels.dim() == 2 and seg_labels.size(1) == seg_logits.size(2):
                    loss_seg = self.loss_funcs['segmentation'](seg_logits, seg_labels)
                    total_loss += self.loss_weights['segmentation'] * loss_seg
                    losses_dict['segmentation'] = loss_seg.detach()
                    # miou, etc -> skip
        # retrieval
        if 'retrieval' in self.tasks:
            ret_labels = targets.get('retrieval', None)
            if ret_labels is not None:
                use_pfc = ('retrieval_pfc' in self.heads)
                if use_pfc:
                    # partial fc forward
                    embeddings = outputs['retrieval']
                    loss_ret = self.heads['retrieval_pfc'](embeddings, ret_labels)
                    # 만약 margin_softmax=ProxyAnchorLoss면?
                    # => partial_fc 내부에서 self.loss_funcs['retrieval'] 호출
                    # => 예시에서는 partial_fc의 forward에서 self.margin_softmax(...) 호출
                    # => 잘 맞춰서 동작
                    total_loss += self.loss_weights['retrieval'] * loss_ret
                    losses_dict['retrieval'] = loss_ret.detach()
                else:
                    # 단순 ProxyAnchorLoss
                    embeddings = outputs['retrieval']
                    loss_ret = self.loss_funcs['retrieval'](embeddings, ret_labels)
                    total_loss += self.loss_weights['retrieval'] * loss_ret
                    losses_dict['retrieval'] = loss_ret.detach()

        return total_loss, losses_dict, metrics_dict
