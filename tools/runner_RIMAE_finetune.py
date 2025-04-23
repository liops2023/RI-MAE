import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import time
import os

from tools import builder
from utils import misc, dist_utils
from utils.AverageMeter import AverageMeter
from utils.logger import print_log
from utils.metrics import calculate_recall_at_k  # 가정

# 예시 transform
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils

def run_net(args, config, train_writer=None, val_writer=None):
    logger = print_log
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), builder.dataset_builder(args, config.dataset.val)

    # 모델 빌드
    rank = dist_utils.get_rank() if args.distributed else 0
    local_rank = args.local_rank
    world_size = dist_utils.get_world_size() if args.distributed else 1

    base_model = builder.model_builder(config.model, rank=rank, local_rank=local_rank, world_size=world_size)

    # ckpt
    start_epoch = 0
    best_metrics = {}
    if args.resume:
        start_epoch, best_metrics_ckpt = builder.resume_model(base_model, args, logger=logger)
        if isinstance(best_metrics_ckpt, dict):
            best_metrics.update(best_metrics_ckpt)

    # GPU
    if args.use_gpu:
        base_model.cuda(local_rank)

    if args.distributed:
        if args.sync_bn:
            base_model = nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log("Using SyncBatchNorm")
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # optimizer
    param_groups = builder.get_param_groups(base_model, config, logger)
    optimizer, scheduler = builder.build_opti_sche(param_groups, config)

    # Start training loop
    for epoch in range(start_epoch, config.max_epoch+1):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        base_model.train()
        # train one epoch
        losses_meter = AverageMeter()
        data_meter = AverageMeter()
        batch_meter = AverageMeter()

        end_time = time.time()
        for step, data_batch in enumerate(train_dataloader):
            data_time = time.time() - end_time
            data_meter.update(data_time)

            # 예시
            taxonomy_ids, model_ids, other_data = data_batch
            points = other_data['points'].cuda(local_rank, non_blocking=True)
            labels = other_data['label'].cuda(local_rank, non_blocking=True)

            # downsample if needed
            # ...
            # transform
            points = data_transforms.PointcloudScaleAndTranslate()(points)

            # targets dict
            targets = {}
            if 'classification' in config.model.tasks:
                targets['classification'] = labels
            if 'retrieval' in config.model.tasks:
                targets['retrieval'] = labels
            # segmentation일 경우 별도 seg_label
            if 'segmentation_label' in other_data:
                targets['segmentation'] = other_data['segmentation_label'].cuda(local_rank, non_blocking=True)

            # forward
            ret = base_model(points, targets=targets)
            # ret = (total_loss, losses_dict, metrics_dict)
            if isinstance(ret, tuple) and len(ret) == 3:
                total_loss, losses_dict, metrics_dict = ret
            else:
                # 혹은 classification만 하는 경우
                total_loss = ret
                losses_dict = {}
                metrics_dict = {}

            optimizer.zero_grad()
            total_loss.backward()
            # grad clip
            if config.get('grad_norm_clip', 0.0) > 0.0:
                nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip)
            optimizer.step()

            losses_meter.update(total_loss.item())
            batch_time = time.time() - end_time
            batch_meter.update(batch_time)
            end_time = time.time()

            if step % 50 == 0:
                if args.local_rank == 0:
                    print_log(f"Epoch [{epoch}/{config.max_epoch}] Step [{step}/{len(train_dataloader)}] "
                              f"Loss: {losses_meter.avg:.4f}, data_time: {data_time:.3f}")

        # scheduler step
        if isinstance(scheduler, (list, tuple)):
            for s in scheduler:
                s.step(epoch)
        else:
            scheduler.step(epoch)

        # validation step
        if epoch % args.val_freq == 0 or epoch == config.max_epoch:
            val_metrics = validate(base_model, test_dataloader, epoch, args, config)
            # save checkpoint if best
            # ...
    return best_metrics


def validate(model, val_loader, epoch, args, config):
    """
    1) Classification: Accuracy
    2) Segmentation: mIoU
    3) Retrieval: Recall@1, Recall@5 등
    """
    model.eval()

    # 어떤 서브태스크가 있는지 확인
    tasks = config.model.get('tasks', [])

    # classification
    all_preds_cls = []
    all_labels_cls = []

    # segmentation
    all_preds_seg = []
    all_labels_seg = []

    # retrieval
    all_embeds = []
    all_retrieval_labels = []

    with torch.no_grad():
        for idx, data_batch in enumerate(val_loader):
            # 데이터 언패킹
            taxonomy_ids, model_ids, other_data = data_batch

            points = other_data['points'].cuda(args.local_rank, non_blocking=True)
            labels = other_data['label'].cuda(args.local_rank, non_blocking=True)

            # 혹시 segmentation 라벨이 있으면 로드
            seg_labels = None
            if 'segmentation_label' in other_data:
                seg_labels = other_data['segmentation_label'].cuda(args.local_rank, non_blocking=True)

            # forward
            out = model(points, targets=None)  # inference 모드

            # Classification
            if 'classification' in tasks and 'classification' in out:
                cls_logits = out['classification']  # (B, num_classes)
                preds = cls_logits.argmax(dim=-1)
                all_preds_cls.append(preds.cpu())
                all_labels_cls.append(labels.cpu())

            # Segmentation
            if 'segmentation' in tasks and 'segmentation' in out and seg_labels is not None:
                seg_logits = out['segmentation']  # (B, seg_cls, G)
                # argmax => (B, G)
                seg_preds = seg_logits.argmax(dim=1)  # channel dim=1
                all_preds_seg.append(seg_preds.cpu())
                all_labels_seg.append(seg_labels.cpu())

            # Retrieval
            if 'retrieval' in tasks and 'retrieval' in out:
                embeds = out['retrieval']  # (B, emb_dim)
                # 라벨: labels(=classification label 재활용) or 별도 retrieval label
                # 여기서는 labels를 재활용한다 가정
                all_embeds.append(embeds.cpu())
                all_retrieval_labels.append(labels.cpu())

    # --- gather all data if in distributed mode ---
    if args.distributed:
        # 예: classification
        all_preds_cls = torch.cat(all_preds_cls, dim=0) if len(all_preds_cls)>0 else None
        all_labels_cls = torch.cat(all_labels_cls, dim=0) if len(all_labels_cls)>0 else None
        # distributed.gather_tensor / all_gather_tensor 등으로 모으기
        # 여기서는 간단히 reduce_tensor 예시
        # 실제론 tensor shape가 크면 all_gather 해야 할 수도 있음
        if all_preds_cls is not None:
            all_preds_cls = gather_concat(all_preds_cls, args.local_rank)
        if all_labels_cls is not None:
            all_labels_cls = gather_concat(all_labels_cls, args.local_rank)
        
        # segmentation
        if len(all_preds_seg) > 0:
            all_preds_seg = torch.cat(all_preds_seg, dim=0)
            all_preds_seg = gather_concat(all_preds_seg, args.local_rank)
            all_labels_seg = torch.cat(all_labels_seg, dim=0)
            all_labels_seg = gather_concat(all_labels_seg, args.local_rank)
        
        # retrieval
        if len(all_embeds) > 0:
            all_embeds = torch.cat(all_embeds, dim=0)
            all_embeds = gather_concat(all_embeds, args.local_rank)
            all_retrieval_labels = torch.cat(all_retrieval_labels, dim=0)
            all_retrieval_labels = gather_concat(all_retrieval_labels, args.local_rank)

    else:
        # 단일 GPU면 그냥 cat 하면 됨
        if len(all_preds_cls) > 0:
            all_preds_cls = torch.cat(all_preds_cls, dim=0)
            all_labels_cls = torch.cat(all_labels_cls, dim=0)
        if len(all_preds_seg) > 0:
            all_preds_seg = torch.cat(all_preds_seg, dim=0)
            all_labels_seg = torch.cat(all_labels_seg, dim=0)
        if len(all_embeds) > 0:
            all_embeds = torch.cat(all_embeds, dim=0)
            all_retrieval_labels = torch.cat(all_retrieval_labels, dim=0)

    # --- 이제 metrics 계산 ---
    metrics = {}

    # 1) classification accuracy
    if 'classification' in tasks and isinstance(all_preds_cls, torch.Tensor):
        correct = (all_preds_cls == all_labels_cls).sum().item()
        total = all_labels_cls.numel()
        acc = correct / max(total,1)
        metrics['classification_acc'] = acc

    # 2) segmentation mIoU (간단 예시)
    if 'segmentation' in tasks and isinstance(all_preds_seg, torch.Tensor):
        metrics['seg_mIoU'] = compute_mIoU(all_preds_seg, all_labels_seg, n_classes=config.model.seg_num_classes)

    # 3) retrieval -> Recall@1, 5 등
    if 'retrieval' in tasks and isinstance(all_embeds, torch.Tensor):
        # 임베딩을 전부 모았으니, recall@1,5...를 측정
        # 여기서는 간단히 recall@1 예시
        recall1 = compute_recall_at_k(all_embeds, all_retrieval_labels, k=1)
        recall5 = compute_recall_at_k(all_embeds, all_retrieval_labels, k=5)
        metrics['retrieval_recall@1'] = recall1
        metrics['retrieval_recall@5'] = recall5

    return metrics


def gather_concat(tensor, local_rank):
    """
    분산 환경에서 모든 rank의 tensor를 concat하는 예시 함수
    shape: (N, ...) 이라고 가정
    """
    world_size = dist.get_world_size()
    all_sizes = [torch.zeros(1, dtype=torch.long, device=tensor.device) for _ in range(world_size)]
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=tensor.device)
    dist.all_gather(all_sizes, local_size)
    all_sizes = [sz.item() for sz in all_sizes]
    max_size = max(all_sizes)

    # pad
    pad_size = max_size - local_size
    if pad_size > 0:
        padding = [0,0]*(len(tensor.shape)-1) + [0,pad_size]
        tensor_padded = F.pad(tensor, padding, mode='constant', value=0)
    else:
        tensor_padded = tensor

    gather_list = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor_padded)

    # concat
    cat_tensors = []
    for i in range(world_size):
        cat_tensors.append(gather_list[i][:all_sizes[i]])
    out = torch.cat(cat_tensors, dim=0)
    return out

def compute_mIoU(preds, labels, n_classes):
    """
    간단한 mIoU 계산 예시
    preds, labels: shape (N, num_points)
    """
    preds = preds.view(-1)
    labels = labels.view(-1)
    valid_mask = (labels >= 0) & (labels < n_classes)
    preds = preds[valid_mask]
    labels = labels[valid_mask]
    total_iou = 0.0
    count = 0
    for c in range(n_classes):
        # IoU for class c
        pred_c = (preds == c)
        label_c = (labels == c)
        inter = (pred_c & label_c).sum().item()
        union = (pred_c | label_c).sum().item()
        if union > 0:
            iou_c = inter / union
            total_iou += iou_c
            count += 1
    if count == 0:
        return 0.0
    return total_iou / count

def compute_recall_at_k(embeds, labels, k=1):
    """
    retrieval recall@k
    간단히 brute force 예시
    """
    embeds_np = embeds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    # l2 normalize
    norm = np.linalg.norm(embeds_np, axis=1, keepdims=True)
    embeds_np = embeds_np / (norm + 1e-10)
    n = embeds_np.shape[0]

    # 전체 유사도 (n x n)
    sims = np.dot(embeds_np, embeds_np.T)  # (n,n)
    # 각 행을 similarity 내림차순 정렬 -> index
    # top k 중에 동일 label이 있으면 success
    correct = 0
    for i in range(n):
        # 자기 자신 제외
        # sims[i,i] = -inf?
        sims[i,i] = -999999.0
        ranking = np.argsort(-sims[i])  # desc
        topk = ranking[:k]
        if labels_np[i] in labels_np[topk]:
            correct += 1
    recall = correct / n
    return recall