import torch
import torch.nn as nn
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
    model.eval()
    # 예시
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for idx, data_batch in enumerate(val_loader):
            taxonomy_ids, model_ids, other_data = data_batch
            points = other_data['points'].cuda(args.local_rank, non_blocking=True)
            labels = other_data['label'].cuda(args.local_rank, non_blocking=True)

            # forward
            ret = model(points, targets=None)
            if isinstance(ret, dict) and 'classification' in ret:
                preds = ret['classification']
                pred_label = preds.argmax(dim=-1)
                all_preds.append(pred_label)
                all_labels.append(labels)
    # gather
    # calc acc
    # ...
    # return metrics
    return {}
