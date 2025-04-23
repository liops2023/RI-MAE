from cgi import test
import imp
from pkgutil import extend_path
import torch
import torch.nn as nn
import os
import json
import random
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import math
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils

# PyTorch3D 관련 모듈 추가
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        PerspectiveCameras,
        RasterizationSettings,
        PointsRasterizationSettings,
        PointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
        NormWeightedCompositor,
        Raysampler,
        NDCGridRaysampler,
        MonteCarloRaysampler,
        ImplicitRenderer,
        RayTracing
    )
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: PyTorch3D not available. Mesh ray casting will not work.")

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

def mesh_to_pointcloud(meshes, batch_size, config, device):
    """
    Meshes를 ray casting을 통해 포인트 클라우드로 변환
    
    Args:
        meshes: PyTorch3D Meshes 객체
        batch_size: 배치 크기
        config: 설정 객체
        device: 연산 장치
        
    Returns:
        points: (B, N, 3) 포인트 클라우드
    """
    if not PYTORCH3D_AVAILABLE:
        raise ImportError("PyTorch3D is required for mesh ray casting.")
    
    # 설정에서 파라미터 가져오기 (ray_casting 설정 활용)
    ray_casting_config = config.get('ray_casting', {})
    
    # ray casting 활성화 여부 확인
    enabled = ray_casting_config.get('enabled', True)
    if not enabled:
        raise ValueError("Ray casting is disabled in the config.")
    
    # 설정에서 파라미터 가져오기
    min_fov = ray_casting_config.get('min_fov', 30)
    max_fov = ray_casting_config.get('max_fov', 60)
    min_resolution = ray_casting_config.get('min_resolution', 64)
    max_resolution = ray_casting_config.get('max_resolution', 128)
    min_depth = ray_casting_config.get('min_depth', 0.1)
    max_depth = ray_casting_config.get('max_depth', 10.0)
    n_points = config.dataset.train.others.get('N_POINTS', config.npoints)
    
    # 각 배치마다 랜덤한 FOV와 해상도 설정
    fov = random.uniform(min_fov, max_fov)
    resolution = random.randint(min_resolution, max_resolution)
    
    # 카메라 설정
    cameras = PerspectiveCameras(
        fov=torch.tensor([fov], device=device).expand(batch_size),
        device=device
    )
    
    # Ray sampler 설정
    raysampler = NDCGridRaysampler(
        image_width=resolution,
        image_height=resolution,
        n_pts_per_ray=1,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    
    # Ray casting 수행
    rays = raysampler(cameras=cameras)
    ray_bundle = rays._replace(
        origins=rays.origins.to(device),
        directions=rays.directions.to(device),
    )
    
    # Ray tracing을 통해 표면 점 찾기
    ray_tracer = RayTracing()
    intersections, _, _ = ray_tracer(meshes, ray_bundle)
    
    # intersections에서 포인트 추출
    # 형태는 (batch_size, image_height, image_width, n_pts_per_ray, 3)
    points = intersections.reshape(batch_size, -1, 3)  # (B, H*W*n_pts, 3)
    
    # 필요한 포인트 수에 맞게 조정 (FPS 사용)
    if points.shape[1] > n_points:
        points = misc.fps(points, n_points)
    elif points.shape[1] < n_points:
        # 포인트가 부족한 경우, 중복하여 채우기
        idx = torch.randint(points.shape[1], (batch_size, n_points), device=device)
        points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        
    return points

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = SVC(C = 0.018, kernel ='linear')
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader) = builder.dataset_builder(args, config.dataset.train)
    (_, test_dataloader) =  builder.dataset_builder(args, config.dataset.val)
    (_, test_rot_dataloader) =  builder.dataset_builder(args, config.dataset.val_rot)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)
    best_rot_metrics = Acc_Metric(0.)
    rot_metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model = nn.DataParallel(base_model).to(device)
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    mse = nn.MSELoss()

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])
        mse_losses = AverageMeter(['mse_Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            
            # 데이터 타입 체크 및 포인트 클라우드 변환
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if dataset_name == 'ShapeNet':
                if PYTORCH3D_AVAILABLE and isinstance(data, dict) and 'mesh' in data and isinstance(data['mesh'], Meshes):
                    # PyTorch3D Mesh 데이터인 경우
                    print_log(f"Processing PyTorch3D Mesh data with ray casting...", logger=logger)
                    meshes = data['mesh'].to(device)
                    batch_size = meshes.batch_size
                    points = mesh_to_pointcloud(meshes, batch_size, config, device)
                else:
                    # 일반 ShapeNet 포인트 클라우드 데이터
                    points = data.to(device)
            elif dataset_name == 'ModelNet':
                if PYTORCH3D_AVAILABLE and isinstance(data, tuple) and len(data) > 2 and isinstance(data[2], dict) and 'mesh' in data[2] and isinstance(data[2]['mesh'], Meshes):
                    # PyTorch3D Mesh 데이터인 경우
                    print_log(f"Processing PyTorch3D Mesh data with ray casting...", logger=logger)
                    meshes = data[2]['mesh'].to(device)
                    batch_size = meshes.batch_size
                    points = mesh_to_pointcloud(meshes, batch_size, config, device)
                else:
                    # 일반 ModelNet 포인트 클라우드 데이터
                    points = data[0].to(device)
                    points = misc.fps(points, npoints)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            points = train_transforms(points)   # data aug by scale&move
            
            targets, output = base_model(points, cutmix = False)
            
            if config.model.use_cutmix:
                targets, output = base_model(points, cutmix = True)

            mse_loss = mse(targets, output)
            loss = mse_loss 

            loss.backward()
            base_model.module.ema_step()
            if config.clip_gradients:
                norm = builder.clip_gradients(base_model, config.clip_grad)

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
            

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()])
                mse_losses.update([mse_loss.item()])
            else:
                losses.update([loss.item()])
                mse_losses.update([mse_loss.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/CD_Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s mse_loss = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], ['%.4f' % l for l in mse_losses.val()],
                            optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/CD_Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:# and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger, rot=False)
            rot_metrics = validate(base_model, extra_train_dataloader, test_rot_dataloader, epoch, val_writer, args, config, logger=logger, rot=True)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
            if rot_metrics.better_than(best_rot_metrics):
                best_rot_metrics = rot_metrics
                builder.save_checkpoint(base_model, optimizer, epoch, rot_metrics, best_rot_metrics, 'ckpt-best-rot', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 5 or epoch%50==0:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None, rot=False):
    if rot:
        print_log(f"[VALIDATION_ROT] Start validating epoch {epoch}", logger = logger)
    else:
        print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            # 데이터 타입 체크 및 포인트 클라우드 변환
            if PYTORCH3D_AVAILABLE and isinstance(data, tuple) and len(data) > 2 and isinstance(data[2], dict) and 'mesh' in data[2] and isinstance(data[2]['mesh'], Meshes):
                # PyTorch3D Mesh 데이터인 경우
                print_log(f"Processing PyTorch3D Mesh data with ray casting in validation...", logger=logger)
                meshes = data[2]['mesh'].to(device)
                batch_size = meshes.batch_size
                points = mesh_to_pointcloud(meshes, batch_size, config, device)
                label = data[1].to(device)
            else:
                # 일반 포인트 클라우드 데이터
                points = data[0].to(device)
                label = data[1].to(device)
                points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach().cpu().numpy())
            train_label.append(target.detach().cpu().numpy())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # 데이터 타입 체크 및 포인트 클라우드 변환
            if PYTORCH3D_AVAILABLE and isinstance(data, tuple) and len(data) > 2 and isinstance(data[2], dict) and 'mesh' in data[2] and isinstance(data[2]['mesh'], Meshes):
                # PyTorch3D Mesh 데이터인 경우
                print_log(f"Processing PyTorch3D Mesh data with ray casting in test...", logger=logger)
                meshes = data[2]['mesh'].to(device)
                batch_size = meshes.batch_size
                points = mesh_to_pointcloud(meshes, batch_size, config, device)
                label = data[1].to(device)
            else:
                # 일반 포인트 클라우드 데이터
                points = data[0].to(device)
                label = data[1].to(device)
                points = misc.fps(points, npoints)
            
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach().cpu().numpy())
            test_label.append(target.detach().cpu().numpy())

        train_features = np.concatenate(train_features)
        train_label = np.concatenate(train_label)
        test_features = np.concatenate(test_features)
        test_label = np.concatenate(test_label)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features, train_label, test_features, test_label)

        if rot:
            print_log('[Validation_ROT] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)
        else: 
            print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()
    
    test(base_model, args, config, logger=logger)

def test(base_model, args, config, logger = None):
    # build dataset
    (_, test_dataloader) =  builder.dataset_builder(args, config.dataset.val)
    (_, test_rot_dataloader) =  builder.dataset_builder(args, config.dataset.val_rot)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)

    metrics = validate(base_model, extra_train_dataloader, test_dataloader, 0, None, args, config, logger=logger, rot=False)
    rot_metrics = validate(base_model, extra_train_dataloader, test_rot_dataloader, 0, None, args, config, logger=logger, rot=True)

def test_tsne(base_model, args, config, logger = None):
    # build dataset
    (_, test_dataloader) =  builder.dataset_builder(args, config.dataset.val)
    (_, test_rot_dataloader) =  builder.dataset_builder(args, config.dataset.val_rot)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            # 데이터 타입 체크 및 포인트 클라우드 변환
            if PYTORCH3D_AVAILABLE and isinstance(data, tuple) and len(data) > 2 and isinstance(data[2], dict) and 'mesh' in data[2] and isinstance(data[2]['mesh'], Meshes):
                # PyTorch3D Mesh 데이터인 경우
                print_log(f"Processing PyTorch3D Mesh data with ray casting in tsne train...", logger=logger)
                meshes = data[2]['mesh'].to(device)
                batch_size = meshes.batch_size
                points = mesh_to_pointcloud(meshes, batch_size, config, device)
                label = data[1].to(device)
            else:
                # 일반 포인트 클라우드 데이터
                points = data[0].to(device)
                label = data[1].to(device)
                points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach().cpu().numpy())
            train_label.append(target.detach().cpu().numpy())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # 데이터 타입 체크 및 포인트 클라우드 변환
            if PYTORCH3D_AVAILABLE and isinstance(data, tuple) and len(data) > 2 and isinstance(data[2], dict) and 'mesh' in data[2] and isinstance(data[2]['mesh'], Meshes):
                # PyTorch3D Mesh 데이터인 경우
                print_log(f"Processing PyTorch3D Mesh data with ray casting in tsne test...", logger=logger)
                meshes = data[2]['mesh'].to(device)
                batch_size = meshes.batch_size
                points = mesh_to_pointcloud(meshes, batch_size, config, device)
                label = data[1].to(device)
            else:
                # 일반 포인트 클라우드 데이터
                points = data[0].to(device)
                label = data[1].to(device)
                points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach().cpu().numpy())
            test_label.append(target.detach().cpu().numpy())

        train_features = np.concatenate(train_features)
        train_label = np.concatenate(train_label)
        test_features = np.concatenate(test_features)
        test_label = np.concatenate(test_label)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)
        
        return train_features, train_label, test_features, test_label

       