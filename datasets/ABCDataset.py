import os
import glob
import random
from typing import Tuple, List

import torch
import numpy as np
import torch.utils.data as data
# import trimesh # Remove trimesh import

# Import PyTorch3D components
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
)

# (선택) 여기서는 rnd_rot, pc_normalize 등 공용 함수를 재사용할 수도 있음
# utils.logger, etc.
from utils.logger import print_log
from .build import DATASETS

# Remove pc_normalize and rnd_rot as they operate on point clouds
# def pc_normalize(pc):
#     centroid = np.mean(pc, axis=0)
#     pc = pc - centroid
#     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
#     pc = pc / (m + 1e-6)
#     return pc
# 
# def rnd_rot():
#     a = np.random.rand() * 2 * np.pi
#     z = np.random.rand() * 2 - 1
#     c = np.random.rand() * 2 * np.pi
#     # ZYZ-Euler
#     # -> 회전행렬 (직접 구현 or 다른 방식)
#     # 간단 예시 (동일한 방법)
#     return rotmat(a, np.arccos(z), c, False)
# 
# def rotmat(a, b, c, hom_coord=False):
#     def z(a):
#         return np.array([[np.cos(a), np.sin(a), 0, 0],
#                          [-np.sin(a), np.cos(a), 0, 0],
#                          [0, 0, 1, 0],
#                          [0, 0, 0, 1]])
#     def y(a):
#         return np.array([[np.cos(a), 0, -np.sin(a), 0],
#                          [0, 1, 0, 0],
#                          [np.sin(a), 0, np.cos(a), 0],
#                          [0, 0, 0, 1]])
#     R = z(a).dot(y(b)).dot(z(c))
#     if hom_coord:
#         return R
#     else:
#         return R[:3, :3]

@DATASETS.register_module()
class ABCDataset(data.Dataset):
    """
    ~/data/abc 경로 아래
      categoryA/
        sample01.obj
        sample02.obj
        ...
      categoryB/
        sample01.obj
        ...
    와 같은 구조.

    config 예시:
      NAME: ABCDataset
      ROOT_DIR: /home/xxx/data/abc
      # N_POINTS: 2048 # No longer needed for mesh loading
      subset: train  (또는 'val','test')
      # rot_aug: True # Removed, apply augmentation on mesh if needed
    """
    def __init__(self, config):
        super().__init__()
        # config에서 필요한 것들
        self.root_dir = config.ROOT_DIR  # "~/data/abc"
        # 점 추출 수 (point cloud 변환 후 샘플링 갯수)
        self.sample_points_num = getattr(config, 'N_POINTS', 2048)
        self.subset = getattr(config, "subset", "train")  # train/val/test

        # Ray‑casting 파라미터 범위 ------------------------------------------------
        # 수평/수직 FOV 동일하게 간주. (deg)
        self.fov_range: Tuple[float, float] = getattr(config, 'FOV_RANGE', (40.0, 70.0))
        # 렌더링 해상도 범위 (정사각형 이미지 size)
        self.res_range: Tuple[int, int] = getattr(config, 'RES_RANGE', (128, 256))
        # 카메라‑타겟 거리 (world units)
        self.dist_range: Tuple[float, float] = getattr(config, 'DIST_RANGE', (2.0, 4.0))
        # 카메라 높이/각도 지정용 elevation, azimuth 범위 (deg)
        self.elev_range: Tuple[float, float] = getattr(config, 'ELEV_RANGE', (-20.0, 20.0))
        self.azim_range: Tuple[float, float] = getattr(config, 'AZIM_RANGE', (-180.0, 180.0))

        # 하위 폴더(카테고리) 스캔
        # 예:  ~/data/abc/*/*.obj
        # split별로 구분하고 싶다면, 예) ~/data/abc/train/categoryA/*.obj ... 식으로 구성해야 하며
        #  -> 여기서는 단순히 root_dir 내 모든 카테고리 폴더를 뒤져서 .obj 목록을 읽는다고 가정
        #  -> 만약 'train','test'가 별도 디렉토리면 그에 맞게 경로 설정
        pattern = os.path.join(self.root_dir, "*", "*.obj")
        # glob
        all_obj_files = sorted(glob.glob(pattern))
        # 예를 들어 subset=='train'이면 train_list.txt를 읽어서 필터링할 수도 있음(사용자가)
        # 여기서는 "train/val/test"를 별도 관리하지 않는다고 가정,
        #  -> 실제로는 config.subset에 따라 리스트를 나누거나, split txt 파일을 읽을 수 있음

        self.samples: List[dict] = []
        self.class2idx = {}
        current_idx = 0

        for f in all_obj_files:
            # f: "/home/xxx/data/abc/categoryA/mesh_001.obj"
            # 카테고리명 추출
            category_name = os.path.basename(os.path.dirname(f))  # categoryA
            if category_name not in self.class2idx:
                self.class2idx[category_name] = current_idx
                current_idx += 1
            label = self.class2idx[category_name]

            # model_id (파일 이름)
            model_id = os.path.splitext(os.path.basename(f))[0]  # "mesh_001"

            self.samples.append({
                'file_path': f,
                'category': category_name,
                'label': label,
                'model_id': model_id
            })

        print_log(f"[ABCDataset] Found {len(self.samples)} .obj files in '{self.root_dir}'", logger='ABCDataset')
        print_log(f"[ABCDataset] #classes={len(self.class2idx)}", logger='ABCDataset')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_info = self.samples[index]
        file_path: str = sample_info['file_path']
        label: int = sample_info['label']
        category: str = sample_info['category']
        model_id: str = sample_info['model_id']

        # 1) Load mesh ------------------------------------------------------------------
        try:
            verts, faces_idx, _ = load_obj(file_path, load_textures=False)
            faces = faces_idx.verts_idx
            mesh = Meshes(verts=[verts], faces=[faces])
        except Exception as e:
            print_log(f"Error loading mesh {file_path}: {e}", logger='ABCDataset')
            verts = torch.zeros((1, 3), dtype=torch.float32)
            faces = torch.zeros((1, 3), dtype=torch.long)
            mesh = Meshes(verts=[verts], faces=[faces])
            label = -1  # mark invalid

        # 2) Convert to point cloud via simple ray‑casting ------------------------------
        points = self._mesh_to_pointcloud(mesh)

        # 3) Pack with the standard return format used in other datasets ---------------
        other_data = {
            'points': points,                           # (N,3) tensor
            'label': torch.tensor(label, dtype=torch.long)
        }

        return category, model_id, other_data

    # ------------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------------
    def _mesh_to_pointcloud(self, mesh: Meshes) -> torch.Tensor:
        """Render a mesh from a randomly sampled camera and obtain 3D points via
        ray‑casting (z‑buffer unprojection).

        The randomisation range is controlled by the following (set in __init__):
            self.fov_range   : Tuple[float, float] in degrees (vertical FOV)
            self.res_range   : Tuple[int,  int]   (min_res, max_res)
            self.dist_range  : Tuple[float,float] camera distance from object
        Returns
        -------
        Tensor (P,3)  –  point cloud in world coordinates.
        """
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        mesh = mesh.to(device)

        # ---- Random camera parameters -------------------------------------------
        fov = random.uniform(*self.fov_range)  # vertical FOV in deg
        res = random.randint(*self.res_range)  # square image for simplicity
        dist = random.uniform(*self.dist_range)
        elev = random.uniform(*self.elev_range)
        azim = random.uniform(*self.azim_range)

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

        raster_settings = RasterizationSettings(
            image_size=res,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(mesh)

        # Depth buffer and mask ---------------------------------------------------
        depth = fragments.zbuf[..., 0]  # (1, H, W)
        mask = fragments.pix_to_face[..., 0] >= 0  # valid pixels

        if mask.sum() == 0:
            # Fallback: sample from surface if nothing visible (rare)
            pts = mesh.sample_points_from_faces(self.sample_points_num)[0]
            return pts.cpu()

        # Screen coords grid (NDC) ----------------------------------------------
        H, W = depth.shape[1:]
        xs = torch.linspace(0, W - 1, W, device=device) + 0.5  # center of pixel
        ys = torch.linspace(0, H - 1, H, device=device) + 0.5
        xs = xs / W * 2 - 1  # [0,W] -> [-1,1]
        ys = ys / H * 2 - 1

        ys_grid, xs_grid = torch.meshgrid(ys, xs, indexing='ij')
        # Flatten
        xs_flat = xs_grid[mask[0]]
        ys_flat = ys_grid[mask[0]]
        zs_flat = depth[0][mask[0]]

        screen_pts = torch.stack([xs_flat, ys_flat, zs_flat], dim=-1).unsqueeze(0)  # (1, P, 3)

        # Unproject to world ------------------------------------------------------
        world_pts = cameras.unproject_points(screen_pts, world_coordinates=True)  # (1,P,3)
        pts = world_pts.squeeze(0)  # (P,3)

        # Optionally subsample to fixed number of points -------------------------
        if pts.shape[0] > self.sample_points_num:
            idx = torch.randperm(pts.shape[0], device=device)[: self.sample_points_num]
            pts = pts[idx]
        elif pts.shape[0] < self.sample_points_num:
            # Repeat to reach desired number
            repeat_factor = int(np.ceil(self.sample_points_num / pts.shape[0]))
            pts = pts.repeat(repeat_factor, 1)[: self.sample_points_num]

        return pts.cpu()
