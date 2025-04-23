import os
import glob
import torch
import numpy as np
import torch.utils.data as data
# import trimesh # Remove trimesh import

# Import PyTorch3D components
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

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
        # self.npoints = config.N_POINTS # No longer sampling points
        self.subset = getattr(config, "subset", "train")  # train/val/test
        # self.rot_aug = getattr(config, "rot_aug", False) # Removed

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

        self.samples = []
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
        file_path = sample_info['file_path']
        label = sample_info['label']
        category = sample_info['category']
        model_id = sample_info['model_id']

        # .obj -> pytorch3d mesh
        try:
            # Load using pytorch3d.io.load_obj
            # Returns verts, faces (Faces object), aux (Auxiliary object)
            verts, faces_idx, _ = load_obj(file_path, load_textures=False)
            faces = faces_idx.verts_idx

            # Create a PyTorch3D Meshes object
            # Meshes expects batch dimensions, so wrap verts and faces in lists
            mesh = Meshes(verts=[verts], faces=[faces])

        except Exception as e:
            print_log(f"Error loading mesh {file_path}: {e}", logger='ABCDataset')
            # Return None or raise error, depending on desired handling
            # Here, we'll return None or raise error if loading fails
            # For simplicity, let's create a dummy mesh if loading fails
            verts = torch.zeros((1, 3), dtype=torch.float32)
            faces = torch.zeros((1, 3), dtype=torch.long)
            mesh = Meshes(verts=[verts], faces=[faces])
            # Also set label to something invalid, e.g., -1, if needed for filtering later
            label = -1 # Indicate failure

        # Removed point cloud specific operations:
        # pc = self.load_obj_as_points(file_path)
        # pc = pc_normalize(pc)
        # if self.rot_aug:
        #     R = rnd_rot()
        #     pc = pc @ R
        # pc_tensor = torch.from_numpy(pc).float() # (N,3)

        # Return the mesh object and label
        return category, model_id, (mesh, label)
