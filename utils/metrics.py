# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-08 14:31:30
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2020-05-25 09:13:32
# @Email:  cshzxie@gmail.com

import logging
import open3d
import torch
import numpy as np
# import faiss # Import faiss if installed

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

def calculate_recall_at_k(embeddings, labels, k_values, normalize=True):
    \"\"\
    Calculates Recall@k for given embeddings and labels.

    Args:
        embeddings (torch.Tensor or np.ndarray): Embeddings matrix (N, D).
        labels (torch.Tensor or np.ndarray): Labels vector (N,).
        k_values (list or tuple): List of k values for Recall@k.
        normalize (bool): Whether to normalize embeddings before distance calculation.

    Returns:
        dict: Dictionary containing Recall@k for each k in k_values.
              e.g., {'R@1': 65.2, 'R@5': 80.1}
    \"\"\
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    N, D = embeddings.shape
    if N == 0:
        print("[Metrics] Warning: Empty embeddings received for Recall@k calculation.")
        return {f'R@{k}': np.nan for k in k_values}

    # Ensure k_values are valid and adjust max_k if necessary
    k_values = sorted([k for k in k_values if isinstance(k, int) and k > 0])
    if not k_values:
        print("[Metrics] Error: No valid positive integer k values provided.")
        return {}
    max_k = max(k_values)

    if max_k >= N:
        print(f"[Metrics] Warning: max_k ({max_k}) >= number of samples ({N}). Adjusting k values.")
        max_k = N - 1
        k_values = [k for k in k_values if k <= max_k]
        if not k_values:
             print("[Metrics] Error: No valid k values left after adjustment (max_k={max_k}).")
             return {f'R@{k}': np.nan for k in k_values} # Return NaN for original k values

    if normalize:
        # Normalize embeddings for cosine similarity calculation
        embeddings_norm = embeddings.astype(np.float32) # Ensure float32 for normalization
        norms = np.linalg.norm(embeddings_norm, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms < 1e-8] = 1e-8 # Check against small value
        embeddings_norm = embeddings_norm / norms
    else:
        # Ensure float32 even if not normalizing, Faiss might require it
        embeddings_norm = embeddings.astype(np.float32) if embeddings.dtype != np.float32 else embeddings

    neighbor_indices = None
    # --- Try Using Faiss --- (Recommended for speed)
    try:
        import faiss
        print("[Metrics] Using Faiss for Recall@k calculation.")
        # Use IndexFlatIP for cosine similarity (since vectors are normalized)
        # Use IndexFlatL2 for Euclidean distance (if normalize=False)
        index = faiss.IndexFlatIP(D) if normalize else faiss.IndexFlatL2(D)

        # Add vectors to index
        index.add(embeddings_norm)

        # Search for top max_k + 1 neighbors (to potentially exclude self)
        distances, indices = index.search(embeddings_norm, max_k + 1)

        # Exclude self from results. Self is usually the first result (distance 0 or 1).
        neighbor_indices = np.zeros((N, max_k), dtype=np.int64)
        for i in range(N):
            # Filter out self (index i)
            valid_neighbors = indices[i, indices[i] != i]
            # Take up to max_k neighbors
            num_valid = min(len(valid_neighbors), max_k)
            neighbor_indices[i, :num_valid] = valid_neighbors[:num_valid]
            # If fewer than max_k neighbors found (excluding self), fill rest with -1 or handle later
            if num_valid < max_k:
                neighbor_indices[i, num_valid:] = -1 # Indicate invalid/missing neighbors

    except ImportError:
        print("[Metrics] Faiss not found or import error. Using slower NumPy fallback for Recall@k.")
    except Exception as e:
        print(f"[Metrics] Error during Faiss processing: {e}. Using NumPy fallback.")

    # --- NumPy Fallback --- (If Faiss failed or not installed)
    if neighbor_indices is None:
        if normalize:
            # Calculate cosine similarity matrix
            similarity_matrix = embeddings_norm @ embeddings_norm.T # (N, N)
            # Set diagonal to a very low value to exclude self
            np.fill_diagonal(similarity_matrix, -np.inf)
            # Get indices of top max_k neighbors
            neighbor_indices = np.argsort(similarity_matrix, axis=1)[:, -max_k:][:, ::-1] # (N, max_k)
        else:
            # Calculate L2 distance matrix (more memory intensive)
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(embeddings_norm, embeddings_norm, metric='euclidean') # (N, N)
            np.fill_diagonal(dist_matrix, np.inf) # Exclude self
            neighbor_indices = np.argsort(dist_matrix, axis=1)[:, :max_k] # (N, max_k)


    # --- Calculate Recall --- 
    recalls = {}
    labels_expanded = labels[:, np.newaxis]
    for k in k_values:
        # Get labels of the top-k neighbors for current k
        # Handle potential -1 indices from Faiss if fewer neighbors were found
        current_neighbor_indices = neighbor_indices[:, :k]

        # Create a mask for valid neighbor indices (>= 0)
        valid_mask = current_neighbor_indices != -1

        # Get labels only for valid neighbors, fill invalid spots (e.g., with a value that won't match)
        neighbor_labels = np.full_like(current_neighbor_indices, -999, dtype=labels.dtype)
        valid_indices_flat = current_neighbor_indices[valid_mask]

        if valid_indices_flat.size > 0: # Check if there are any valid indices
             neighbor_labels[valid_mask] = labels[valid_indices_flat]

        # Check if the true label exists in the valid top-k neighbors
        matches = np.any(neighbor_labels == labels_expanded, axis=1)

        # Calculate recall
        recall_value = np.mean(matches) * 100.0
        recalls[f'R@{k}'] = recall_value

    return recalls

class Metrics(object):
    ITEMS = [{
        'name': 'F-Score',
        'enabled': True,
        'eval_func': 'cls._get_f_score',
        'is_greater_better': True,
        'init_value': 0
    }, {
        'name': 'CDL1',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel1',
        'eval_object': ChamferDistanceL1(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }, {
        'name': 'CDL2',
        'enabled': True,
        'eval_func': 'cls._get_chamfer_distancel2',
        'eval_object': ChamferDistanceL2(ignore_zeros=True),
        'is_greater_better': False,
        'init_value': 32767
    }]

    @classmethod
    def get(cls, pred, gt):
        _items = cls.items()
        _values = [0] * len(_items)
        for i, item in enumerate(_items):
            eval_func = eval(item['eval_func'])
            _values[i] = eval_func(pred, gt)

        return _values

    @classmethod
    def items(cls):
        return [i for i in cls.ITEMS if i['enabled']]

    @classmethod
    def names(cls):
        _items = cls.items()
        return [i['name'] for i in _items]

    @classmethod
    def _get_f_score(cls, pred, gt, th=0.01):
        
        """References: https://github.com/lmb-freiburg/what3d/blob/master/util.py"""
        b = pred.size(0)
        assert pred.size(0) == gt.size(0)
        if b != 1:
            f_score_list = []
            for idx in range(b):
                f_score_list.append(cls._get_f_score(pred[idx:idx+1], gt[idx:idx+1]))
            return sum(f_score_list)/len(f_score_list)
        else:
            pred = cls._get_open3d_ptcloud(pred)
            gt = cls._get_open3d_ptcloud(gt)

            dist1 = pred.compute_point_cloud_distance(gt)
            dist2 = gt.compute_point_cloud_distance(pred)

            recall = float(sum(d < th for d in dist2)) / float(len(dist2))
            precision = float(sum(d < th for d in dist1)) / float(len(dist1))
            return 2 * recall * precision / (recall + precision) if recall + precision else 0

    @classmethod
    def _get_open3d_ptcloud(cls, tensor):
        """pred and gt bs is 1"""
        tensor = tensor.squeeze().cpu().numpy()
        ptcloud = open3d.geometry.PointCloud()
        ptcloud.points = open3d.utility.Vector3dVector(tensor)

        return ptcloud

    @classmethod
    def _get_chamfer_distancel1(cls, pred, gt):
        chamfer_distance = cls.ITEMS[1]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000

    @classmethod
    def _get_chamfer_distancel2(cls, pred, gt):
        chamfer_distance = cls.ITEMS[2]['eval_object']
        return chamfer_distance(pred, gt).item() * 1000

    def __init__(self, metric_name, values):
        self._items = Metrics.items()
        self._values = [item['init_value'] for item in self._items]
        self.metric_name = metric_name

        if type(values).__name__ == 'list':
            self._values = values
        elif type(values).__name__ == 'dict':
            metric_indexes = {}
            for idx, item in enumerate(self._items):
                item_name = item['name']
                metric_indexes[item_name] = idx
            for k, v in values.items():
                if k not in metric_indexes:
                    logging.warn('Ignore Metric[Name=%s] due to disability.' % k)
                    continue
                self._values[metric_indexes[k]] = v
        else:
            raise Exception('Unsupported value type: %s' % type(values))

    def state_dict(self):
        _dict = dict()
        for i in range(len(self._items)):
            item = self._items[i]['name']
            value = self._values[i]
            _dict[item] = value

        return _dict

    def __repr__(self):
        return str(self.state_dict())

    def better_than(self, other):
        if other is None:
            return True

        _index = -1
        for i, _item in enumerate(self._items):
            if _item['name'] == self.metric_name:
                _index = i
                break
        if _index == -1:
            raise Exception('Invalid metric name to compare.')

        _metric = self._items[i]
        _value = self._values[_index]
        other_value = other._values[_index]
        return _value > other_value if _metric['is_greater_better'] else _value < other_value
