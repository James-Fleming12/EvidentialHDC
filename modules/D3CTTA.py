import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class D3CTTA(nn.Module):
    """
    Accurate implementation of Distance-Aware Domain-Agnostic Test-Time Adaptation (D3CTTA).
    Ref: https://arxiv.org/abs/2403.11111 (CVPR 2024)
    
    Supports both:
    1. Standard 3D Sparse Voxel networks (e.g., MinkUNet18 from MinkowskiEngine with 96-dim feature output).
    2. 2D Range-View encoders (e.g., CENet, Range-View ResNet, HDC encoders with 128-dim or 256-dim output).
    """
    def __init__(self, feature_extractor, num_classes=17, feature_dim=None, proj_dim=1024, lambda_ridge=0.1, source_prototypes=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.proj_dim = proj_dim
        self.lambda_ridge = lambda_ridge
        
        # Zero out bias in semantic output head if present (standard in TTA warmup)
        if hasattr(self.feature_extractor, 'semantic_output') and hasattr(self.feature_extractor.semantic_output, 'bias'):
            if self.feature_extractor.semantic_output.bias is not None:
                self.feature_extractor.semantic_output.bias.data.zero_()
        
        # Determine initial feature dimension
        if feature_dim is not None:
            self.feature_dim = feature_dim
        elif source_prototypes is not None:
            self.feature_dim = source_prototypes.shape[-1]
        elif hasattr(self.feature_extractor, 'semantic_output'):
            self.feature_dim = self.feature_extractor.semantic_output.weight.shape[1]
        elif hasattr(self.feature_extractor, 'final') and hasattr(self.feature_extractor.final, 'kernel'):
            # D3CTTA / MinkowskiEngine style classifier weights [feature_dim, num_classes]
            self.feature_dim = self.feature_extractor.final.kernel.shape[0]
        else:
            self.feature_dim = 128  # Default fallback
            
        # 1. Initialize random projection matrix W (unscaled standard normal N(0, 1) matching D3CTTA w_rand)
        self.W = None
        self.init_projection_matrix(self.feature_dim)
        
        # 2. Extract original prototypes (warmup supports)
        self.source_prototypes = None
        self.init_source_prototypes(source_prototypes)
            
        # 3. Distance-Aware Prototype Learning (DAPL) setup
        self.num_areas_d = 3
        self.proto = [self.source_prototypes.clone() for _ in range(self.num_areas_d)]
        self.alpha = 0.95  # EMA momentum
        self.min_feat = 1

        # 4. Recursive Ridge Regression setup
        self.domain_id = 0
        self.domains_bn_stats = {}  # domain_id -> {'mu': mu, 'sigma': sigma}
        self.G_d = {}  # domain_id -> [proj_dim, proj_dim]
        self.C_d = {}  # domain_id -> [proj_dim, num_classes]
        
        self.prev_mu = None
        self.feat_source = None  # Temporary storage for unprojected features between forward and update
        self.pred_source = None  # Temporary storage for base predictions
        
        self.create_new_domain(0)

    def init_projection_matrix(self, feature_dim):
        """Initialize random projection matrix with unscaled standard normal distribution N(0, 1)."""
        self.feature_dim = feature_dim
        self.W = nn.Linear(feature_dim, self.proj_dim, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.W.weight, mean=0.0, std=1.0)  # Unscaled std=1.0 matching D3CTTA w_rand
        self.W.weight.requires_grad = False
        if next(self.parameters(), None) is not None:
            self.W = self.W.to(next(self.parameters()).device)

    def init_source_prototypes(self, source_prototypes=None):
        """Extract or initialize source prototypes from the feature extractor."""
        if source_prototypes is not None:
            self.source_prototypes = source_prototypes.clone()
        elif hasattr(self.feature_extractor, 'semantic_output'):
            source_weight = self.feature_extractor.semantic_output.weight.data.clone()
            self.source_prototypes = source_weight.view(self.num_classes, -1)
        elif hasattr(self.feature_extractor, 'final') and hasattr(self.feature_extractor.final, 'kernel'):
            # Support D3CTTA / MinkowskiEngine sparse voxel classifier heads [feature_dim, num_classes]
            self.source_prototypes = self.feature_extractor.final.kernel.data.clone().T
        elif hasattr(self.feature_extractor, 'classifier') and isinstance(self.feature_extractor.classifier, nn.Linear):
            self.source_prototypes = self.feature_extractor.classifier.weight.data.clone()
        else:
            self.source_prototypes = torch.zeros(self.num_classes, self.feature_dim)
        
        if next(self.parameters(), None) is not None:
            self.source_prototypes = self.source_prototypes.to(next(self.parameters()).device)

    def get_last_bn_stats(self):
        """Extract running mean and std from the last BatchNorm layer for domain shift detection."""
        last_bn = None
        for module in self.feature_extractor.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                last_bn = module
            elif module.__class__.__name__ in ['MinkowskiBatchNorm', 'DistanceBasedBatchNorm']:
                last_bn = module
        if last_bn is not None and hasattr(last_bn, 'running_mean') and last_bn.running_mean is not None:
            return last_bn.running_mean.detach().clone(), torch.sqrt(last_bn.running_var.detach().clone() + 1e-5)
        return None, None

    def create_new_domain(self, domain_id, mu=None, sigma=None):
        """Initialize recursive ridge regression matrices G_d and C_d for a newly detected domain."""
        device = next(self.parameters(), torch.tensor(0)).device
        if isinstance(device, torch.Tensor):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G_d[domain_id] = torch.zeros(self.proj_dim, self.proj_dim, device=device)
        self.C_d[domain_id] = torch.zeros(self.proj_dim, self.num_classes, device=device)
        if mu is not None and sigma is not None:
            self.domains_bn_stats[domain_id] = {'mu': mu, 'sigma': sigma}

    def forward(self, x, xyz=None, *args, **kwargs):
        with torch.no_grad():
            out = self.feature_extractor(x)
            if isinstance(out, tuple):
                if len(out) == 2:
                    base_pred, feat = out
                else: 
                    base_pred = out[0]
                    feat = out[-1]
            else:
                feat = out
                base_pred = None
            
            # Extract feature tensor (support 2D sparse point features [N, C], 3D [B, N, C], and 4D range images [B, C, H, W])
            if hasattr(feat, 'F'):  # MinkowskiEngine SparseTensor
                feat_flat = feat.F
            elif feat.dim() == 4:
                feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
            elif feat.dim() == 3:
                feat_flat = feat.reshape(-1, feat.shape[-1])
            else:
                feat_flat = feat

            # Dynamically adapt feature_dim and projection matrix W if extractor output dimension differs (e.g., 96 vs 128)
            current_dim = feat_flat.shape[-1]
            if self.W is None or current_dim != self.feature_dim:
                self.init_projection_matrix(current_dim)
                self.init_source_prototypes()
                self.proto = [self.source_prototypes.clone() for _ in range(self.num_areas_d)]
                self.W = self.W.to(feat_flat.device)
                self.source_prototypes = self.source_prototypes.to(feat_flat.device)
                for i in range(len(self.proto)):
                    self.proto[i] = self.proto[i].to(feat_flat.device)

            if base_pred is not None:
                if hasattr(base_pred, 'F'):
                    self.pred_source = base_pred.F
                elif base_pred.dim() == 4:
                    self.pred_source = base_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                elif base_pred.dim() == 3:
                    self.pred_source = base_pred.reshape(-1, self.num_classes)
                else:
                    self.pred_source = base_pred
            else:
                self.pred_source = feat_flat @ self.source_prototypes.T.to(feat_flat.device)
                
            self.feat_source = feat_flat
            
            # Random projection activation: ReLU(W * feat)
            h = F.relu(self.W(feat_flat))
            
            mu, sigma = self.get_last_bn_stats()
            
            # Domain shift detection via BatchNorm statistics cosine similarity
            if mu is not None and self.prev_mu is not None:
                cos_sim = F.cosine_similarity(mu, self.prev_mu, dim=0)
                if cos_sim <= 0.85:
                    best_dist = float('inf')
                    best_domain = -1
                    for d_id, stats in self.domains_bn_stats.items():
                        dist = torch.sum((mu - stats['mu'])**2 + (sigma - stats['sigma'])**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_domain = d_id
                    
                    if best_domain != -1 and best_dist < 10.0:
                        self.domain_id = best_domain
                    else:
                        self.domain_id = len(self.domains_bn_stats)
                        self.create_new_domain(self.domain_id, mu, sigma)
                        
            self.prev_mu = mu
            if self.domain_id not in self.domains_bn_stats and mu is not None:
                self.domains_bn_stats[self.domain_id] = {'mu': mu, 'sigma': sigma}

            device = h.device
            G = self.G_d[self.domain_id].to(device)
            C = self.C_d[self.domain_id].to(device)
            
            if C.sum() == 0 and self.pred_source is not None:
                logits = self.pred_source
            else:
                I = torch.eye(self.proj_dim, device=device)
                G_inv = torch.linalg.inv(G + self.lambda_ridge * I)
                logits = h @ G_inv @ C
            
        return logits, None, torch.arange(logits.shape[0], device=logits.device), h

    def distance_partition(self, points):
        """Partition points into 3 distance zones matching D3CTTA."""
        if points is None:
            return [list(range(self.feat_source.shape[0]))] * self.num_areas_d
            
        distance = torch.sqrt(points[:, 0]**2 + points[:, 1]**2)
        distance = torch.clamp(distance, 1e-3, 50.0 - 1e-3)  # Range up to 50m as per paper
        distance_list = np.linspace(0, 50.0, self.num_areas_d + 1)
        
        distance_labels = np.digitize(distance.detach().cpu().numpy(), bins=distance_list) - 1
        distance_labels = np.clip(distance_labels, 0, self.num_areas_d - 1)
        
        idx_all = []
        for i in range(self.num_areas_d):
            idx_all.append(list(np.where(distance_labels == i)[0]))
        return idx_all

    def update_proto_multi(self, pred_proto, feat, area):
        """EMA update of regional prototypes (DAPL)."""
        pred_label = pred_proto
        for i in range(self.num_classes):
            index_class = (pred_label == i)
            feat_i = feat[index_class].detach()
            if feat_i.shape[0] < self.min_feat:
                continue
            mean = torch.mean(feat_i, dim=0)
            self.proto[area][i] = self.alpha * self.proto[area][i] + (1 - self.alpha) * mean

    def select_pseudo(self, pred_seg, ent, ratio):
        """
        D3CTTA dynamic per-class pseudo-label selection based on entropy.
        Selects the top `ratio` fraction of most confident points independently for each class.
        """
        selected_indices = []
        pred_labels = pred_seg.argmax(dim=1)
        
        for label in range(self.num_classes):
            label_indices = torch.nonzero(pred_labels == label).squeeze(1)
            if label_indices.numel() <= 1:
                continue
            
            label_entropy = ent[label_indices]
            sorted_idx = torch.argsort(label_entropy)
            num_selected = max(1, math.ceil(ratio * label_indices.numel()))
            num_selected = min(num_selected, label_indices.numel())
            
            selected_label_indices = label_indices[sorted_idx[:num_selected]]
            selected_indices.append(selected_label_indices)
            
        if len(selected_indices) == 0:
            return torch.zeros(len(pred_seg), dtype=torch.bool, device=pred_seg.device)
            
        selected_cat = torch.cat(selected_indices)
        mask = torch.zeros(len(pred_seg), dtype=torch.bool, device=pred_seg.device)
        mask[selected_cat] = True
        return mask

    def prior_filter(self, pred, points):
        """
        Geometric Prior Filtering using Open3D (or SciPy KDTree fallback).
        Returns two distinct masks (ground_filter, manmade_filter) matching D3CTTA logic.
        """
        if points is None:
            ones = torch.ones(len(pred), dtype=torch.bool, device=pred.device)
            return ones, ones
            
        orig_device = pred.device
        pred_labels = pred.argmax(1).detach().cpu().numpy()
        points_np = points.detach().cpu().numpy()
        
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
            normals = np.fabs(np.asarray(pcd.normals)[:, 2])
        except ImportError:
            try: # the one that actually runs (since open3d cant be installed in my env)
                from scipy.spatial import cKDTree
                tree = cKDTree(points_np)
                dists, idxs = tree.query(points_np, k=min(30, len(points_np)), distance_upper_bound=2.0, workers=-1)
                
                invalid_mask = np.isinf(dists)
                idxs[invalid_mask] = 0
                neighbors = points_np[idxs]
                
                query_points = points_np[:, np.newaxis, :]
                neighbors = np.where(invalid_mask[:, :, np.newaxis], query_points, neighbors)
                
                centroids = np.mean(neighbors, axis=1, keepdims=True)
                centered = neighbors - centroids
                covs = np.einsum('nij,nik->njk', centered, centered)
                
                w, v = np.linalg.eigh(covs)
                normals = np.abs(v[:, 2, 0])
            except ImportError:
                ones = torch.ones(len(pred), dtype=torch.bool, device=orig_device)
                return ones, ones

        plane_norm_index = normals > 0.9
        manmade_norm_index = normals < 0.1
        
        # Taxonomy mapping (approx for 17-class SemanticKITTI / NuScenes vs D3CTTA 7-class taxonomy)
        # Plane (Road, Sidewalk, Parking, Other Ground)
        plane_pred_index = ((pred_labels == 11) | (pred_labels == 12) | (pred_labels == 13) | (pred_labels == 14))
        # Manmade (Building, Fence, Trunk, Pole)
        manmade_pred_index = ((pred_labels == 15) | (pred_labels == 16) | (pred_labels == 5))
        # Others (Cars, Pedestrians, Vegetation, etc.)
        other_index = ~(plane_pred_index | manmade_pred_index)

        ground_index = plane_pred_index & plane_norm_index
        manmade_index = manmade_pred_index & manmade_norm_index

        g_filter = ground_index | other_index | manmade_pred_index
        m_filter = manmade_index | other_index | plane_pred_index
        
        return torch.tensor(g_filter, device=orig_device), torch.tensor(m_filter, device=orig_device)

    def optimise_ridge_parameter(self, Features, Y):
        """
        D3CTTA dynamic ridge regression parameter optimization.
        Searches over candidate ridge values on an 80/20 train/val split of filtered batch features.
        """
        if Features.shape[0] < 10:
            return self.lambda_ridge
            
        ridges = 10.0 ** np.arange(-8, 9, dtype=np.float32)
        num_val_samples = int(Features.shape[0] * 0.8)
        if num_val_samples < 5 or num_val_samples >= Features.shape[0]:
            return self.lambda_ridge
            
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        I = torch.eye(G_val.size(0), device=Features.device)
        
        for ridge in ridges:
            try:
                Wo = torch.linalg.solve(G_val + float(ridge) * I, Q_val).T
                Y_train_pred = Features[num_val_samples:, :] @ Wo.T
                loss = F.mse_loss(Y_train_pred, Y[num_val_samples:, :]).item()
                losses.append(loss)
            except RuntimeError:
                losses.append(float('inf'))
                
        if len(losses) == 0 or all(math.isinf(l) for l in losses):
            return self.lambda_ridge
            
        best_ridge = float(ridges[np.argmin(losses)])
        self.lambda_ridge = best_ridge
        return best_ridge

    def inference_update(self, h, predictions, xyz):
        """
        Accurate 5-step D3CTTA online adaptation loop:
        1. KNN consistency check on base predictions to determine dynamic ratio.
        2. Per-class entropy pseudo-label selection using dynamic ratio.
        3. Two-mask geometric prior filtering.
        4. Distance-Aware Prototype Learning (DAPL) regional prototype EMA updates.
        5. Recursive Ridge Regression update with dynamic ridge parameter optimization.
        """
        device = h.device
        if xyz is not None and xyz.dim() == 4:
            xyz = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
            
        if self.feat_source is None:
            return
            
        valid_mask = (self.feat_source.sum(dim=1) != 0)
        valid_idx = torch.nonzero(valid_mask).squeeze(1)
        
        if valid_idx.numel() <= 20:
            return
            
        feat_valid = self.feat_source[valid_idx]
        h_valid = h[valid_idx]
        pred_source_valid = self.pred_source[valid_idx]
        xyz_valid = xyz[valid_idx] if xyz is not None else None
        
        # 1. KNN Consistency on initial predictions (determines dynamic selection ratio)
        pred_source_argmax = pred_source_valid.argmax(dim=1)
        consistent_indices = torch.zeros(feat_valid.shape[0], dtype=torch.bool, device=device)
        
        if xyz_valid is not None and len(xyz_valid) > 20:
            chunk_size = 2000
            for i in range(0, feat_valid.shape[0], chunk_size):
                end = min(i + chunk_size, feat_valid.shape[0])
                xyz_chunk = xyz_valid[i:end]
                
                dists = torch.cdist(xyz_chunk.unsqueeze(0), xyz_valid.unsqueeze(0)).squeeze(0)
                k_val = min(21, feat_valid.shape[0])
                _, knn_idx = torch.topk(dists, k=k_val, dim=1, largest=False)
                
                knn_preds = pred_source_argmax[knn_idx]
                center_preds = pred_source_argmax[i:end].unsqueeze(1)
                consistency = (knn_preds == center_preds).float().mean(dim=1)
                
                consistent_indices[i:end] = consistency > 0.8
        else:
            consistent_indices = torch.ones(feat_valid.shape[0], dtype=torch.bool, device=device)
            
        ratio = float(consistent_indices.sum().item()) / max(1.0, float(len(consistent_indices)))

        # 2. Per-Class Entropy Pseudo-Label Selection using dynamic ratio
        ent = softmax_entropy(pred_source_valid)
        indices_ent = self.select_pseudo(pred_source_valid, ent, ratio)

        # 3. Two-mask Geometric Prior Filtering
        g_index, m_index = self.prior_filter(pred_source_valid, xyz_valid)
        
        # Combine initial filters for DAPL update
        indices_filter = consistent_indices & indices_ent & g_index & m_index

        # 4. Distance-Aware Prototype Learning (DAPL)
        indices_parts = self.distance_partition(xyz_valid)
        pred_proto = torch.ones_like(pred_source_valid)
        
        for i in range(self.num_areas_d):
            indices = indices_parts[i]
            if len(indices) == 0:
                continue
                
            proto_i = self.proto[i].to(device)
            pred_proto[indices] = feat_valid[indices] @ F.normalize(proto_i, dim=1).T
            
            valid_area_mask = torch.zeros(len(feat_valid), dtype=torch.bool, device=device)
            valid_area_mask[indices] = True
            update_mask = valid_area_mask & indices_filter
            
            if update_mask.sum() > 0:
                self.update_proto_multi(pred_source_valid.argmax(1)[update_mask], feat_valid[update_mask], i)

        # 5. Recursive Ridge Regression Update on refined DAPL pseudo-labels
        pred_proto_argmax = pred_proto.argmax(dim=1)
        g_index_proto, m_index_proto = self.prior_filter(pred_proto, xyz_valid)
        final_indices = consistent_indices & g_index_proto & m_index_proto
        
        h_filtered = h_valid[final_indices]
        pred_filtered = pred_proto_argmax[final_indices]
        
        if h_filtered.shape[0] > 0:
            self.G_d[self.domain_id] = self.G_d[self.domain_id].to(device)
            self.C_d[self.domain_id] = self.C_d[self.domain_id].to(device)
            self.G_d[self.domain_id] += h_filtered.T @ h_filtered
            
            y_one_hot = F.one_hot(pred_filtered, num_classes=self.num_classes).float().to(device)
            self.C_d[self.domain_id] += h_filtered.T @ y_one_hot
            
            # Dynamically optimize ridge regression parameter
            self.optimise_ridge_parameter(h_filtered.detach(), y_one_hot.detach())