import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def softmax_entropy(x):
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class D3CTTA(nn.Module):
    def __init__(self, feature_extractor, num_classes=13, feature_dim=128, proj_dim=145, lambda_ridge=0.1, source_prototypes=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.lambda_ridge = lambda_ridge
        
        if hasattr(self.feature_extractor, 'semantic_output') and hasattr(self.feature_extractor.semantic_output, 'bias'):
            if self.feature_extractor.semantic_output.bias is not None:
                self.feature_extractor.semantic_output.bias.data.zero_()
        
        # 1. Initialize random projection matrix W
        self.W = nn.Linear(feature_dim, proj_dim, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.W.weight, mean=0, std=1.0 / math.sqrt(feature_dim))
        self.W.weight.requires_grad = False
        
        # 2. Extract original prototypes (warmup supports)
        if source_prototypes is not None:
            self.source_prototypes = source_prototypes.clone()
        elif hasattr(self.feature_extractor, 'semantic_output'):
            source_weight = self.feature_extractor.semantic_output.weight.data.clone()
            # source_weight is [num_classes, feature_dim, 1, 1]
            self.source_prototypes = source_weight.view(num_classes, feature_dim)
        else:
            self.source_prototypes = torch.zeros(num_classes, feature_dim)
            
        # 3. Distance-Aware Prototype Learning (DAPL) setup
        self.num_areas_d = 3
        # proto stores 3 independent sets of prototypes (one for each distance zone)
        self.proto = [self.source_prototypes.clone() for _ in range(self.num_areas_d)]
        self.alpha = 0.95  # EMA momentum
        self.min_feat = 1

        # 4. Recursive Ridge Regression setup
        self.domain_id = 0
        self.domains_bn_stats = {} # domain_id -> {'mu': mu, 'sigma': sigma}
        self.G_d = {} # domain_id -> [proj_dim, proj_dim]
        self.C_d = {} # domain_id -> [proj_dim, num_classes]
        
        self.prev_mu = None
        self.feat_source = None  # Temporary storage for unprojected features between forward and update
        self.pred_source = None  # Temporary storage for base predictions
        
        self.create_new_domain(0)

    def get_last_bn_stats(self):
        last_bn = None
        for module in self.feature_extractor.modules():
            if isinstance(module, nn.BatchNorm2d):
                last_bn = module
        if last_bn is not None:
            return last_bn.running_mean.detach().clone(), torch.sqrt(last_bn.running_var.detach().clone() + 1e-5)
        return None, None

    def create_new_domain(self, domain_id, mu=None, sigma=None):
        device = next(self.parameters()).device
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
            
            feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, self.feature_dim)
            if base_pred is not None:
                self.pred_source = base_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            else:
                self.pred_source = feat_flat @ self.source_prototypes.T.to(feat_flat.device)
                
            self.feat_source = feat_flat
            
            h = F.relu(self.W(feat_flat))
            
            mu, sigma = self.get_last_bn_stats()
            
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
        distance = torch.clamp(distance, 0+1e-3, 50-1e-3) # Range up to 50m as per paper
        distance_list = np.linspace(0, 50, self.num_areas_d + 1)
        
        distance_labels = np.digitize(distance.detach().cpu().numpy(), bins=distance_list) - 1
        distance_labels = np.clip(distance_labels, 0, self.num_areas_d - 1)
        
        idx_all = []
        for i in range(self.num_areas_d):
            idx_all.append(list(np.where(distance_labels == i)[0]))
        return idx_all

    def update_proto_multi(self, pred_proto, feat, area):
        """EMA update of regional prototypes."""
        pred_label = pred_proto
        for i in range(self.num_classes):
            index_class = (pred_label == i)
            feat_i = feat[index_class].detach().cpu()
            if feat_i.shape[0] < self.min_feat:
                continue
            mean = torch.mean(feat_i, dim=0)
            self.proto[area][i] = self.alpha * self.proto[area][i] + (1 - self.alpha) * mean

    def prior_filter(self, pred, points):
        """Geometric Prior Filtering using Open3D (from D3CTTA)."""
        if points is None:
            return torch.ones(len(pred), dtype=torch.bool, device=pred.device)
            
        orig_device = pred.device
        pred = pred.argmax(1).detach().cpu().numpy()
        points_np = points.detach().cpu().numpy()
        
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
            normals = np.fabs(np.asarray(pcd.normals)[:, 2])
        except ImportError:
            # Fallback if open3d is missing (assume everything passes the geometric filter)
            print("Open3D not installed. Skipping geometric prior filter.")
            normals = np.ones(len(points_np)) * 0.5

        plane_norm_index = normals > 0.9
        manmade_norm_index = normals < 0.1
        # Map classes according to SemanticKITTI/NuScenes taxonomy (assuming Road=11, Sidewalk=12, Building=13)
        # Note: Original D3CTTA used custom 7-class taxonomy. We approximate for 17-class:
        # Plane (Road, Sidewalk, Parking, Other Ground) = 11, 12, 13, 14
        plane_pred_index = ((pred == 11) | (pred == 12) | (pred == 13) | (pred == 14))
        # Manmade (Building, Fence, Trunk, Pole) = 15, 16, 17... (approx)
        manmade_pred_index = (pred == 15) | (pred == 16) | (pred == 5)
        # Others (Cars, Pedestrians, Vegetation, etc) = everything else
        other_index = ~(plane_pred_index | manmade_pred_index)

        ground_index = plane_pred_index & plane_norm_index
        manmade_index = manmade_pred_index & manmade_norm_index

        valid_filter = ground_index | other_index | manmade_pred_index | manmade_index | plane_pred_index
        # Return a single boolean mask of geometrically valid points
        return torch.tensor(valid_filter, device=orig_device) if isinstance(valid_filter, np.ndarray) else torch.ones(len(pred), dtype=torch.bool, device=orig_device)

    def inference_update(self, h, predictions, xyz):
        device = h.device
        if xyz is not None and xyz.dim() == 4:
            xyz = xyz.permute(0, 2, 3, 1).reshape(-1, 3)
            
        if self.feat_source is None:
            return
            
        N = h.shape[0]
        valid_mask = (self.feat_source.sum(dim=1) != 0)
        valid_idx = torch.nonzero(valid_mask).squeeze()
        
        if valid_idx.numel() <= 20:
            return
            
        feat_valid = self.feat_source[valid_idx]
        h_valid = h[valid_idx]
        pred_source_valid = self.pred_source[valid_idx]
        xyz_valid = xyz[valid_idx] if xyz is not None else None
        
        # 1. Entropy Filtering (Top Ratio)
        ent = softmax_entropy(pred_source_valid)
        ent_threshold = torch.quantile(ent, 0.20)  # Keep top 20% most confident
        indices_ent = ent < ent_threshold

        # 2. Geometric Prior Filtering
        indices_filter = self.prior_filter(pred_source_valid, xyz_valid)
        
        # Combine filters
        combined_filter = indices_ent & indices_filter

        # 3. Distance-Aware Prototype Learning (DAPL)
        indices_parts = self.distance_partition(xyz_valid)
        pred_proto = torch.ones_like(pred_source_valid)
        
        for i in range(self.num_areas_d):
            indices = indices_parts[i]
            if len(indices) == 0:
                continue
                
            proto_i = self.proto[i].to(device)
            # Assign pseudo-labels based on regional prototypes
            pred_proto[indices] = feat_valid[indices] @ F.normalize(proto_i, dim=1).T
            
            # Update regional prototypes (only using points that passed the filters)
            valid_area_mask = torch.zeros(len(feat_valid), dtype=torch.bool, device=device)
            valid_area_mask[indices] = True
            update_mask = valid_area_mask & combined_filter
            
            if update_mask.sum() > 0:
                self.update_proto_multi(pred_source_valid.argmax(1)[update_mask], feat_valid[update_mask], i)

        # 4. KNN Consistency on the new DAPL pseudo-labels
        pred_proto_argmax = pred_proto.argmax(dim=1)
        keep_mask = torch.zeros(feat_valid.shape[0], dtype=torch.bool, device=device)
        chunk_size = 2000
        for i in range(0, feat_valid.shape[0], chunk_size):
            end = min(i + chunk_size, feat_valid.shape[0])
            xyz_chunk = xyz_valid[i:end]
            
            dists = torch.cdist(xyz_chunk.unsqueeze(0), xyz_valid.unsqueeze(0)).squeeze(0)
            _, knn_idx = torch.topk(dists, k=min(21, feat_valid.shape[0]), dim=1, largest=False)
            
            knn_preds = pred_proto_argmax[knn_idx]
            center_preds = pred_proto_argmax[i:end].unsqueeze(1)
            consistency = (knn_preds == center_preds).float().mean(dim=1)
            
            keep_mask[i:end] = consistency > 0.8
            
        final_indices = keep_mask & combined_filter
        
        # 5. Recursive Ridge Regression Update
        h_filtered = h_valid[final_indices]
        pred_filtered = pred_proto_argmax[final_indices]
        
        if h_filtered.shape[0] > 0:
            self.G_d[self.domain_id] = self.G_d[self.domain_id].to(h_filtered.device)
            self.C_d[self.domain_id] = self.C_d[self.domain_id].to(h_filtered.device)
            self.G_d[self.domain_id] += h_filtered.T @ h_filtered
            y_one_hot = F.one_hot(pred_filtered, num_classes=self.num_classes).float().to(h_filtered.device)
            self.C_d[self.domain_id] += h_filtered.T @ y_one_hot
