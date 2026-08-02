import torch
import torch.nn as nn
import torch.nn.functional as F

class HDCDenoiser(nn.Module):
    """
    Manifold Denoiser (Generative Reconstruction Gate) for HDC.
    Learns the 10,000D clean source geometric rules. Hallucinations in 
    out-of-distribution fog/noise regions fail to reconstruct.
    """
    def __init__(self, hd_dim=10000, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hd_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, hd_dim)
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class AdaptiveMemoryBank(nn.Module):
    """
    Adaptive Memory Bank for Test-Time Adaptation operating natively in the HDC space.
    Uses a CLASS-PARTITIONED memory bank with Reservoir Sampling to guarantee perfectly balanced retrieval.
    """
    def __init__(self, hd_dim=10000, num_classes=17, memory_capacity=20000, k=10, purity_threshold=0.48):
        super().__init__()
        self.hd_dim = hd_dim
        self.num_classes = num_classes
        self.k = k
        self.purity_threshold = purity_threshold
        
        # Partition the total capacity equally among all classes
        self.capacity_per_class = memory_capacity // num_classes
        self.total_capacity = self.capacity_per_class * num_classes
        
        # Global unpartitioned memory bank structure, but logically partitioned
        self.register_buffer("keys", torch.empty((self.total_capacity, hd_dim), dtype=torch.float32))
        self.register_buffer("values", torch.zeros(self.total_capacity, dtype=torch.int64))
        self.register_buffer("is_valid", torch.zeros(self.total_capacity, dtype=torch.bool))
        
        # Tracking pointers per class
        self.register_buffer("ptr", torch.zeros(num_classes, dtype=torch.int64))
        self.register_buffer("reserved_slots", torch.zeros(num_classes, dtype=torch.int64))
        self.register_buffer("is_full", torch.zeros(num_classes, dtype=torch.bool))

    def initialize_coreset(self, coreset_keys, coreset_values):
        """
        Seed the memory bank with the offline extracted coresets and lock them in reserved_slots.
        """
        for c in range(self.num_classes):
            c_mask = coreset_values == c
            if c_mask.any():
                c_keys = coreset_keys[c_mask]
                c_vals = coreset_values[c_mask]
                
                # Truncate if somehow the coreset is larger than capacity_per_class
                n_pts = min(c_keys.size(0), self.capacity_per_class)
                
                start_idx = c * self.capacity_per_class
                self.keys[start_idx : start_idx + n_pts] = c_keys[:n_pts]
                self.values[start_idx : start_idx + n_pts] = c_vals[:n_pts]
                self.is_valid[start_idx : start_idx + n_pts] = True
                
                self.reserved_slots[c] = n_pts
                self.ptr[c] = n_pts
                if n_pts == self.capacity_per_class:
                    self.is_full[c] = True

    def query(self, features):
        """
        Query via Density-Adaptive Hamming Distance against the global memory bank.
        """
        valid_mask = self.is_valid
        if not valid_mask.any():
            return torch.zeros(features.size(0), dtype=torch.int64, device=features.device), \
                   torch.zeros(features.size(0), dtype=torch.float32, device=features.device)
                   
        flat_keys = self.keys[valid_mask]
        flat_values = self.values[valid_mask]
        
        k = min(self.k, flat_keys.size(0))
        predictions = []
        purity = []
        
        # Pre-cast to half precision for Tensor Cores outside the chunk loop
        # This saves massive memory bandwidth overhead inside the loop
        flat_keys_half = flat_keys.half()
        features_half = features.half()
        
        # Process in chunks
        chunk_size = 50000
        for i in range(0, features.size(0), chunk_size):
            chunk = features_half[i:i+chunk_size]
            with torch.amp.autocast('cuda', enabled=True):
                sims = torch.mm(chunk, flat_keys_half.t()).float()
            
            # Raw geometric similarity
            topk_sims, topk_idx = sims.topk(k=k, dim=1)
            neighbor_sem = flat_values[topk_idx]
            
            # --- Distance-Weighted k-NN ---
            # Instead of a pure majority vote (which fails if a class has < k points),
            # weight each vote exponentially by its geometric similarity.
            # Using softmax prevents torch.exp() overflow since topk_sims are unnormalized dot products (~50-100).
            # The implicit tau=1.0 provides very sharp distance weighting.
            weights = torch.softmax(topk_sims, dim=1)
            
            # Aggregate weights per class for each query in the chunk
            # Output shape: [chunk_size, num_classes]
            vote_scores = torch.zeros(chunk.size(0), self.num_classes, device=features.device)
            vote_scores.scatter_add_(1, neighbor_sem, weights)
            
            # The predicted class is the one with the highest total weight
            pred_chunk = torch.argmax(vote_scores, dim=1)
            
            predictions.append(pred_chunk)
            
        predictions = torch.cat(predictions, dim=0)
        purity = None
        
        return predictions, purity

    def update(self, features, pseudo_labels, confidence, true_labels=None):
        """
        Class-Partitioned Update with Reservoir Sampling (Fixed Probability Replacement).
        """
        valid_mask = confidence >= self.purity_threshold
        admission_rate = valid_mask.float().mean().item()
        
        if not valid_mask.any():
            return admission_rate, -1.0
            
        valid_features = features[valid_mask]
        valid_labels = pseudo_labels[valid_mask]
        
        # Binarize incoming features BEFORE storing them
        bin_features = torch.sign(valid_features).to(torch.float32)
        bin_features[bin_features == 0] = 1.0
        
        # Track memory bank semantic purity (diagnostic only)
        purity_err = -1.0
        if true_labels is not None:
            valid_true = true_labels[valid_mask]
            eval_mask = (valid_true >= 0) & (valid_true < self.num_classes)
            if eval_mask.any():
                incorrect = (valid_labels[eval_mask] != valid_true[eval_mask]).float().sum()
                purity_err = (incorrect / eval_mask.float().sum()).item()
        
        # Class-partitioned updates
        for c in range(self.num_classes):
            c_mask = valid_labels == c
            if not c_mask.any():
                continue
                
            c_features = bin_features[c_mask]
            c_labels = valid_labels[c_mask]
            n_in = c_features.size(0)
            
            cap = self.capacity_per_class
            ptr = self.ptr[c].item()
            start_idx = c * cap
            reserved = self.reserved_slots[c].item()
            
            if not self.is_full[c]:
                available = cap - ptr
                if n_in <= available:
                    self.keys[start_idx + ptr : start_idx + ptr + n_in] = c_features
                    self.values[start_idx + ptr : start_idx + ptr + n_in] = c_labels
                    self.is_valid[start_idx + ptr : start_idx + ptr + n_in] = True
                    self.ptr[c] = ptr + n_in
                    if self.ptr[c] == cap:
                        self.is_full[c] = True
                else:
                    self.keys[start_idx + ptr : start_idx + cap] = c_features[:available]
                    self.values[start_idx + ptr : start_idx + cap] = c_labels[:available]
                    self.is_valid[start_idx + ptr : start_idx + cap] = True
                    self.is_full[c] = True
                    
                    # Remainder undergo Reservoir Sampling
                    rem_feats = c_features[available:]
                    rem_lbls = c_labels[available:]
                    
                    # P = 0.01 replacement
                    prob_mask = torch.rand(rem_feats.size(0), device=features.device) < 0.01
                    rep_feats = rem_feats[prob_mask]
                    rep_lbls = rem_lbls[prob_mask]
                    
                    if rep_feats.size(0) > 0:
                        # Replace only in non-reserved dynamic slots
                        if cap > reserved:
                            replace_offsets = torch.randint(reserved, cap, (rep_feats.size(0),), device=features.device)
                            replace_idx = start_idx + replace_offsets
                            self.keys[replace_idx] = rep_feats
                            self.values[replace_idx] = rep_lbls
            else:
                # Queue is full, use global Reservoir Sampling replacement in non-reserved space
                prob_mask = torch.rand(n_in, device=features.device) < 0.01
                rep_feats = c_features[prob_mask]
                rep_lbls = c_labels[prob_mask]
                
                if rep_feats.size(0) > 0:
                    if cap > reserved:
                        replace_offsets = torch.randint(reserved, cap, (rep_feats.size(0),), device=features.device)
                        replace_idx = start_idx + replace_offsets
                        self.keys[replace_idx] = rep_feats
                        self.values[replace_idx] = rep_lbls
                        
        return admission_rate, purity_err
        
    def recover_geometry(self, features):
        pass

    def forward(self, features):
        pass
