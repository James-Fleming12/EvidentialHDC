import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMemoryBank(nn.Module):
    """
    Adaptive Memory Bank for Test-Time Adaptation operating natively in the HDC space.
    Uses a class-balanced FIFO Buffer to prevent majority-class collapse.
    """
    def __init__(self, hd_dim=10000, num_classes=17, memory_capacity=10000, k=10, purity_threshold=0.8):
        super().__init__()
        self.hd_dim = hd_dim
        self.num_classes = num_classes
        self.k = k
        self.purity_threshold = purity_threshold
        self.capacity_per_class = memory_capacity // num_classes
        
        # We store keys as float32 bipolar {-1, 1} to simulate binarization using PyTorch mm.
        self.register_buffer("keys", torch.empty((num_classes, self.capacity_per_class, hd_dim), dtype=torch.float32))
        self.register_buffer("class_sizes", torch.zeros(num_classes, dtype=torch.int64))
        self.register_buffer("class_ptrs", torch.zeros(num_classes, dtype=torch.int64))
        
    def query(self, features):
        """
        Query via Hamming Distance against the flat memory bank.
        """
        valid_keys = []
        valid_values = []
        for c in range(self.num_classes):
            size = self.class_sizes[c].item()
            if size > 0:
                valid_keys.append(self.keys[c, :size])
                valid_values.append(torch.full((size,), c, dtype=torch.int64, device=self.keys.device))
                
        if len(valid_keys) == 0:
            # Fallback if memory bank is completely empty
            return torch.zeros(features.size(0), dtype=torch.int64, device=features.device), \
                   torch.zeros(features.size(0), dtype=torch.float32, device=features.device)
                   
        flat_keys = torch.cat(valid_keys, dim=0)
        flat_values = torch.cat(valid_values, dim=0)
        
        # Binarize incoming query
        bin_features = torch.sign(features).to(torch.float32)
        bin_features[bin_features == 0] = 1.0 
        
        k = min(self.k, flat_keys.size(0))
        predictions = []
        purity = []
        
        # Process in chunks using float16 for massive speedup
        chunk_size = 50000
        for i in range(0, bin_features.size(0), chunk_size):
            chunk = bin_features[i:i+chunk_size]
            sims = torch.mm(chunk.half(), flat_keys.t().half()).float()
            
            topk_sims, topk_idx = sims.topk(k=k, dim=1)
            neighbor_sem = flat_values[topk_idx]
            
            pred_chunk = torch.mode(neighbor_sem, dim=1).values
            predictions.append(pred_chunk)
            
            agreements = (neighbor_sem == pred_chunk.unsqueeze(1)).float()
            purity.append(agreements.mean(dim=1))
            
        predictions = torch.cat(predictions, dim=0)
        purity = torch.cat(purity, dim=0)
        
        return predictions, purity

    def update(self, features, pseudo_labels, confidence):
        """
        Graph-Pruned Update with Class-Balanced FIFO.
        """
        # Graph Pruning: Only admit highly confident points
        valid_mask = confidence >= self.purity_threshold
        admission_rate = valid_mask.float().mean().item()
        
        if not valid_mask.any():
            return admission_rate
            
        valid_features = features[valid_mask]
        valid_labels = pseudo_labels[valid_mask]
        
        bin_features = torch.sign(valid_features).to(torch.float32)
        bin_features[bin_features == 0] = 1.0
        
        cap = self.capacity_per_class
        for c in range(self.num_classes):
            c_mask = valid_labels == c
            c_points = bin_features[c_mask]
            n_in = c_points.size(0)
            if n_in == 0:
                continue
                
            ptr = self.class_ptrs[c].item()
            
            if n_in >= cap:
                # Randomly sample to avoid spatial bias
                perm = torch.randperm(n_in, device=c_points.device)[:cap]
                self.keys[c] = c_points[perm]
                self.class_ptrs[c] = 0
                self.class_sizes[c] = cap
            else:
                end_idx = ptr + n_in
                if end_idx <= cap:
                    self.keys[c, ptr:end_idx] = c_points
                    self.class_ptrs[c] = end_idx % cap
                    self.class_sizes[c] = min(cap, self.class_sizes[c].item() + n_in)
                else:
                    overflow = end_idx - cap
                    chunk1 = n_in - overflow
                    self.keys[c, ptr:] = c_points[:chunk1]
                    self.keys[c, :overflow] = c_points[chunk1:]
                    self.class_ptrs[c] = overflow
                    self.class_sizes[c] = cap
                    
        return admission_rate
        
    def recover_geometry(self, features):
        pass

    def forward(self, features):
        pass
