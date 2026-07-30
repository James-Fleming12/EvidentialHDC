import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMemoryBank(nn.Module):
    """
    Adaptive Memory Bank for Test-Time Adaptation operating natively in the HDC space.
    First Iteration: Binary Buffer, Hamming Query, and Graph-Pruned Update.
    """
    def __init__(self, hd_dim=10000, num_classes=17, memory_capacity=10000, k=10, purity_threshold=0.8):
        super().__init__()
        self.hd_dim = hd_dim
        self.num_classes = num_classes
        self.memory_capacity = memory_capacity
        self.k = k
        self.purity_threshold = purity_threshold
        
        # 1-Bit Binary Buffer
        # We store keys as float32 bipolar {-1, 1} to simulate binarization using fast PyTorch mm.
        self.register_buffer("keys", torch.empty((0, hd_dim), dtype=torch.float32))
        self.register_buffer("values", torch.empty((0,), dtype=torch.int64))
        self.ptr = 0
        self.is_full = False
        
    def query(self, features):
        """
        Query via Hamming Distance.
        """
        if self.keys.size(0) == 0:
            # Fallback if memory bank is completely empty
            return torch.zeros(features.size(0), dtype=torch.int64, device=features.device), \
                   torch.zeros(features.size(0), dtype=torch.float32, device=features.device)
                   
        # Binarize incoming features
        bin_features = torch.sign(features)
        bin_features[bin_features == 0] = 1.0 
        
        # Convert bank to float32 for fast matmul (simulating XOR bitcount hardware)
        # Dot product of bipolar vectors gives exact topological preservation
        sims = torch.mm(bin_features, self.keys.t())
        
        # Get k nearest neighbors
        k = min(self.k, self.keys.size(0))
        topk_sims, topk_idx = sims.topk(k=k, dim=1)
        
        neighbor_sem = self.values[topk_idx]
        
        # Majority voting
        predictions = torch.mode(neighbor_sem, dim=1).values
        
        # Compute purity (fraction of neighbors that agree with the majority vote)
        agreements = (neighbor_sem == predictions.unsqueeze(1)).float()
        purity = agreements.mean(dim=1)
        
        return predictions, purity

    def update(self, features, pseudo_labels, confidence):
        """
        Graph-Pruned Update.
        confidence here is the 'purity' returned from query().
        """
        # Graph Pruning: Only admit points whose neighborhood graph is highly pure
        valid_mask = confidence >= self.purity_threshold
        admission_rate = valid_mask.float().mean().item()
        
        if not valid_mask.any():
            return admission_rate
            
        valid_features = features[valid_mask]
        valid_labels = pseudo_labels[valid_mask]
        
        # Binarize before storing
        bin_features = torch.sign(valid_features)
        bin_features[bin_features == 0] = 1.0
        
        n_incoming = valid_features.size(0)
        
        if self.keys.size(0) < self.memory_capacity:
            # Still filling up
            space_left = self.memory_capacity - self.keys.size(0)
            if n_incoming <= space_left:
                self.keys = torch.cat([self.keys, bin_features], dim=0)
                self.values = torch.cat([self.values, valid_labels], dim=0)
            else:
                self.keys = torch.cat([self.keys, bin_features[:space_left]], dim=0)
                self.values = torch.cat([self.values, valid_labels[:space_left]], dim=0)
                self.is_full = True
                self.ptr = 0
                # Recursive call to handle the rest via FIFO
                self.update(valid_features[space_left:], valid_labels[space_left:], confidence[valid_mask][space_left:])
        else:
            # FIFO Queue logic
            if n_incoming >= self.memory_capacity:
                # Completely overwrite
                self.keys = bin_features[-self.memory_capacity:]
                self.values = valid_labels[-self.memory_capacity:]
                self.ptr = 0
            else:
                end_idx = self.ptr + n_incoming
                if end_idx <= self.memory_capacity:
                    self.keys[self.ptr:end_idx] = bin_features
                    self.values[self.ptr:end_idx] = valid_labels
                    self.ptr = (self.ptr + n_incoming) % self.memory_capacity
                else:
                    overflow = end_idx - self.memory_capacity
                    chunk1 = n_incoming - overflow
                    self.keys[self.ptr:] = bin_features[:chunk1]
                    self.values[self.ptr:] = valid_labels[:chunk1]
                    
                    self.keys[:overflow] = bin_features[chunk1:]
                    self.values[:overflow] = valid_labels[chunk1:]
                    self.ptr = overflow
        
        return admission_rate
                    
    def recover_geometry(self, features):
        pass

    def forward(self, features):
        pass
